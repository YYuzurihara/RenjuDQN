"""ResNet-based Deep Q-Network mapping a board tensor to per-cell Q-values."""

from __future__ import annotations

import torch
from torch import nn

from .rules import BOARD_SIZE


class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian weight noise (Fortunato et al., 2018,
    "Noisy Networks for Exploration"). Injects learned, per-parameter noise into the
    layer's weights so the network's own forward pass provides exploration, as a
    drop-in alternative to epsilon-greedy action selection.

    Noise is applied only while `self.training` is True (mirroring dropout/batchnorm),
    falling back to the mean weights in eval mode for a deterministic forward pass.
    `RenjuResNetDQN.set_noise_training` toggles this independently of the rest of the
    model's train()/eval() mode, since callers that need batchnorm in eval mode (e.g.
    self-play's single-sample inference) still want this layer's noise switched on.
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        bound = 1.0 / (self.in_features**0.5)
        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / (self.in_features**0.5))
        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(self.sigma_init / (self.out_features**0.5))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        noise = torch.randn(size)
        return noise.sign() * noise.abs().sqrt()

    def reset_noise(self) -> None:
        device = self.weight_mu.device
        epsilon_in = self._scale_noise(self.in_features).to(device)
        epsilon_out = self._scale_noise(self.out_features).to(device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return nn.functional.linear(x, weight, bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class RenjuResNetDQN(nn.Module):
    """Stem -> stacked residual blocks -> Q-value head over `num_move_labels` cells.

    `dueling` splits the head into a scalar state value `V(s)` and a per-action
    advantage `A(s,a)`, combined as `Q = V + (A - mean(A))` to fix the well-known
    identifiability issue of a plain `V + A` sum.

    `noisy` swaps the output head's `nn.Linear` layers for `NoisyLinear`, giving the
    network its own source of exploration noise as an alternative to epsilon-greedy
    action selection (call `reset_noise()` to resample it).
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_blocks: int,
        head_channels: int,
        num_move_labels: int,
        board_size: int = BOARD_SIZE,
        dueling: bool = False,
        noisy: bool = False,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.noisy = noisy
        linear = NoisyLinear if noisy else nn.Linear

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])
        self.head_conv = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        head_dim = head_channels * board_size * board_size
        if dueling:
            self.value_head = linear(head_dim, 1)
            self.advantage_head = linear(head_dim, num_move_labels)
        else:
            self.q_head = linear(head_dim, num_move_labels)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def noisy_sigma_norms(self) -> dict[str, float]:
        """Mean |sigma| per `NoisyLinear` submodule, keyed by its attribute name (e.g.
        "q_head" or "value_head"/"advantage_head"); empty if `noisy=False`.

        `NoisyLinear.weight_sigma`/`bias_sigma` are plain learned parameters with no
        lower-bound clamp, so gradient descent is free to shrink them toward zero if that
        happens to reduce TD loss -- silently collapsing self-play exploration without any
        other visible symptom. Tracking this alongside td_loss distinguishes "converged" from
        "stopped exploring".
        """
        norms: dict[str, float] = {}
        for name, module in self.named_modules():
            if isinstance(module, NoisyLinear):
                sigma = torch.cat(
                    [module.weight_sigma.detach().flatten(), module.bias_sigma.detach().flatten()]
                )
                norms[name] = sigma.abs().mean().item()
        return norms

    def set_noise_training(self, mode: bool = True) -> None:
        """Toggle NoisyLinear's noise application independent of the rest of the model's
        train()/eval() mode -- callers that need batchnorm in eval mode (single-sample
        self-play inference) call this after `.eval()` to still get exploration noise.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.train(mode)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(state)
        hidden = self.blocks(hidden)
        hidden = self.head_conv(hidden)
        hidden = hidden.flatten(start_dim=1)

        if self.dueling:
            value = self.value_head(hidden)
            advantage = self.advantage_head(hidden)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.q_head(hidden)
