"""DQN training loop: TD-error updates against a target network, fed by a replay
buffer that is warmed up from `mcts.cpp` self-play logs and then grown online via
epsilon-greedy self-play (Plan.md phase4).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from tqdm import tqdm

from .board_encoder import encode_board
from .dataset import ReplayDataset
from .env import SelectMove, play_self_play_game
from .model import RenjuResNetDQN
from .replay_buffer import ReplayBuffer
from .utils import ensure_mlflow_experiment, flatten_config, select_device, set_seed


def build_model(cfg: DictConfig) -> RenjuResNetDQN:
    return RenjuResNetDQN(
        in_channels=cfg.model.in_channels,
        channels=cfg.model.channels,
        num_blocks=cfg.model.num_blocks,
        head_channels=cfg.model.head_channels,
        num_move_labels=cfg.model.num_move_labels,
        dueling=cfg.model.dueling,
    )


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    if cfg.optimizer.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    if cfg.scheduler.name == "none":
        return None
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def epsilon_for_step(
    step: int, epsilon_start: float, epsilon_end: float, decay_steps: int
) -> float:
    """Linear anneal from `epsilon_start` (step 0) to `epsilon_end` (step >= decay_steps)."""
    if decay_steps <= 0:
        return epsilon_end
    fraction = min(1.0, step / decay_steps)
    return epsilon_start + fraction * (epsilon_end - epsilon_start)


def make_epsilon_greedy_policy(
    model: nn.Module,
    device: torch.device,
    epsilon: float,
    generator: torch.Generator,
) -> SelectMove:
    """Build a `select_move` callback for `env.play_self_play_game` at a fixed epsilon."""

    def select_move(board: list[int], player: int, prev_move: int | None, mask: list[bool]) -> int:
        legal_indices = [index for index, ok in enumerate(mask) if ok]
        if torch.rand((), generator=generator).item() < epsilon:
            choice = int(torch.randint(0, len(legal_indices), (1,), generator=generator).item())
            return legal_indices[choice]

        state = encode_board(board, player, prev_move).unsqueeze(0).to(device)
        legal_mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
        model.eval()
        with torch.no_grad():
            q_values = model(state).squeeze(0)
        q_values = q_values.masked_fill(~legal_mask_tensor, float("-inf"))
        return int(q_values.argmax().item())

    return select_move


def dqn_update(
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    replay_buffer: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
    gradient_clip_norm: float | None,
    generator: torch.Generator,
) -> tuple[float, float]:
    """One TD-error gradient step; returns `(loss, mean_q_taken)`."""
    state, action, reward, next_state, done, next_legal_mask = replay_buffer.sample(
        batch_size, generator=generator
    )
    state = state.to(device)
    action = action.to(device)
    reward = reward.to(device)
    next_state = next_state.to(device)
    done = done.to(device)
    next_legal_mask = next_legal_mask.to(device)

    online_model.train()
    q_values = online_model(state)
    q_taken = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_model(next_state)
        next_q = next_q.masked_fill(~next_legal_mask, float("-inf"))
        next_q_max = next_q.max(dim=1).values
        next_q_max = torch.where(done, torch.zeros_like(next_q_max), next_q_max)
        # next_state is encoded from the opponent's perspective, so their best value is negated.
        target = reward - gamma * next_q_max

    loss = criterion(q_taken, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if gradient_clip_norm is not None and gradient_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), gradient_clip_norm)
    optimizer.step()

    return loss.item(), q_taken.mean().item()


def train_model(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = select_device(cfg.train.device)
    generator = torch.Generator().manual_seed(cfg.seed)

    online_model = build_model(cfg).to(device)
    target_model = build_model(cfg).to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad_(False)

    optimizer = build_optimizer(online_model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.SmoothL1Loss()

    replay_buffer = ReplayBuffer(capacity=cfg.train.replay_buffer_capacity, gamma=cfg.train.gamma)
    warmup_dataset = ReplayDataset(cfg.data.path, gamma=cfg.train.gamma, max_rows=cfg.data.max_rows)
    replay_buffer.warmup_from_dataset(warmup_dataset)
    if len(replay_buffer) < cfg.train.batch_size:
        raise ValueError(
            f"Warmup buffer ({len(replay_buffer)} transitions) is smaller than "
            f"batch_size ({cfg.train.batch_size})."
        )

    output_root = Path(cfg.train.output_root)
    checkpoint_dir = output_root / cfg.train.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = config_dir / "resolved_config.yaml"
    OmegaConf.save(cfg, resolved_config_path, resolve=True)

    ensure_mlflow_experiment(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_root=cfg.mlflow.artifact_root,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    run_name = f"{cfg.mlflow.run_name_prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    best_epoch_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / cfg.train.checkpoint_name
    best_model_state: dict[str, torch.Tensor] | None = None
    global_step = 0

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(flatten_config(cfg))
        mlflow.log_artifact(str(resolved_config_path), artifact_path="configs")

        for epoch in range(1, cfg.train.max_epochs + 1):
            epoch_loss_total = 0.0
            epoch_q_total = 0.0
            epoch_updates = 0
            epoch_progress = tqdm(
                range(cfg.train.steps_per_epoch),
                desc=f"Epoch {epoch}/{cfg.train.max_epochs} [self-play]",
                leave=True,
                dynamic_ncols=True,
                file=sys.stdout,
            )

            for _ in epoch_progress:
                epsilon = epsilon_for_step(
                    global_step,
                    cfg.train.epsilon_start,
                    cfg.train.epsilon_end,
                    cfg.train.epsilon_decay_steps,
                )
                select_move = make_epsilon_greedy_policy(online_model, device, epsilon, generator)
                rows, winner = play_self_play_game(select_move)
                replay_buffer.push_game(rows, winner)

                for _ in range(cfg.train.updates_per_game):
                    loss, mean_q = dqn_update(
                        online_model,
                        target_model,
                        optimizer,
                        criterion,
                        replay_buffer,
                        cfg.train.batch_size,
                        cfg.train.gamma,
                        device,
                        cfg.train.gradient_clip_norm,
                        generator,
                    )
                    if scheduler is not None:
                        scheduler.step()

                    global_step += 1
                    epoch_loss_total += loss
                    epoch_q_total += mean_q
                    epoch_updates += 1

                    if global_step % cfg.train.target_update_interval == 0:
                        target_model.load_state_dict(online_model.state_dict())

                    if global_step % cfg.train.log_every_steps == 0:
                        mlflow.log_metric("train_step_td_loss", loss, step=global_step)
                        mlflow.log_metric("train_step_mean_q", mean_q, step=global_step)
                        mlflow.log_metric("epsilon", epsilon, step=global_step)

                epoch_progress.set_postfix(
                    td_loss=f"{epoch_loss_total / epoch_updates:.4f}",
                    mean_q=f"{epoch_q_total / epoch_updates:.4f}",
                    buffer=len(replay_buffer),
                    epsilon=f"{epsilon:.3f}",
                )

            epoch_progress.close()
            epoch_metrics = {
                "td_loss": epoch_loss_total / epoch_updates,
                "mean_q": epoch_q_total / epoch_updates,
            }
            mlflow.log_metric("train_td_loss", epoch_metrics["td_loss"], step=epoch)
            mlflow.log_metric("train_mean_q", epoch_metrics["mean_q"], step=epoch)
            mlflow.log_metric("replay_buffer_size", len(replay_buffer), step=epoch)

            is_best = epoch_metrics["td_loss"] < best_epoch_loss
            if is_best:
                best_epoch_loss = epoch_metrics["td_loss"]
                best_model_state = {
                    key: value.detach().cpu() for key, value in online_model.state_dict().items()
                }
                checkpoint = {
                    "model_state_dict": best_model_state,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                    "epoch": epoch,
                    "train_td_loss": best_epoch_loss,
                }
                torch.save(checkpoint, best_checkpoint_path)
                mlflow.log_metric("best_train_td_loss", best_epoch_loss, step=epoch)

            print(
                f"epoch={epoch} "
                f"train_td_loss={epoch_metrics['td_loss']:.4f} "
                f"train_mean_q={epoch_metrics['mean_q']:.4f} "
                f"best_train_td_loss={best_epoch_loss:.4f}",
                flush=True,
            )

            if is_best:
                print(
                    f"best_checkpoint_updated={best_checkpoint_path.resolve()} "
                    f"train_td_loss={best_epoch_loss:.4f}",
                    flush=True,
                )

        if best_model_state is None:
            raise RuntimeError("Training completed without producing a checkpoint.")

        mlflow.log_artifact(str(best_checkpoint_path), artifact_path="checkpoints")

        if cfg.mlflow.log_model:
            best_model = build_model(cfg)
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            mlflow.pytorch.log_model(best_model, name="model")

    print(f"best_checkpoint={best_checkpoint_path.resolve()}", flush=True)
