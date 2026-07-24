"""Evaluation for a trained checkpoint: TD-error/mean-Q on replayed transitions, and
self-play win rate against a baseline opponent (Plan.md phase5). Neither metric is
loss/accuracy over classification labels -- the SL notion of "accuracy" has no DQN
equivalent.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch import nn

from .dataset import ReplayDataset
from .env import DRAW, play_self_play_game
from .predict import build_model_from_checkpoint
from .replay_buffer import ReplayBuffer
from .rules import BLACK, WHITE
from .train import make_epsilon_greedy_policy
from .utils import select_device


@torch.no_grad()
def evaluate_td_error(
    model: nn.Module,
    replay_buffer: ReplayBuffer,
    criterion: nn.Module,
    device: torch.device,
    gamma: float,
    num_batches: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Mean TD loss / mean taken-Q over `num_batches` sampled from `replay_buffer`.

    `model` serves as both online and target network: this checkpoint-evaluation
    entry point has no separate target-network snapshot to compare against, only the
    single saved checkpoint.
    """
    model.eval()
    total_loss = 0.0
    total_q = 0.0

    for _ in range(num_batches):
        state, action, reward, next_state, done, next_legal_mask = replay_buffer.sample(
            batch_size, generator=generator
        )
        state, action, reward = state.to(device), action.to(device), reward.to(device)
        next_state, done = next_state.to(device), done.to(device)
        next_legal_mask = next_legal_mask.to(device)

        q_values = model(state)
        q_taken = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        next_q = model(next_state)
        next_q = next_q.masked_fill(~next_legal_mask, float("-inf"))
        next_q_max = next_q.max(dim=1).values
        next_q_max = torch.where(done, torch.zeros_like(next_q_max), next_q_max)
        # next_state is encoded from the opponent's perspective, so their best value is negated.
        target = reward - gamma * next_q_max

        total_loss += criterion(q_taken, target).item()
        total_q += q_taken.mean().item()

    return {"td_loss": total_loss / num_batches, "mean_q": total_q / num_batches}


def evaluate_win_rate(
    model: nn.Module,
    device: torch.device,
    num_games: int,
    opponent_epsilon: float = 1.0,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Play `num_games`, alternating which color the greedy `model` plays, against an
    epsilon-greedy opponent (`opponent_epsilon=1.0`, the default, is a fully random
    baseline). Returns win/draw/loss rates from the greedy agent's perspective.
    """
    generator = generator if generator is not None else torch.Generator()
    agent_policy = make_epsilon_greedy_policy(model, device, epsilon=0.0, generator=generator)
    opponent_policy = make_epsilon_greedy_policy(
        model, device, epsilon=opponent_epsilon, generator=generator
    )

    wins = draws = losses = 0
    for game_index in range(num_games):
        agent_color = BLACK if game_index % 2 == 0 else WHITE

        def select_move(board, player, prev_move, mask, agent_color=agent_color):
            policy = agent_policy if player == agent_color else opponent_policy
            return policy(board, player, prev_move, mask)

        _, winner = play_self_play_game(select_move)
        if winner == DRAW:
            draws += 1
        elif winner == agent_color:
            wins += 1
        else:
            losses += 1

    total = wins + draws + losses
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "games": float(total),
    }


def evaluate_checkpoint(cfg: DictConfig) -> None:
    if not cfg.predict.checkpoint_path:
        raise ValueError("Set predict.checkpoint_path to a saved checkpoint.")

    device = select_device(cfg.train.device)
    checkpoint = torch.load(cfg.predict.checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, fallback_cfg=cfg).to(device)
    model.eval()

    generator = torch.Generator().manual_seed(cfg.seed)

    dataset = ReplayDataset(cfg.data.path, gamma=cfg.train.gamma, max_rows=cfg.data.max_rows)
    replay_buffer = ReplayBuffer(capacity=len(dataset), gamma=cfg.train.gamma)
    replay_buffer.warmup_from_dataset(dataset)

    td_metrics = evaluate_td_error(
        model,
        replay_buffer,
        nn.SmoothL1Loss(),
        device,
        gamma=cfg.train.gamma,
        num_batches=cfg.predict.num_eval_batches,
        batch_size=min(cfg.train.batch_size, len(replay_buffer)),
        generator=generator,
    )
    print(f"eval_td_loss={td_metrics['td_loss']:.6f} eval_mean_q={td_metrics['mean_q']:.6f}")

    win_metrics = evaluate_win_rate(
        model,
        device,
        num_games=cfg.predict.num_eval_games,
        opponent_epsilon=cfg.predict.eval_opponent_epsilon,
        generator=generator,
    )
    print(
        f"eval_win_rate={win_metrics['win_rate']:.4f} "
        f"eval_draw_rate={win_metrics['draw_rate']:.4f} "
        f"eval_loss_rate={win_metrics['loss_rate']:.4f} "
        f"games={int(win_metrics['games'])}"
    )
