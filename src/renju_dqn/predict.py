"""Q-value inference for a trained checkpoint: encode a board, run the ResNet DQN, and
pick a move by argmax over legal cells (or epsilon-greedy, for evaluation purposes).
"""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

from .board_encoder import encode_board, encode_legal_move_mask
from .model import RenjuResNetDQN
from .rules import BOARD_CELLS, idx_to_rc, infer_player
from .utils import select_device


def parse_board_csv(board_csv: str) -> list[int]:
    values = [int(value.strip()) for value in board_csv.split(",") if value.strip()]
    if len(values) != BOARD_CELLS:
        raise ValueError(f"Expected {BOARD_CELLS} board cells, got {len(values)}.")
    return values


def load_board(cfg: DictConfig) -> list[int]:
    if cfg.predict.board_csv:
        return parse_board_csv(cfg.predict.board_csv)
    if cfg.predict.board_path:
        content = Path(cfg.predict.board_path).read_text(encoding="utf-8").strip()
        return parse_board_csv(content)
    raise ValueError("Set predict.board_csv or predict.board_path for inference.")


def build_model_from_checkpoint(
    checkpoint: dict, fallback_cfg: DictConfig | None = None
) -> RenjuResNetDQN:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is not None:
        model_cfg = checkpoint_config["model"]
        exploration = checkpoint_config.get("train", {}).get("exploration", "epsilon_greedy")
    elif fallback_cfg is not None:
        model_cfg = fallback_cfg.model
        exploration = OmegaConf.select(fallback_cfg, "train.exploration", default="epsilon_greedy")
    else:
        raise ValueError("Checkpoint has no embedded config and no fallback_cfg was given.")

    model = RenjuResNetDQN(
        in_channels=model_cfg["in_channels"],
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        head_channels=model_cfg["head_channels"],
        num_move_labels=model_cfg["num_move_labels"],
        dueling=model_cfg["dueling"],
        noisy=exploration == "noisy_net",
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def select_move_from_q(
    q_values: torch.Tensor,
    legal_mask: torch.Tensor,
    epsilon: float = 0.0,
    generator: torch.Generator | None = None,
) -> int:
    """Argmax over `q_values` restricted to `legal_mask`; with probability `epsilon`,
    pick a uniformly random legal move instead (for epsilon-greedy evaluation)."""
    if not bool(legal_mask.any()):
        raise ValueError("No legal moves available for the provided board.")

    if epsilon > 0.0 and torch.rand((), generator=generator).item() < epsilon:
        legal_indices = legal_mask.nonzero(as_tuple=True)[0]
        choice = int(torch.randint(0, legal_indices.numel(), (1,), generator=generator).item())
        return int(legal_indices[choice].item())

    masked = q_values.masked_fill(~legal_mask, float("-inf"))
    return int(masked.argmax().item())


def predict_from_checkpoint(cfg: DictConfig) -> None:
    if not cfg.predict.checkpoint_path:
        raise ValueError("Set predict.checkpoint_path to a saved checkpoint.")

    board = load_board(cfg)
    device = select_device(cfg.train.device)
    checkpoint = torch.load(cfg.predict.checkpoint_path, map_location=device, weights_only=False)
    model = build_model_from_checkpoint(checkpoint, fallback_cfg=cfg).to(device)
    model.eval()

    # The board CSV carries no move history, so the last-move plane is left blank.
    player = infer_player(board)
    state = encode_board(board, player, last_move=None).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state).squeeze(0)

    if cfg.predict.apply_legal_mask:
        legal_mask = encode_legal_move_mask(board).to(device)
        best_index = select_move_from_q(q_values, legal_mask, epsilon=cfg.predict.epsilon)
        q_values = q_values.masked_fill(~legal_mask, float("-inf"))
    else:
        best_index = int(q_values.argmax().item())

    row, col = idx_to_rc(best_index)
    print(
        f"predicted_index={best_index} row={row} col={col} q_value={q_values[best_index].item():.6f}"
    )

    top_k = min(cfg.predict.top_k, q_values.numel())
    values, indices = torch.topk(q_values, k=top_k)
    for rank, (value, index) in enumerate(
        zip(values.tolist(), indices.tolist(), strict=True), start=1
    ):
        r, c = idx_to_rc(index)
        print(f"top{rank}_index={index} row={r} col={c} q_value={value:.6f}")
