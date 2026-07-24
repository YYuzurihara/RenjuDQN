#!/usr/bin/env python3
r"""Export a trained RenjuResNetDQN checkpoint to ONNX plus browser metadata.

Usage:
    uv run python ./scripts/export_onnx.py \
        --checkpoint artifacts/checkpoints/best_model.pt \
        --output docs/renju_dqn.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_dqn.model import RenjuResNetDQN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a saved PyTorch checkpoint.")
    parser.add_argument("--output", required=True, help="Path to write the ONNX model.")
    parser.add_argument(
        "--metadata-output",
        default=None,
        help="Optional path to write metadata JSON. Defaults to <output>.metadata.json.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to target. Defaults to 18.",
    )
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint: dict) -> tuple[RenjuResNetDQN, dict]:
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError("Checkpoint does not contain embedded config data.")

    model_cfg = checkpoint_config["model"]
    model = RenjuResNetDQN(
        in_channels=model_cfg["in_channels"],
        channels=model_cfg["channels"],
        num_blocks=model_cfg["num_blocks"],
        head_channels=model_cfg["head_channels"],
        num_move_labels=model_cfg["num_move_labels"],
        dueling=model_cfg["dueling"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint_config


def build_sample_input(checkpoint_config: dict) -> torch.Tensor:
    model_cfg = checkpoint_config["model"]
    data_cfg = checkpoint_config.get("data", {})
    in_channels = int(model_cfg["in_channels"])
    board_size = int(data_cfg.get("board_size", 15))
    return torch.zeros((1, in_channels, board_size, board_size), dtype=torch.float32)


def build_metadata(checkpoint_config: dict) -> dict:
    model_cfg = checkpoint_config["model"]
    data_cfg = checkpoint_config.get("data", {})
    board_size = int(data_cfg.get("board_size", 15))
    board_cells = board_size * board_size

    return {
        "model_type": "RenjuResNetDQN",
        "input_name": "state",
        "output_name": "q_values",
        "input_dtype": "float32",
        "output_dtype": "float32",
        "board_size": board_size,
        "board_cells": board_cells,
        "in_channels": int(model_cfg["in_channels"]),
        "num_move_labels": int(model_cfg["num_move_labels"]),
        "supports_batch": False,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    metadata_path = (
        Path(args.metadata_output).resolve()
        if args.metadata_output
        else output_path.with_suffix(f"{output_path.suffix}.metadata.json")
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model, checkpoint_config = build_model_from_checkpoint(checkpoint)
    sample_input = build_sample_input(checkpoint_config)

    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = exc.name or "onnx dependency"
        raise SystemExit(
            "ONNX export dependencies are missing. "
            f"Install `{missing}` and retry. "
            "For this project, run either "
            "`uv add onnx onnxscript` to add them permanently, or "
            "`uv run --with onnx --with onnxscript python ./scripts/export_onnx.py ...` "
            "to use an isolated ephemeral environment."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_program = torch.onnx.export(
        model,
        (sample_input,),
        opset_version=args.opset,
        input_names=["state"],
        output_names=["q_values"],
        dynamo=True,
        external_data=False,
        dynamic_shapes=None,
        optimize=True,
        verify=False,
    )
    onnx_program.save(str(output_path))

    metadata = build_metadata(checkpoint_config)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"onnx_model={output_path}")
    print(f"metadata_json={metadata_path}")


if __name__ == "__main__":
    main()
