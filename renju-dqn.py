#!/usr/bin/env python3
"""Hydra entrypoint for Renju DQN training and inference."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))
    if cfg.mode == "train":
        from renju_dqn.train import train_model

        train_model(cfg)
        return
    if cfg.mode == "predict":
        raise NotImplementedError(
            "renju_dqn.predict is not implemented yet (Plan.md phase 5: inference/evaluation)."
        )
    raise ValueError(f"Unsupported mode: {cfg.mode}")


if __name__ == "__main__":
    main()
