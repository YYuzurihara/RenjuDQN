"""Shared utility helpers (device/seed/config/MLflow), architecture-independent."""

from __future__ import annotations

import random
from pathlib import Path

import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(configured_device: str) -> torch.device:
    if configured_device != "auto":
        return torch.device(configured_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def flatten_dict(data: dict, prefix: str) -> dict[str, str | int | float | bool]:
    flattened: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        composite_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, composite_key))
        elif isinstance(value, (str, int, float, bool)):
            flattened[composite_key] = value
        elif value is None:
            flattened[composite_key] = "null"
        else:
            flattened[composite_key] = str(value)
    return flattened


def flatten_config(cfg: DictConfig, prefix: str = "") -> dict[str, str | int | float | bool]:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        return {}

    flattened: dict[str, str | int | float | bool] = {}
    for key, value in data.items():
        composite_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, composite_key))
        elif isinstance(value, (str, int, float, bool)):
            flattened[composite_key] = value
        elif value is None:
            flattened[composite_key] = "null"
        else:
            flattened[composite_key] = str(value)
    return flattened


def ensure_mlflow_experiment(tracking_uri: str, experiment_name: str, artifact_root: str) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return

    artifact_uri = Path(artifact_root).resolve().as_uri()
    client.create_experiment(experiment_name, artifact_location=artifact_uri)
