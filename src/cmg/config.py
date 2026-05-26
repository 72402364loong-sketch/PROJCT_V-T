from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def sync_tactile_model_config(data_config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    from cmg.data.tactile import normalize_tactile_input_axes, tactile_input_dim_for_axes

    axes = normalize_tactile_input_axes(data_config.get('tactile_input_axes'))
    input_dim = tactile_input_dim_for_axes(axes)
    synced = dict(model_config)
    tactile_config = dict(synced.get('tactile', {}))
    tactile_config['input_dim'] = input_dim
    tactile_config['num_taxels'] = 12
    tactile_config['axis_dim'] = max(1, input_dim // 12)
    synced['tactile'] = tactile_config
    return synced
