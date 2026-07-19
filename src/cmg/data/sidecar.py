from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def build_window_sidecar_relative_path(
    sample_id: str,
    window_id: str,
    *,
    cache_relative_root: str | Path = Path('processed') / 'cache',
) -> Path:
    return Path(cache_relative_root) / sample_id / f'{window_id}.npz'


def resolve_window_sidecar_path(project_root: str | Path, relative_path: str | Path) -> Path:
    root = Path(project_root)
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative
    if relative.parts and relative.parts[0] == 'data':
        return root / relative
    return root / 'data' / relative


def _normalize_np_payload(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value
        else:
            normalized[key] = np.asarray(value)
    return normalized


def write_window_sidecar(
    project_root: str | Path,
    sample_id: str,
    window_id: str,
    payload: dict[str, Any],
    *,
    cache_relative_root: str | Path = Path('processed') / 'cache',
) -> str:
    relative_path = build_window_sidecar_relative_path(
        sample_id,
        window_id,
        cache_relative_root=cache_relative_root,
    )
    absolute_path = resolve_window_sidecar_path(project_root, relative_path)
    absolute_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(absolute_path, **_normalize_np_payload(payload))
    return relative_path.as_posix()


def load_window_sidecar(project_root: str | Path, relative_path: str | Path) -> dict[str, np.ndarray]:
    absolute_path = resolve_window_sidecar_path(project_root, relative_path)
    with np.load(absolute_path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}
