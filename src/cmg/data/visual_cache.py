from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


def resolve_visual_feature_cache_dir(project_root: str | Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None or str(raw_path).strip() == '':
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path


def resolve_sample_visual_feature_cache_path(
    project_root: str | Path,
    cache_dir: str | Path,
    sample_id: str,
) -> Path:
    root = resolve_visual_feature_cache_dir(project_root, cache_dir)
    if root is None:
        raise ValueError('visual feature cache directory is not configured.')
    return root / f'{sample_id}.npz'


def write_sample_visual_feature_cache(
    project_root: str | Path,
    cache_dir: str | Path,
    sample_id: str,
    window_ids: list[str],
    visual_features: np.ndarray,
) -> Path:
    path = resolve_sample_visual_feature_cache_path(project_root, cache_dir, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        window_ids=np.asarray(window_ids, dtype=np.str_),
        visual_features=np.asarray(visual_features, dtype=np.float32),
    )
    return path


@lru_cache(maxsize=512)
def load_sample_visual_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    cache_path = Path(path)
    with np.load(cache_path, allow_pickle=False) as payload:
        return {
            'window_ids': payload['window_ids'].astype(np.str_).copy(),
            'visual_features': payload['visual_features'].astype(np.float32).copy(),
        }