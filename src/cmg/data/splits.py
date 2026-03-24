from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from cmg.config import load_yaml



def resolve_split_config(path: str | Path) -> dict:
    return load_yaml(path)



def resolve_sample_ids(
    samples: pd.DataFrame,
    split_path: str | Path,
    subset: str,
) -> list[str]:
    config = resolve_split_config(split_path)
    kind = config['kind']
    if kind == 'object_level_unseen':
        object_ids = config[f'{subset}_object_ids']
        return (
            samples.loc[samples['object_id'].isin(object_ids), 'sample_id']
            .sort_values()
            .tolist()
        )
    if kind == 'sample_level_random':
        object_ids = config['object_ids']
        ratios = config['ratios']
        filtered = samples.loc[samples['object_id'].isin(object_ids)].copy()
        rng = random.Random(int(config.get('seed', 7)))
        subsets: dict[str, list[str]] = {'train': [], 'val': [], 'test': []}
        for _, group in filtered.groupby('object_id'):
            sample_ids = sorted(group['sample_id'].tolist())
            rng.shuffle(sample_ids)
            count = len(sample_ids)
            train_end = max(1, int(round(count * ratios['train'])))
            val_end = min(count, train_end + max(1, int(round(count * ratios['val']))))
            subsets['train'].extend(sample_ids[:train_end])
            subsets['val'].extend(sample_ids[train_end:val_end])
            subsets['test'].extend(sample_ids[val_end:])
        return sorted(subsets[subset])
    raise ValueError(f'Unsupported split kind: {kind}')
