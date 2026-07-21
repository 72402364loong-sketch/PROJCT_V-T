from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from cmg.config import load_yaml


OBJECT_LEVEL_SUBSETS = ('train', 'val', 'test')


def validate_object_level_split(config: dict, *, source: str | Path | None = None) -> None:
    label = str(source or config.get('name', '<split>'))
    object_ids_by_subset: dict[str, list[str]] = {}
    for subset in OBJECT_LEVEL_SUBSETS:
        key = f'{subset}_object_ids'
        if key not in config:
            raise ValueError(f'{label}: missing required field {key!r}.')
        object_ids = [str(value) for value in config[key]]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError(f'{label}: duplicated object_id within {key}.')
        object_ids_by_subset[subset] = object_ids

    for index, left_subset in enumerate(OBJECT_LEVEL_SUBSETS):
        for right_subset in OBJECT_LEVEL_SUBSETS[index + 1 :]:
            overlap = sorted(set(object_ids_by_subset[left_subset]) & set(object_ids_by_subset[right_subset]))
            if overlap:
                raise ValueError(
                    f'{label}: object leakage between {left_subset} and {right_subset}: {overlap}.'
                )

    required_test_object_ids = {str(value) for value in config.get('required_test_object_ids', [])}
    missing_required = sorted(required_test_object_ids - set(object_ids_by_subset['test']))
    if missing_required:
        raise ValueError(f'{label}: required test objects are missing from test_object_ids: {missing_required}.')



def resolve_split_config(path: str | Path) -> dict:
    config = load_yaml(path)
    if config.get('kind') == 'object_level_unseen':
        validate_object_level_split(config, source=path)
    return config



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


def derive_split_labels(samples: pd.DataFrame, split_path: str | Path) -> pd.Series:
    config = resolve_split_config(split_path)
    if config.get('kind') != 'object_level_unseen':
        raise ValueError('Processed split derivation requires an object_level_unseen split.')

    subset_by_object: dict[str, str] = {}
    for subset in OBJECT_LEVEL_SUBSETS:
        for object_id in config[f'{subset}_object_ids']:
            subset_by_object[str(object_id)] = subset

    labels = samples['object_id'].astype(str).map(subset_by_object).fillna('excluded')
    if 'physical_object_uid' in samples.columns:
        audit = pd.DataFrame({
            'physical_object_uid': samples['physical_object_uid'].astype(str),
            'split': labels,
        })
        leakage = audit.groupby('physical_object_uid')['split'].nunique()
        leaking_uids = sorted(leakage.loc[leakage > 1].index.tolist())
        if leaking_uids:
            raise ValueError(f'physical_object_uid leakage across derived splits: {leaking_uids}')
    return labels.astype('string')
