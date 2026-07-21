from __future__ import annotations

from pathlib import Path

import pandas as pd

from cmg.data.splits import resolve_sample_ids, resolve_split_config, validate_object_level_split


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / 'data' / 'splits' / 'split_unseen_fixed_test_obj004_obj007_v1.yaml'


def test_all_split_configs_resolve() -> None:
    for split_path in sorted((ROOT / 'data' / 'splits').glob('*.yaml')):
        config = resolve_split_config(split_path)
        assert config['kind'] in {'object_level_unseen', 'sample_level_random'}


def test_fixed_holdout_object_membership() -> None:
    config = resolve_split_config(SPLIT_PATH)

    assert config['name'] == 'split_unseen_fixed_test_obj004_obj007_v1'
    assert set(config['required_test_object_ids']) == {'OBJ004', 'OBJ007'}
    assert set(config['train_object_ids']) == {
        'OBJ001',
        'OBJ002',
        'OBJ003',
        'OBJ009',
        'OBJ011',
        'OBJ012',
        'OBJ013',
        'OBJ016',
        'OBJ017',
        'OBJ018',
    }
    assert set(config['val_object_ids']) == {'OBJ005', 'OBJ010', 'OBJ014'}
    assert set(config['test_object_ids']) == {'OBJ004', 'OBJ006', 'OBJ007', 'OBJ008', 'OBJ015'}


def test_required_test_object_constraint_is_enforced() -> None:
    config = resolve_split_config(SPLIT_PATH)
    broken = {**config, 'test_object_ids': ['OBJ004', 'OBJ006', 'OBJ008', 'OBJ015']}
    try:
        validate_object_level_split(broken, source='broken-test-split')
    except ValueError as exc:
        assert 'OBJ007' in str(exc)
    else:
        raise AssertionError('Missing required test object should invalidate the split.')


def test_fixed_holdout_keeps_both_directions_in_one_subset() -> None:
    samples = pd.read_csv(ROOT / 'data' / 'annotations' / 'sample_events.csv')
    subset_by_sample: dict[str, str] = {}
    for subset in ('train', 'val', 'test'):
        for sample_id in resolve_sample_ids(samples, SPLIT_PATH, subset=subset):
            assert sample_id not in subset_by_sample
            subset_by_sample[str(sample_id)] = subset

    main_samples = samples.copy()
    assert set(main_samples['object_id']) == {f'OBJ{index:03d}' for index in range(1, 19)}
    assert set(main_samples['sample_id']) == set(subset_by_sample)

    for object_id, group in main_samples.groupby('object_id'):
        subsets = {subset_by_sample[str(sample_id)] for sample_id in group['sample_id']}
        assert len(subsets) == 1, f'{object_id} crosses subsets: {sorted(subsets)}'

    fixed_test = main_samples.loc[main_samples['object_id'].isin({'OBJ004', 'OBJ007'})]
    assert set(fixed_test['direction']) == {'W2A', 'A2W'}
    assert {subset_by_sample[str(sample_id)] for sample_id in fixed_test['sample_id']} == {'test'}
