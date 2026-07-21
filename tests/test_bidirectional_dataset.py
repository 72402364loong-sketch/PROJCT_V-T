from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from cmg.data.dataset import CrossMediumSequenceDataset, sequence_collate_fn


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed' / 'policy_20hz_bidirectional_v1_fixed_test'
SPLIT_PATH = ROOT / 'data' / 'splits' / 'split_unseen_fixed_test_obj004_obj007_v1.yaml'
SPLIT_VERSION = 'split_unseen_fixed_test_obj004_obj007_v1'


def test_reference_window_indices_follow_a2w_source_medium() -> None:
    dataset = object.__new__(CrossMediumSequenceDataset)
    dataset.reference_force_window_count = 3
    windows = [
        {
            'phase_label': phase,
            'is_stable_mask': phase != 'Interface',
            'reference_medium': 'air',
            'reference_interval_start': 1.0,
            'reference_interval_end': 1.75,
            'policy_timestamp': timestamp,
        }
        for phase, timestamp in [
            ('Air', 0.95),
            ('Air', 1.05),
            ('Air', 1.50),
            ('Interface', 1.75),
            ('Water', 2.50),
        ]
    ]
    assert dataset._resolve_reference_window_indices(windows) == [1, 2]


def test_fixed_reference_interval_uses_exact_raw_time_bounds() -> None:
    dataset = object.__new__(CrossMediumSequenceDataset)
    dataset.reference_force_window_count = 3
    dataset.reference_force_statistic = 'median'
    dataset.reference_force_duration_sec = 0.75
    windows = [
        {
            'phase_label': 'Air',
            'is_stable_mask': True,
            'reference_medium': 'air',
            'reference_interval_start': 1.0,
            'reference_interval_end': 1.75,
            'policy_timestamp': 1.25,
            'tactile_start_idx': 0,
            'tactile_end_idx': 2,
        }
    ]
    time_axis = np.arange(0.0, 3.0, 0.25, dtype=np.float32)
    force_curve = time_axis * 10.0
    value, has_reference, _ = dataset._compute_sample_reference_force(
        windows,
        force_curve,
        time_axis=time_axis,
        reference_start_time=1.0,
        reference_end_time=1.75,
        strict_interval=True,
    )
    expected = np.median(force_curve[(time_axis >= 1.0) & (time_axis < 1.75)])
    assert has_reference
    assert np.isclose(value, expected)


def _select_smoke_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(PROCESSED / 'samples.csv')
    stable_pair = samples.loc[
        samples['object_id'].eq('OBJ001')
        & samples['trial_result'].eq('stable')
    ].groupby('direction', group_keys=False).head(1)
    fail = samples.loc[
        samples['object_id'].eq('OBJ001')
        & samples['direction'].eq('W2A')
        & samples['trial_result'].eq('fail')
    ].head(1)
    selected_samples = pd.concat([stable_pair, fail], ignore_index=True)
    sample_ids = set(selected_samples['sample_id'].astype(str))

    matching_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(PROCESSED / 'windows.csv', chunksize=25000):
        selected = chunk.loc[chunk['sample_id'].astype(str).isin(sample_ids)]
        if not selected.empty:
            matching_chunks.append(selected)
    all_windows = pd.concat(matching_chunks, ignore_index=True)

    selected_windows: list[pd.DataFrame] = []
    sample_by_id = selected_samples.set_index('sample_id')
    for sample_id, group in all_windows.groupby('sample_id'):
        group = group.sort_values('policy_timestamp')
        sample = sample_by_id.loc[sample_id]
        if sample['trial_result'] == 'stable':
            reference = group.loc[
                group['policy_timestamp'].ge(float(sample['reference_start_time']))
                & group['policy_timestamp'].lt(float(sample['reference_end_time']))
            ].tail(2)
            interface = group.loc[group['phase_label'].eq('Interface')].head(2)
            target = group.tail(1)
            selected_windows.append(pd.concat([reference, interface, target]).drop_duplicates('window_id'))
        else:
            selected_windows.append(group.head(3))
    windows = pd.concat(selected_windows, ignore_index=True).sort_values(['sample_id', 'policy_timestamp'])
    return selected_samples, windows


def _make_smoke_dataset(temp_root: Path) -> CrossMediumSequenceDataset:
    samples, windows = _select_smoke_tables()
    samples_path = temp_root / 'samples.csv'
    windows_path = temp_root / 'windows.csv'
    samples.to_csv(samples_path, index=False)
    windows.to_csv(windows_path, index=False)
    return CrossMediumSequenceDataset(
        project_root=ROOT,
        split_path=SPLIT_PATH,
        subset='train',
        num_frames_per_window=2,
        image_size=64,
        roi=None,
        tactile_dt=0.039,
        acdc_alpha=0.1,
        expert_alpha=0.1,
        normal_sign_table={
            'finger0': [1.0, 1.0, 1.0, 1.0],
            'finger1': [1.0, 1.0, 1.0, 1.0],
            'finger2': [1.0, 1.0, 1.0, 1.0],
        },
        tactile_points_per_window=8,
        tactile_input_axes=['z'],
        standardize_tactile=False,
        samples_path=samples_path,
        windows_path=windows_path,
        reference_force_window_count=3,
        reference_force_statistic='median',
        reference_force_duration_sec=0.75,
        reference_force_source='raw_tactile',
        target_aggregation='median',
        target_aggregation_sec=0.08,
        target_min_samples=2,
        target_fallback='latest_causal',
        expert_force_mode='local_reference_delta',
        expert_force_smoothing='zero_phase_ema',
        expert_force_baseline_mode='none',
        expert_force_baseline_window_sec=0.5,
        expert_force_interface_margin_sec=0.1,
        force_target_mode='signed_sum',
        attribute_taxonomy='coarse_v2',
        physical_attribute_table=ROOT / 'data' / 'annotations' / 'object_physical_attributes.csv',
        physical_attribute_norm_stats_path=temp_root / 'physical_attribute_norm_stats.json',
        schema_version='bidirectional-causal-v4',
        dataset_version='bidirectional_v1',
        split_version=SPLIT_VERSION,
        force_baseline_by_direction={'W2A': 'pre_contact_median', 'A2W': 'none'},
    )


def test_bidirectional_dataset_item_collate_and_fail_masks() -> None:
    with TemporaryDirectory() as temp_dir:
        dataset = _make_smoke_dataset(Path(temp_dir))
        items = {str(dataset.sample_records[index]['sample_id']): dataset[index] for index in range(len(dataset))}

        stable_items = [item for item in items.values() if item['trial_result'] == 'stable']
        assert {item['direction'] for item in stable_items} == {'W2A', 'A2W'}
        assert {item['direction_index'] for item in stable_items} == {0, 1}
        assert {item['reference_medium'] for item in stable_items} == {'water', 'air'}
        assert {item['force_baseline_mode'] for item in stable_items} == {'pre_contact_median', 'none'}
        assert all(item['reference_eligible'] for item in stable_items)
        assert all(np.isfinite(item['reference_forces']).all() for item in stable_items)

        fail_item = next(item for item in items.values() if item['trial_result'] == 'fail')
        assert fail_item['policy_supervision_eligible'] == 0
        assert fail_item['reference_eligible'] == 0
        for field in [
            'has_expert', 'has_control_target', 'has_reference',
            'reference_supervision_masks', 'delta_supervision_masks', 'quiet_supervision_masks',
        ]:
            assert not any(fail_item[field])
        assert not np.isfinite(np.asarray(fail_item['finger_control_force_targets'])).any()

        batch = sequence_collate_fn(stable_items)
        assert batch['direction_ids'].shape == (2,)
        assert set(batch['direction_ids'].tolist()) == {0, 1}
        assert set(batch['directions']) == {'W2A', 'A2W'}
        assert set(batch['source_media']) == {'water', 'air'}
        assert set(batch['reference_media']) == {'water', 'air'}
        assert batch['policy_supervision_eligible'].all()
        assert batch['reference_eligible'].all()
