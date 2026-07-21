from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from cmg.data.preprocess import build_samples_table, build_windows_table
from cmg.data.sidecar import load_window_sidecar
from cmg.data.windowing import compute_phase_label_at_timestamp, label_window


ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = ROOT / 'data' / 'splits' / 'split_unseen_fixed_test_obj004_obj007_v1.yaml'
SPLIT_VERSION = 'split_unseen_fixed_test_obj004_obj007_v1'


def _samples():
    return build_samples_table(
        ROOT,
        split_path=SPLIT_PATH,
        split_version=SPLIT_VERSION,
        reference_duration_sec=0.75,
    )


def _build_a2w_windows(*, write_sidecar_cache: bool = False, cache_root: str | Path = 'processed/cache'):
    samples = _samples()
    sample = samples.loc[
        samples['object_id'].eq('OBJ001')
        & samples['direction'].eq('A2W')
        & samples['trial_result'].eq('stable')
    ].head(1)
    return build_windows_table(
        sample,
        ROOT,
        video_fps=30.0,
        tactile_dt=0.039,
        window_size_sec=1.0,
        window_stride_sec=5.0,
        interface_overlap_sec=0.25,
        stable_margin_sec=0.25,
        short_tail_min_sec=0.5,
        num_frames_per_window=8,
        tactile_points_per_window=26,
        write_sidecar_cache=write_sidecar_cache,
        policy_timestamp_anchor='window_end',
        causal_only=True,
        sidecar_cache_relative_root=cache_root,
        target_aggregation_sec=0.08,
        reference_duration_sec=0.75,
        phase_label_mode='policy_timestamp',
    )


def test_a2w_phase_order_is_air_interface_water() -> None:
    assert compute_phase_label_at_timestamp(0.5, 1.0, 2.0, source_phase='Air', target_phase='Water') == 'Air'
    assert compute_phase_label_at_timestamp(1.5, 1.0, 2.0, source_phase='Air', target_phase='Water') == 'Interface'
    assert compute_phase_label_at_timestamp(2.5, 1.0, 2.0, source_phase='Air', target_phase='Water') == 'Water'

    before = label_window(0.0, 0.5, 1.0, 2.0, 0.25, 0.25, source_phase='Air', target_phase='Water')
    after = label_window(2.5, 3.0, 1.0, 2.0, 0.25, 0.25, source_phase='Air', target_phase='Water')
    assert (before.phase_label, before.stable_phase) == ('Air', 'Air')
    assert (after.phase_label, after.stable_phase) == ('Water', 'Water')


def test_samples_derive_split_direction_and_reference_contract() -> None:
    samples = _samples()
    assert len(samples) == 564
    assert samples['split'].value_counts().to_dict() == {
        'train': 314,
        'test': 158,
        'val': 92,
    }
    assert 'excluded' not in set(samples['split'])
    assert samples.groupby('physical_object_uid')['split'].nunique().max() == 1
    assert samples.groupby('direction')['has_reference_candidate'].sum().to_dict() == {'A2W': 262, 'W2A': 262}
    assert samples.loc[samples['trial_result'].eq('fail'), 'policy_supervision_eligible'].sum() == 0
    assert samples.loc[samples['trial_result'].eq('fail'), 'has_reference_candidate'].sum() == 0
    assert set(samples.loc[samples['direction'].eq('A2W'), 'force_baseline_mode']) == {'none'}
    assert set(samples.loc[samples['direction'].eq('W2A'), 'force_baseline_mode']) == {'pre_contact_median'}


def test_a2w_windows_propagate_direction_split_and_reference() -> None:
    windows = _build_a2w_windows()
    assert not windows.empty
    assert set(windows['direction']) == {'A2W'}
    assert set(windows['source_phase']) == {'Air'}
    assert set(windows['target_phase']) == {'Water'}
    assert set(windows['split']) == {'train'}
    assert set(windows['split_version']) == {SPLIT_VERSION}
    assert windows['reference_eligible'].eq(1).all()
    assert windows['policy_supervision_eligible'].eq(1).all()
    ordered_phases = windows.sort_values('policy_timestamp')['phase_label'].tolist()
    assert ordered_phases[0] == 'Air'
    assert ordered_phases[-1] == 'Water'


def test_sidecar_contains_bidirectional_contract() -> None:
    with TemporaryDirectory() as temp_dir:
        windows = _build_a2w_windows(write_sidecar_cache=True, cache_root=Path(temp_dir))
        payload = load_window_sidecar(ROOT, windows.iloc[0]['sidecar_cache_path'])
        required = {
            'direction',
            'direction_index',
            'source_medium',
            'target_medium',
            'reference_medium',
            'split',
            'split_version',
            'physical_object_uid',
            'reference_eligible',
            'policy_supervision_eligible',
        }
        assert required.issubset(payload)
        assert payload['direction'].item() == 'A2W'
        assert int(payload['direction_index'].item()) == 1
        assert bool(payload['reference_eligible'].item())


def test_bidirectional_v4_config_is_versioned_and_isolated() -> None:
    path = ROOT / 'configs' / 'data' / 'policy_20hz_bidirectional_v4.yaml'
    with path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    assert config['schema_version'] == 'bidirectional-causal-v4'
    assert config['dataset_version'] == 'bidirectional_v1'
    assert config['split_version'] == SPLIT_VERSION
    assert config['processed_dir'].endswith('policy_20hz_bidirectional_v1_fixed_test')
    assert config['reference']['medium'] == 'source_medium'
    assert config['force_baseline'] == {'W2A': 'pre_contact_median', 'A2W': 'none'}
