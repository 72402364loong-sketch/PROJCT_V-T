from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cmg.config import load_yaml
from cmg.data.dataset import CrossMediumSequenceDataset
from cmg.data.windowing import compute_phase_label_at_timestamp
from scripts.diagnose_policy_timeshift import compute_timeshift_metrics


ROOT = Path(__file__).resolve().parents[1]


def test_f20_stage_inherits_policy_and_causal_data_config() -> None:
    config = load_yaml(ROOT / 'configs' / 'stages' / 'stage38j_f20_causal.yaml')
    assert config['data']['policy_rate_hz'] == 20
    assert config['data']['policy_timestamp_anchor'] == 'window_end'
    assert config['data']['target']['aggregation_sec'] == 0.08
    assert config['data']['reference']['duration_sec'] == 0.75
    assert config['model']['policy']['head_type'] == 'state_residual_per_finger_sign_specific'


def test_causal_target_uses_only_short_history() -> None:
    dataset = object.__new__(CrossMediumSequenceDataset)
    dataset.target_aggregation_sec = 0.08
    dataset.target_min_samples = 2
    dataset.target_aggregation = 'median'
    dataset.target_fallback = 'latest_causal'
    timestamps = np.arange(0.0, 0.201, 0.039, dtype=np.float32)
    curve = np.stack([timestamps * 10, timestamps * 20, timestamps * 30], axis=1)

    actual = dataset._compute_causal_target_vector(curve, timestamps, 0.10)
    expected = np.median(curve[(timestamps >= 0.02) & (timestamps <= 0.10)], axis=0)
    np.testing.assert_allclose(actual, expected)


def test_timeshift_diagnostic_finds_one_step_lag() -> None:
    rows = pd.DataFrame(
        {
            'sample_id': ['S'] * 5,
            'finger': ['finger0'] * 5,
            'policy_timestamp': [0.0, 0.05, 0.10, 0.15, 0.20],
            'target_delta': [0.0, 1.0, 2.0, 3.0, 4.0],
            'pred_delta': [0.0, 0.0, 1.0, 2.0, 3.0],
            'has_target': [1] * 5,
        }
    )
    metrics = compute_timeshift_metrics(rows, max_shift_steps=2, stride_sec=0.05)
    best = metrics.loc[metrics['delta_mae'].idxmin()]
    assert int(best['shift_steps']) == -1
    assert float(best['shift_ms']) == -50.0


def test_policy_timestamp_phase_labels_are_instantaneous() -> None:
    assert compute_phase_label_at_timestamp(0.99, 1.0, 2.0) == 'Water'
    assert compute_phase_label_at_timestamp(1.0, 1.0, 2.0) == 'Interface'
    assert compute_phase_label_at_timestamp(2.0, 1.0, 2.0) == 'Interface'
    assert compute_phase_label_at_timestamp(2.01, 1.0, 2.0) == 'Air'


def test_f20_v3_stage_uses_versioned_instantaneous_phase_data() -> None:
    config = load_yaml(ROOT / 'configs' / 'stages' / 'stage38j_f20_v3_medium_recalibration.yaml')
    assert config['data']['phase_label_mode'] == 'policy_timestamp'
    assert config['data']['windows_path'].endswith('policy_20hz_causal_v3/windows.csv')
    assert config['train']['phase_class_weights']['Interface'] == 4.0


def test_f20_v3_policy_stage_freezes_medium_and_selects_residual_quality() -> None:
    config = load_yaml(ROOT / 'configs' / 'stages' / 'stage38j_f20_v3_causal.yaml')
    assert config['data']['phase_label_mode'] == 'policy_timestamp'
    assert config['data']['windows_path'].endswith('policy_20hz_causal_v3/windows.csv')
    assert 'medium_head' in config['freeze_modules']
    assert config['selection_metric'] == 'finger_large_delta_balanced_score'
    assert config['policy_loss']['residual_stable_zero_weight'] == 3.0
