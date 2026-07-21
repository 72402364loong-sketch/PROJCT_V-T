from __future__ import annotations

from pathlib import Path

import torch

from cmg.config import load_yaml, resolve_training_configs
from cmg.direction_metrics import DirectionMetricAccumulator
from cmg.losses import compute_losses, direction_class_macro_sample_mean, direction_macro_sample_mean
from cmg.training import Trainer, evaluate_w2a_retention_guard


ROOT = Path(__file__).resolve().parents[1]


def _loss_batch(
    errors: torch.Tensor,
    *,
    direction_ids: torch.Tensor,
    window_mask: torch.Tensor,
    control_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch_size, windows = errors.shape
    if control_mask is None:
        control_mask = window_mask.clone()
    phase_labels = torch.ones(batch_size, windows, dtype=torch.long)
    batch = {
        'window_mask': window_mask,
        'stable_masks': torch.zeros_like(window_mask),
        'phase_labels': phase_labels,
        'fragility_label': torch.zeros(batch_size, dtype=torch.long),
        'geometry_label': torch.zeros(batch_size, dtype=torch.long),
        'surface_label': torch.zeros(batch_size, dtype=torch.long),
        'object_index': torch.arange(batch_size, dtype=torch.long),
        'sample_index': torch.arange(batch_size, dtype=torch.long),
        'stable_phases': torch.full((batch_size, windows), -1, dtype=torch.long),
        'direction_ids': direction_ids,
        'expert_forces': torch.zeros(batch_size, windows),
        'has_expert': control_mask,
        'has_control_target': control_mask,
        'control_force_targets': torch.zeros(batch_size, windows),
    }
    flat_windows = batch_size * windows
    outputs = {
        'medium_logits': torch.zeros(batch_size, windows, 3),
        'fragility_logits': torch.zeros(flat_windows, 2),
        'geometry_logits': torch.zeros(flat_windows, 2),
        'surface_logits': torch.zeros(flat_windows, 2),
        'z_v': torch.zeros(flat_windows, 2),
        'z_t': torch.zeros(flat_windows, 2),
        'z_content': torch.zeros(flat_windows, 2),
        'force_pred': errors.clone(),
    }
    return outputs, batch


def _compute(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    med_weight: float = 0.0,
    pol_weight: float = 0.0,
    reduction: dict | None = None,
) -> dict[str, torch.Tensor]:
    return compute_losses(
        outputs,
        batch,
        loss_weights={'med': med_weight, 'pol': pol_weight},
        policy_loss_config={'type': 'mse'},
        temperature_clip=0.07,
        temperature_inv=0.07,
        phase_class_weights=torch.ones(3),
        loss_reduction_config=reduction,
    )


def test_direction_macro_sample_mean_balances_directions() -> None:
    losses = torch.tensor([1.0, 3.0, 10.0, 10.0])
    directions = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    macro, parts = direction_macro_sample_mean(losses, directions)
    torch.testing.assert_close(parts['w2a'], torch.tensor(2.0))
    torch.testing.assert_close(parts['a2w'], torch.tensor(10.0))
    torch.testing.assert_close(macro, torch.tensor(6.0))


def test_medium_direction_macro_is_sample_first_not_window_pooled() -> None:
    outputs, batch = _loss_batch(
        torch.zeros(2, 3),
        direction_ids=torch.tensor([0, 1]),
        window_mask=torch.tensor([[True, False, False], [True, True, True]]),
    )
    outputs['medium_logits'][0, 0] = torch.tensor([0.0, 3.0, 0.0])
    outputs['medium_logits'][1] = torch.tensor([[3.0, 0.0, 0.0]] * 3)
    macro = _compute(
        outputs,
        batch,
        med_weight=1.0,
        reduction={'mode': 'sample_direction_macro'},
    )
    pooled = _compute(outputs, batch, med_weight=1.0)
    loss_w2a = torch.nn.functional.cross_entropy(outputs['medium_logits'][0, :1], torch.tensor([1]))
    loss_a2w = torch.nn.functional.cross_entropy(outputs['medium_logits'][1], torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(macro['med_w2a'], loss_w2a)
    torch.testing.assert_close(macro['med_a2w'], loss_a2w)
    torch.testing.assert_close(macro['med'], 0.5 * (loss_w2a + loss_a2w))
    assert not torch.allclose(macro['med'], pooled['med'])


def test_direction_class_macro_is_sample_then_class_then_direction_balanced() -> None:
    window_losses = torch.tensor(
        [
            [1.0, 1.0, 10.0, 100.0],
            [3.0, 3.0, 30.0, 300.0],
            [6.0, 6.0, 6.0, 6.0],
        ]
    )
    targets = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ]
    )
    macro, parts = direction_class_macro_sample_mean(
        window_losses,
        targets,
        torch.ones_like(targets, dtype=torch.bool),
        torch.tensor([0, 0, 1]),
        num_classes=3,
        require_all_directions=True,
        require_all_classes_per_direction=True,
    )
    # W2A: class means are 2, 20, 200 after equal sample averaging.
    torch.testing.assert_close(parts['w2a'], torch.tensor(74.0))
    torch.testing.assert_close(parts['a2w'], torch.tensor(6.0))
    torch.testing.assert_close(macro, torch.tensor(40.0))


def test_medium_warm_balanced_blends_original_and_class_macro_equally() -> None:
    outputs, batch = _loss_batch(
        torch.zeros(2, 4),
        direction_ids=torch.tensor([0, 1]),
        window_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    batch['phase_labels'] = torch.tensor([[0, 2, 2, 2], [0, 1, 2, 2]])
    outputs['medium_logits'] = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
    )
    losses = _compute(
        outputs,
        batch,
        med_weight=1.0,
        reduction={
            'mode': 'sample_direction_class_macro_blend',
            'original_direction_macro_weight': 0.5,
            'direction_class_macro_weight': 0.5,
        },
    )
    torch.testing.assert_close(
        losses['med'],
        0.5 * losses['med_original'] + 0.5 * losses['med_class_balanced'],
    )
    torch.testing.assert_close(
        losses['med_a2w'],
        0.5 * losses['med_original'].new_tensor(
            torch.nn.functional.cross_entropy(outputs['medium_logits'][1], batch['phase_labels'][1]).item()
        )
        + 0.5 * losses['med_class_balanced_a2w'],
    )


def test_policy_direction_macro_excludes_unsupervised_sample() -> None:
    outputs, batch = _loss_batch(
        torch.tensor([[1.0, 0.0, 0.0], [3.0, 3.0, 3.0], [2.0, 2.0, 0.0], [99.0, 99.0, 99.0]]),
        direction_ids=torch.tensor([0, 0, 1, 1]),
        window_mask=torch.tensor(
            [[True, False, False], [True, True, True], [True, True, False], [True, True, True]]
        ),
        control_mask=torch.tensor(
            [[True, False, False], [True, True, True], [True, True, False], [False, False, False]]
        ),
    )
    losses = _compute(
        outputs,
        batch,
        pol_weight=1.0,
        reduction={'mode': 'sample_direction_macro'},
    )
    torch.testing.assert_close(losses['pol_w2a'], torch.tensor(5.0))
    torch.testing.assert_close(losses['pol_a2w'], torch.tensor(4.0))
    torch.testing.assert_close(losses['pol'], torch.tensor(4.5))


def test_direction_macro_requires_explicit_direction_ids() -> None:
    outputs, batch = _loss_batch(
        torch.zeros(2, 1),
        direction_ids=torch.tensor([0, 1]),
        window_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    batch.pop('direction_ids')
    try:
        _compute(outputs, batch, med_weight=1.0, reduction={'mode': 'sample_direction_macro'})
    except RuntimeError as error:
        assert 'requires batch["direction_ids"]' in str(error)
    else:
        raise AssertionError('Direction-macro loss accepted a missing direction id.')


def test_direction_metrics_emit_w2a_a2w_and_macro_values() -> None:
    batch_size, windows, fingers = 2, 2, 3
    batch = {
        'direction_ids': torch.tensor([0, 1]),
        'window_mask': torch.ones(batch_size, windows, dtype=torch.bool),
        'phase_labels': torch.tensor([[0, 1], [1, 2]]),
        'stable_masks': torch.tensor([[True, False], [False, True]]),
        'finger_control_force_targets': torch.zeros(batch_size, windows, fingers),
        'finger_delta_force_targets': torch.ones(batch_size, windows, fingers) * 150.0,
        'has_finger_control_target': torch.ones(batch_size, windows, fingers, dtype=torch.bool),
        'delta_supervision_masks': torch.tensor([[False, True], [True, False]]),
    }
    control_pred = torch.zeros(batch_size, windows, fingers)
    control_pred[0, 1] = 1.0
    control_pred[1, 0] = 3.0
    delta_pred = torch.ones(batch_size, windows, fingers) * 150.0
    delta_pred[1, 0] = -150.0
    outputs = {
        'medium_logits': torch.nn.functional.one_hot(batch['phase_labels'], 3).float() * 5.0,
        'force_interface_delta': torch.zeros(batch_size, windows),
        'interface_gate': torch.ones(batch_size, windows),
        'finger_force_pred': control_pred,
        'finger_force_interface_delta': delta_pred,
    }
    accumulator = DirectionMetricAccumulator(finger_large_delta_threshold=100.0)
    accumulator.update(outputs, batch)
    metrics = accumulator.finalize()
    assert metrics['medium_f1_interface_w2a'] == 1.0
    assert metrics['medium_f1_interface_a2w'] == 1.0
    assert metrics['medium_water_interface_hmean_w2a'] == 1.0
    assert metrics['medium_water_interface_hmean_a2w'] == 0.0
    assert metrics['medium_water_interface_hmean_macro_direction'] == 0.5
    assert metrics['finger_control_interface_mae_w2a'] == 1.0
    assert metrics['finger_control_interface_mae_a2w'] == 3.0
    assert metrics['finger_control_interface_mae_macro_direction'] == 2.0
    assert metrics['finger_large_delta_wrong_sign_rate_w2a'] == 0.0
    assert metrics['finger_large_delta_wrong_sign_rate_a2w'] == 1.0
    assert metrics['direction_macro_complete'] is True


def test_w2a_retention_guard_supports_min_and_max_metrics() -> None:
    min_pass = evaluate_w2a_retention_guard(
        {'finger_control_interface_mae_w2a': 104.9, 'finger_control_interface_count_w2a': 10},
        {
            'enabled': True,
            'metric': 'finger_control_interface_mae_w2a',
            'baseline_value': 100.0,
            'mode': 'min',
            'max_relative_degradation': 0.05,
        },
    )
    min_fail = evaluate_w2a_retention_guard(
        {'finger_control_interface_mae_w2a': 105.1, 'finger_control_interface_count_w2a': 10},
        {
            'enabled': True,
            'metric': 'finger_control_interface_mae_w2a',
            'baseline_value': 100.0,
            'mode': 'min',
            'max_relative_degradation': 0.05,
        },
    )
    max_pass = evaluate_w2a_retention_guard(
        {'medium_f1_interface_w2a': 0.76, 'medium_window_count_w2a': 10},
        {
            'enabled': True,
            'metric': 'medium_f1_interface_w2a',
            'baseline_value': 0.8,
            'mode': 'max',
            'max_relative_degradation': 0.05,
        },
    )
    assert min_pass['passed'] is True
    assert min_fail['passed'] is False
    assert max_pass['passed'] is True
    assert min_pass['threshold'] == 105.0


def test_macro_selection_rejects_incomplete_direction_metrics() -> None:
    trainer = object.__new__(Trainer)
    trainer.selection_metric = 'finger_control_interface_mae_macro_direction'
    metrics = {
        'finger_control_interface_mae_macro_direction': 12.0,
        'finger_control_interface_mae_macro_direction_complete': False,
    }
    try:
        trainer._selection_value(metrics)
    except RuntimeError as error:
        assert 'incomplete' in str(error)
    else:
        raise AssertionError('Macro selection accepted an incomplete direction metric.')


def test_bidirectional_train_config_enables_m3_reduction_contract() -> None:
    config = load_yaml(ROOT / 'configs' / 'train' / 'direction_conditioned_v1.yaml')
    assert config['loss_reduction']['mode'] == 'sample_direction_macro'
    assert config['w2a_retention_guard']['max_relative_degradation'] == 0.05


def test_stage39b_warm_balanced_config_is_isolated_and_direction_explicit() -> None:
    _, _, train, stage, _ = resolve_training_configs(
        ROOT,
        ROOT / 'configs/stages/stage39b_v2_warm_balanced_medium.yaml',
    )
    assert stage['name'] == 'stage39b_v2_warm_balanced_medium'
    assert stage['initialization']['expected_source_stage'] == 'stage39a_direction_adapter_warmup'
    assert train['loss_reduction']['mode'] == 'sample_direction_class_macro_blend'
    assert train['loss_reduction']['original_direction_macro_weight'] == 0.5
    assert train['loss_reduction']['direction_class_macro_weight'] == 0.5
    assert stage['selection_metric'] == 'medium_water_interface_hmean_macro_direction'
