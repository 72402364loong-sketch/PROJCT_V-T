from __future__ import annotations

from pathlib import Path

import torch

from cmg.config import (
    resolve_training_configs,
    validate_direction_training_contract,
    validate_initialization_source_contract,
)
from cmg.models import CrossMediumSystem
from cmg.training import (
    audit_model_parameters,
    audit_optimizer_groups,
    build_optimizer,
    enforce_frozen_modules_eval,
    freeze_modules,
)
from scripts.validate_model_phase_m3 import _tiny_direction_model_config


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    'stage39a_direction_adapter_warmup.yaml',
    'stage39b_bidirectional_medium.yaml',
    'stage39c_bidirectional_policy.yaml',
    'stage39d_bidirectional_joint.yaml',
)


def _resolved(stage_name: str):
    return resolve_training_configs(ROOT, ROOT / 'configs' / 'stages' / stage_name)


def test_stage39_configs_resolve_frozen_data_model_and_train_sources() -> None:
    for stage_name in STAGES:
        data, model, train, stage, sources = _resolved(stage_name)
        assert data['dataset_version'] == 'bidirectional_v1'
        assert model['direction_conditioning']['enabled'] is True
        assert train['sampling_mode'] == 'direction_object_aware'
        assert train['loss_reduction']['mode'] == 'sample_direction_macro'
        assert stage['split'] == 'data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml'
        assert sources['data']['project_path'] == 'configs/data/policy_20hz_bidirectional_v4.yaml'
        assert sources['model']['project_path'] == 'configs/model/direction_conditioned_v1.yaml'
        assert sources['train']['project_path'] == 'configs/train/direction_conditioned_v1.yaml'


def test_stage39_formal_training_is_ready_after_e0_baseline() -> None:
    data, model, train, stage, _ = _resolved(STAGES[0])
    report = validate_direction_training_contract(
        ROOT,
        data,
        model,
        train,
        stage,
        initialization_path=ROOT / 'runs/stage38j_f20_v3_causal/checkpoints/best.pt',
    )
    assert report['validated'] is True
    assert report['readiness_status'] == 'ready'
    assert report['guard_enabled'] is True


def test_stage39_ready_contract_does_not_require_pending_e0_override() -> None:
    data, model, train, stage, _ = _resolved(STAGES[0])
    report = validate_direction_training_contract(
        ROOT,
        data,
        model,
        train,
        stage,
        initialization_path=ROOT / 'runs/stage38j_f20_v3_causal/checkpoints/best.pt',
    )
    assert report['validated'] is True
    assert report['readiness_status'] == 'ready'
    assert report['guard_enabled'] is True


def test_ready_stage_requires_enabled_guard_and_baseline() -> None:
    data, model, train, stage, _ = _resolved(STAGES[1])
    stage = dict(stage)
    stage['training_readiness'] = {'status': 'ready'}
    stage['w2a_retention_guard'] = {'enabled': False}
    try:
        validate_direction_training_contract(
            ROOT,
            data,
            model,
            train,
            stage,
            initialization_path=ROOT / 'placeholder.pt',
        )
    except ValueError as error:
        assert 'enable w2a_retention_guard' in str(error)
    else:
        raise AssertionError('Ready stage accepted a disabled W2A guard.')


def test_stage39_freeze_contract_matches_expected_trainable_prefixes() -> None:
    for stage_name in STAGES:
        _, _, _, stage, _ = _resolved(stage_name)
        model = CrossMediumSystem(_tiny_direction_model_config(12))
        freeze_modules(
            model,
            stage['freeze_modules'],
            strict=True,
            lock_eval=True,
            train_mode_module_names=stage.get('frozen_train_mode_modules', []),
        )
        trainable = audit_model_parameters(model)['trainable_parameter_names']
        expected = stage['expected_trainable_prefixes']
        assert trainable
        assert all(any(name == prefix or name.startswith(prefix + '.') for prefix in expected) for name in trainable)
        assert all(any(name == prefix or name.startswith(prefix + '.') for name in trainable) for prefix in expected)


def test_stage39_optimizer_groups_apply_declared_learning_rates() -> None:
    _, _, train, stage, _ = _resolved('stage39b_bidirectional_medium.yaml')
    model = CrossMediumSystem(_tiny_direction_model_config(12))
    freeze_modules(model, stage['freeze_modules'])
    groups = {group['name']: group for group in audit_optimizer_groups(model, build_optimizer(model, train))}
    assert groups['base']['learning_rate'] == 1e-5
    assert groups['base_direction_embedding']['learning_rate'] == 2e-5
    assert groups['base_medium_direction_adapter']['learning_rate'] == 5e-5


def test_stage39a_frozen_gru_retains_cudnn_backward_state() -> None:
    if not torch.cuda.is_available():
        return
    _, _, _, stage, _ = _resolved('stage39a_direction_adapter_warmup.yaml')
    model = CrossMediumSystem(_tiny_direction_model_config(12)).cuda()
    report = freeze_modules(
        model,
        stage['freeze_modules'],
        strict=True,
        lock_eval=True,
        train_mode_module_names=stage.get('frozen_train_mode_modules', []),
    )
    model.train()
    enforce_frozen_modules_eval(model)
    assert report['train_mode_module_names'] == ['medium_head']
    assert model.medium_head.training is True
    assert model.visual_encoder.training is False
    sequence = torch.randn(2, 5, model.medium_head.gru.input_size, device='cuda', requires_grad=True)
    lengths = torch.tensor([5, 4], device='cuda')
    logits, _, _, _ = model.medium_head(sequence, lengths)
    logits.sum().backward()
    assert sequence.grad is not None
    assert torch.isfinite(sequence.grad).all()


def test_stage39_initialization_source_stage_is_strict() -> None:
    _, _, _, stage, _ = _resolved('stage39c_bidirectional_policy.yaml')
    validate_initialization_source_contract(stage, {'stage_name': 'stage39b_bidirectional_medium'})
    try:
        validate_initialization_source_contract(stage, {'stage_name': 'stage39a_direction_adapter_warmup'})
    except ValueError as error:
        assert 'expected_source_stage' in str(error)
    else:
        raise AssertionError('Wrong cross-stage initialization source was accepted.')


def test_stage39_selection_and_guard_metrics_are_direction_explicit() -> None:
    for stage_name in STAGES:
        _, _, _, stage, _ = _resolved(stage_name)
        assert stage['selection_metric'].endswith('_macro_direction')
        assert stage['w2a_retention_guard']['metric'].endswith('_w2a')
        assert stage['w2a_retention_guard']['enabled'] is True
        assert stage['w2a_retention_guard']['baseline_value'] > 0.0
        assert stage['w2a_retention_guard']['max_relative_degradation'] == 0.05
        assert stage['training_readiness']['status'] == 'ready'
        assert stage['initialization']['required'] is True
