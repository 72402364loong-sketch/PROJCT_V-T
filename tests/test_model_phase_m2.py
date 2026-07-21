from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from cmg.config import load_yaml
from cmg.models.modules import DirectionEmbedding, PolicyHead, ResidualFiLMAdapter
from cmg.models.system import CrossMediumSystem
from cmg.online import OnlineInferenceStub
from cmg.training import load_model_weights


ROOT = Path(__file__).resolve().parents[1]


def _tiny_model_config(*, direction_enabled: bool) -> dict:
    return {
        'visual': {
            'backbone': 'small_cnn',
            'pretrained': False,
            'freeze_backbone': False,
            'use_lora': False,
            'token_dim': 8,
            'hidden_dim': 8,
            'proj_dim': 4,
            'adapter_rank': 2,
            'max_windows_per_encode': 8,
            'pooling': {
                'score_hidden_dim': 4,
                'pre_norm': True,
                'output_post_norm': True,
                'use_mean_residual': True,
                'dropout': 0.0,
            },
        },
        'tactile': {
            'input_dim': 4,
            'num_taxels': 4,
            'axis_dim': 1,
            'content_encoder_type': 'cnn1d',
            'content_hidden_dim': 6,
            'content_layers': 1,
            'content_kernel_size': 3,
            'evidence_hidden_dim': 5,
            'evidence_kernel_size': 3,
        },
        'medium': {'hidden_dim': 7},
        'attributes': {
            'hidden_dim': 6,
            'object_feature_dim': 4,
            'stop_gradient': True,
            'fragility_classes': 2,
            'geometry_classes': 2,
            'surface_classes': 2,
            'metric_tasks': ['geometry'],
        },
        'physical_attributes': {'enabled': False},
        'policy': {
            'hidden_dim': 8,
            'film_hidden_dim': 4,
            'head_type': 'state_residual_per_finger_sign_specific',
            'base_source': 'learned',
            'use_reference_force_context': False,
            'residual_output_scale': 2.0,
            'finger_count': 3,
            'finger_embedding_dim': 2,
        },
        'losses': {'temperature_clip': 0.07, 'temperature_inv': 0.07},
        'direction_conditioning': {
            'enabled': direction_enabled,
            'num_directions': 2,
            'embedding_dim': 3,
            'zero_init': True,
            'require_explicit_direction': True,
            'return_diagnostics': True,
            'medium': {'enabled': True, 'mode': 'residual_film', 'hidden_dim': 4},
            'policy': {'enabled': True, 'mode': 'residual_film', 'hidden_dim': 4},
        },
    }


def _tiny_sequence_batch(*, batch_size: int = 2, windows: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(21)
    tactile_points = 7
    frames = 2
    return {
        'video': torch.randn(batch_size, windows, frames, 3, 16, 16),
        'frame_mask': torch.ones(batch_size, windows, frames, dtype=torch.bool),
        'tactile_high': torch.randn(batch_size, windows, tactile_points, 4),
        'tactile_low': torch.randn(batch_size, windows, tactile_points, 4),
        'tactile_mask': torch.ones(batch_size, windows, tactile_points, dtype=torch.bool),
        'window_mask': torch.ones(batch_size, windows, dtype=torch.bool),
        'window_lengths': torch.full((batch_size,), windows, dtype=torch.long),
        'stable_masks': torch.ones(batch_size, windows, dtype=torch.bool),
        'stable_phases': torch.zeros(batch_size, windows, dtype=torch.long),
        'direction_ids': torch.tensor([index % 2 for index in range(batch_size)], dtype=torch.long),
    }


def _online_batch(sequence_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        'video': sequence_batch['video'][:, 0],
        'frame_mask': sequence_batch['frame_mask'][:, 0],
        'tactile_high': sequence_batch['tactile_high'][:, 0],
        'tactile_low': sequence_batch['tactile_low'][:, 0],
        'tactile_mask': sequence_batch['tactile_mask'][:, 0],
        'direction_ids': sequence_batch['direction_ids'],
    }


def test_direction_embedding_and_adapter_validate_and_preserve_identity() -> None:
    embedding = DirectionEmbedding(num_directions=2, embedding_dim=3)
    condition = embedding(torch.tensor([0, 1], dtype=torch.long))
    adapter = ResidualFiLMAdapter(condition_dim=3, feature_dim=5, hidden_dim=4, zero_init=True)
    features = torch.randn(2, 4, 5)
    actual = adapter(features, condition)
    torch.testing.assert_close(actual, features, rtol=0.0, atol=0.0)
    gamma, beta = adapter.modulation(condition)
    assert torch.count_nonzero(gamma).item() == 0
    assert torch.count_nonzero(beta).item() == 0

    try:
        embedding(torch.tensor([2], dtype=torch.long))
    except RuntimeError as error:
        assert 'within [0, 1]' in str(error)
    else:
        raise AssertionError('Out-of-range direction id was accepted.')


def test_zero_initialized_adapter_output_layer_receives_gradient() -> None:
    torch.manual_seed(5)
    embedding = DirectionEmbedding(num_directions=2, embedding_dim=3)
    adapter = ResidualFiLMAdapter(condition_dim=3, feature_dim=5, hidden_dim=4, zero_init=True)
    features = torch.randn(2, 5)
    conditioned = adapter(features, embedding(torch.tensor([0, 1], dtype=torch.long)))
    loss = conditioned.square().mean()
    loss.backward()
    assert adapter.output_layer.weight.grad is not None
    assert adapter.output_layer.bias.grad is not None
    assert torch.isfinite(adapter.output_layer.weight.grad).all()
    assert torch.count_nonzero(adapter.output_layer.weight.grad).item() > 0


def test_direction_model_requires_explicit_valid_direction_ids() -> None:
    model = CrossMediumSystem(_tiny_model_config(direction_enabled=True)).eval()
    batch = _tiny_sequence_batch()
    missing = dict(batch)
    missing.pop('direction_ids')
    try:
        model(missing)
    except RuntimeError as error:
        assert 'requires batch["direction_ids"]' in str(error)
    else:
        raise AssertionError('Direction-conditioned model accepted a missing direction id.')

    invalid = dict(batch)
    invalid['direction_ids'] = torch.tensor([0, 2], dtype=torch.long)
    try:
        model(invalid)
    except RuntimeError as error:
        assert 'within [0, 1]' in str(error)
    else:
        raise AssertionError('Direction-conditioned model accepted an invalid direction id.')


def test_zero_init_warmstart_preserves_legacy_outputs_and_shapes() -> None:
    torch.manual_seed(8)
    legacy = CrossMediumSystem(_tiny_model_config(direction_enabled=False)).eval()
    direction_model = CrossMediumSystem(_tiny_model_config(direction_enabled=True)).eval()
    batch = _tiny_sequence_batch()

    with tempfile.TemporaryDirectory(prefix='cmg_m2_identity_') as temp_dir:
        checkpoint_path = Path(temp_dir) / 'legacy.pt'
        torch.save({'model': legacy.state_dict(), 'stage_name': 'legacy'}, checkpoint_path)
        info = load_model_weights(
            direction_model,
            checkpoint_path,
            strict=True,
            allowed_missing_prefixes=[
                'direction_embedding.*',
                'medium_direction_adapter.*',
                'policy_direction_adapter.*',
            ],
        )

    assert info['unexpected_keys'] == []
    assert info['disallowed_missing_keys'] == []
    assert all(
        key.startswith(('direction_embedding.', 'medium_direction_adapter.', 'policy_direction_adapter.'))
        for key in info['missing_keys']
    )
    with torch.no_grad():
        legacy_outputs = legacy(batch)
        direction_outputs = direction_model(batch)
        flipped = dict(batch)
        flipped['direction_ids'] = 1 - batch['direction_ids']
        flipped_outputs = direction_model(flipped)
    for key in (
        'medium_logits',
        'medium_probs',
        'medium_sequence_features',
        'force_pred',
        'force_base',
        'force_interface_delta',
        'finger_force_pred',
        'finger_force_base',
        'finger_force_interface_delta',
    ):
        torch.testing.assert_close(direction_outputs[key], legacy_outputs[key], rtol=0.0, atol=0.0)
        torch.testing.assert_close(flipped_outputs[key], direction_outputs[key], rtol=0.0, atol=0.0)


def test_policy_direction_modulation_does_not_change_base_branch() -> None:
    torch.manual_seed(13)
    head = PolicyHead(
        input_dim=6,
        hidden_dim=8,
        film_hidden_dim=4,
        head_type='state_residual_per_finger',
        state_input_dim=5,
        residual_output_scale=2.0,
        finger_count=3,
        finger_embedding_dim=2,
    ).eval()
    adapter = ResidualFiLMAdapter(condition_dim=3, feature_dim=8, hidden_dim=4, zero_init=False)
    task_context = torch.randn(2, 6)
    state_context = torch.randn(2, 5)
    p_medium = torch.softmax(torch.randn(2, 3), dim=-1)
    left_modulation = adapter.modulation(torch.zeros(2, 3))
    right_modulation = adapter.modulation(torch.ones(2, 3))
    left = head(
        task_context,
        p_medium,
        state_context=state_context,
        residual_modulation=left_modulation,
    )
    right = head(
        task_context,
        p_medium,
        state_context=state_context,
        residual_modulation=right_modulation,
    )
    torch.testing.assert_close(left['finger_force_base'], right['finger_force_base'], rtol=0.0, atol=0.0)
    assert not torch.allclose(left['finger_force_interface_delta'], right['finger_force_interface_delta'])


def test_sequence_and_online_step_match_for_one_window() -> None:
    torch.manual_seed(34)
    model = CrossMediumSystem(_tiny_model_config(direction_enabled=True)).eval()
    batch = _tiny_sequence_batch()
    with torch.no_grad():
        sequence_outputs = model(batch)
        online_outputs = model.forward_online_step(_online_batch(batch))
    for key in (
        'medium_logits',
        'medium_probs',
        'medium_sequence_features',
        'force_pred',
        'force_base',
        'force_interface_delta',
        'finger_force_pred',
        'finger_force_base',
        'finger_force_interface_delta',
        'transition_direction_embedding',
    ):
        sequence_value = sequence_outputs[key]
        if key in {'medium_logits', 'medium_probs', 'medium_sequence_features'}:
            sequence_value = sequence_value[:, 0]
        torch.testing.assert_close(sequence_value, online_outputs[key], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        sequence_outputs['medium_hidden'],
        online_outputs['medium_hidden'][0],
        rtol=1e-6,
        atol=1e-6,
    )


def test_online_inference_stub_locks_model_to_eval_mode() -> None:
    model = CrossMediumSystem(_tiny_model_config(direction_enabled=True)).train()
    stub = OnlineInferenceStub(model, device=torch.device('cpu'))
    assert stub.model.training is False
    assert all(module.training is False for module in stub.model.modules())


def test_direction_conditioned_v1_config_matches_m2_contract() -> None:
    config = load_yaml(ROOT / 'configs' / 'model' / 'direction_conditioned_v1.yaml')
    direction = config['direction_conditioning']
    assert direction['enabled'] is True
    assert direction['num_directions'] == 2
    assert direction['embedding_dim'] == 16
    assert direction['zero_init'] is True
    assert direction['require_explicit_direction'] is True
    assert direction['medium']['mode'] == 'residual_film'
    assert direction['policy']['mode'] == 'residual_film'
    assert config['policy']['head_type'] == 'state_residual_per_finger_sign_specific'
