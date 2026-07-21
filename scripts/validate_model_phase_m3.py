from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml, resolve_training_configs, sha256_file, sync_tactile_model_config
from cmg.direction_metrics import DirectionMetricAccumulator
from cmg.losses import compute_losses
from cmg.models import CrossMediumSystem
from cmg.training import run_model_epoch
from scripts.smoke_bidirectional_sampler import compact_windows
from scripts.train import make_dataset, make_train_loader
from scripts.validate_model_phase_m1 import validate_frozen_data_release_integrity


def _tiny_direction_model_config(input_dim: int) -> dict[str, Any]:
    return {
        'visual': {
            'backbone': 'small_cnn',
            'pretrained': False,
            'freeze_backbone': False,
            'use_lora': False,
            'token_dim': 16,
            'hidden_dim': 16,
            'proj_dim': 8,
            'adapter_rank': 2,
            'max_windows_per_encode': 8,
            'pooling': {
                'score_hidden_dim': 8,
                'pre_norm': True,
                'output_post_norm': True,
                'use_mean_residual': True,
                'dropout': 0.0,
            },
        },
        'tactile': {
            'input_dim': input_dim,
            'num_taxels': input_dim,
            'axis_dim': 1,
            'content_encoder_type': 'cnn1d',
            'content_hidden_dim': 12,
            'content_layers': 1,
            'content_kernel_size': 3,
            'evidence_hidden_dim': 10,
            'evidence_kernel_size': 3,
        },
        'medium': {'hidden_dim': 10},
        'attributes': {
            'hidden_dim': 12,
            'object_feature_dim': 8,
            'stop_gradient': True,
            'fragility_classes': 2,
            'geometry_classes': 2,
            'surface_classes': 2,
            'metric_tasks': ['geometry'],
        },
        'physical_attributes': {'enabled': False},
        'policy': {
            'hidden_dim': 12,
            'film_hidden_dim': 8,
            'head_type': 'state_residual_per_finger_sign_specific',
            'base_source': 'reference_force',
            'use_reference_force_context': True,
            'reference_force_context_scale': 100.0,
            'residual_output_scale': 2.0,
            'finger_count': 3,
            'finger_embedding_dim': 4,
        },
        'losses': {'temperature_clip': 0.07, 'temperature_inv': 0.07},
        'direction_conditioning': {
            'enabled': True,
            'num_directions': 2,
            'embedding_dim': 6,
            'zero_init': True,
            'require_explicit_direction': True,
            'return_diagnostics': False,
            'medium': {'enabled': True, 'mode': 'residual_film', 'hidden_dim': 8},
            'policy': {'enabled': True, 'mode': 'residual_film', 'hidden_dim': 8},
        },
    }


def validate_m2_prerequisite(project_root: Path, _: Path) -> dict[str, Any]:
    report_path = project_root / 'data/processed/stats/model_phase_m2_validation.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['error_count'] == 0
    return {
        'report_path': str(report_path.resolve()),
        'report_sha256': sha256_file(report_path),
        'passed_check_count': report['passed_check_count'],
        'error_count': report['error_count'],
    }


def validate_config_resolution(project_root: Path, temp_root: Path) -> dict[str, Any]:
    stage_path = temp_root / 'm3_config_smoke.yaml'
    stage_path.write_text(
        '\n'.join(
            [
                'name: model_phase_m3_config_smoke',
                'data_config: configs/data/policy_20hz_bidirectional_v4.yaml',
                'model_config: configs/model/direction_conditioned_v1.yaml',
                'train_config: configs/train/direction_conditioned_v1.yaml',
                'split: data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    data, model, train, _, sources = resolve_training_configs(project_root, stage_path)
    assert data['dataset_version'] == 'bidirectional_v1'
    assert model['direction_conditioning']['enabled'] is True
    assert train['sampling_mode'] == 'direction_object_aware'
    assert train['loss_reduction']['mode'] == 'sample_direction_macro'
    assert train['w2a_retention_guard']['max_relative_degradation'] == 0.05
    return {
        'train_config_path': sources['train']['path'],
        'train_config_sha256': sources['train']['sha256'],
        'train_config_effective_sha256': sources['train']['effective_sha256'],
        'train_dependency_count': len(sources['train']['dependencies']),
        'loss_reduction_mode': train['loss_reduction']['mode'],
        'sampling_mode': train['sampling_mode'],
    }


def validate_function_tests(project_root: Path, _: Path) -> dict[str, Any]:
    passed: list[str] = []
    failures: list[dict[str, str]] = []
    for path in sorted((project_root / 'tests').glob('test_*.py')):
        if path.name in {'test_model_phase_m1.py', 'test_model_phase_m2.py'}:
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            failures.append({'test': path.name, 'message': 'Unable to load test module.'})
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in sorted(vars(module).items()):
            if not name.startswith('test_') or not callable(function) or inspect.signature(function).parameters:
                continue
            test_name = f'{path.name}::{name}'
            try:
                function()
                passed.append(test_name)
            except Exception:  # noqa: BLE001
                failures.append({'test': test_name, 'message': traceback.format_exc()})
    assert not failures, json.dumps(failures, ensure_ascii=False)
    m3_tests = [name for name in passed if name.startswith('test_model_phase_m3.py::')]
    assert len(m3_tests) == 8
    return {
        'passed_count': len(passed),
        'failed_count': len(failures),
        'm3_test_count': len(m3_tests),
        'm3_tests': m3_tests,
    }


def validate_real_paired_batch_smoke(project_root: Path, _: Path) -> dict[str, Any]:
    torch.manual_seed(39)
    data_config = deep_update(
        load_yaml(project_root / 'configs/data/default.yaml'),
        load_yaml(project_root / 'configs/data/policy_20hz_bidirectional_v4.yaml'),
    )
    data_config.update(
        {
            'num_frames_per_window': 2,
            'image_size': 64,
            'roi': None,
            'tactile_points_per_window': 8,
            'visual_feature_cache_dir': None,
        }
    )
    train_config = deep_update(
        load_yaml(project_root / 'configs/train/bidirectional_v1.yaml'),
        load_yaml(project_root / 'configs/train/direction_conditioned_v1.yaml'),
    )
    train_config.update({'num_workers': 0, 'pin_memory': False, 'persistent_workers': False})
    stage_config = {'tail_mode': 'all_valid'}
    split_path = project_root / data_config['split_path']
    dataset = make_dataset(project_root, split_path, 'train', data_config, train_config, stage_config)
    compact_windows(dataset)
    batch = next(iter(make_train_loader(dataset, train_config)))
    direction_counts = {
        direction: int(batch['directions'].count(direction))
        for direction in ('W2A', 'A2W')
    }
    assert direction_counts == {'W2A': 4, 'A2W': 4}

    input_dim = int(batch['tactile_high'].shape[-1])
    model = CrossMediumSystem(_tiny_direction_model_config(input_dim)).train()
    # Formal M3 warm-starts from a trained W2A residual head. The tiny test model's
    # sign-specific output layers are zero-initialized, so make them representative
    # before checking whether the newly zero-initialized direction adapter receives gradient.
    with torch.no_grad():
        for layer in (
            model.policy_head.finger_residual_pos_output_layer,
            model.policy_head.finger_residual_neg_output_layer,
            model.policy_head.finger_residual_direction_output_layer,
        ):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)
    outputs = model(batch)
    losses = compute_losses(
        outputs,
        batch,
        loss_weights={'med': 1.0, 'pol': 1.0},
        policy_loss_config={
            'type': 'explicit_reference_delta_per_finger_smooth_l1',
            'all_weight': 0.0,
            'interface_weight': 0.0,
            'stable_weight': 0.0,
            'base_reference_weight': 0.0,
            'base_stable_weight': 0.0,
            'residual_interface_weight': 1.0,
            'residual_stable_zero_weight': 1.0,
            'residual_non_interface_weight': 0.25,
            'delta_normalization_scale': 100.0,
            'beta': 1.0,
        },
        temperature_clip=0.07,
        temperature_inv=0.07,
        phase_class_weights=torch.ones(3),
        loss_reduction_config=train_config['loss_reduction'],
    )
    assert torch.isfinite(losses['total'])
    losses['total'].backward()
    adapter_gradients = {
        'medium_output_weight': model.medium_direction_adapter.output_layer.weight.grad,
        'policy_output_weight': model.policy_direction_adapter.output_layer.weight.grad,
    }
    for gradient_name, gradient in adapter_gradients.items():
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient).item() > 0, f'{gradient_name} gradient is all zero.'

    accumulator = DirectionMetricAccumulator(finger_large_delta_threshold=100.0)
    accumulator.update(outputs, batch)
    metrics = accumulator.finalize()
    assert metrics['medium_f1_interface_macro_direction_complete'] is True
    assert metrics['finger_control_interface_mae_macro_direction_complete'] is True

    class _OneBatchLoader:
        def __init__(self, only_batch: dict[str, Any], source_dataset: Any) -> None:
            self.only_batch = only_batch
            self.dataset = source_dataset

        def __iter__(self):
            yield self.only_batch

        def __len__(self) -> int:
            return 1

    epoch_metrics = run_model_epoch(
        model,
        _OneBatchLoader(batch, dataset),
        device=torch.device('cpu'),
        training=False,
        stage_config={
            'loss_weights': {'med': 1.0, 'pol': 1.0},
            'policy_loss': {
                'type': 'explicit_reference_delta_per_finger_smooth_l1',
                'all_weight': 0.0,
                'interface_weight': 0.0,
                'stable_weight': 0.0,
                'base_reference_weight': 0.0,
                'base_stable_weight': 0.0,
                'residual_interface_weight': 1.0,
                'residual_stable_zero_weight': 1.0,
                'residual_non_interface_weight': 0.25,
                'delta_normalization_scale': 100.0,
                'beta': 1.0,
            },
        },
        model_config=_tiny_direction_model_config(input_dim),
        phase_class_weights=torch.ones(3),
        loss_reduction_config=train_config['loss_reduction'],
    )
    assert epoch_metrics['medium_f1_interface_macro_direction_complete'] is True
    assert epoch_metrics['finger_control_interface_mae_macro_direction_complete'] is True
    assert epoch_metrics['loss_med_w2a'] > 0.0
    assert epoch_metrics['loss_med_a2w'] > 0.0
    return {
        'batch_size': int(batch['direction_ids'].numel()),
        'direction_counts': direction_counts,
        'window_counts': [int(value) for value in batch['window_lengths'].tolist()],
        'loss_total': float(losses['total'].detach().item()),
        'loss_med_w2a': float(losses['med_w2a'].detach().item()),
        'loss_med_a2w': float(losses['med_a2w'].detach().item()),
        'loss_pol_w2a': float(losses['pol_w2a'].detach().item()),
        'loss_pol_a2w': float(losses['pol_a2w'].detach().item()),
        'adapter_gradient_nonzero': {key: int(torch.count_nonzero(value).item()) for key, value in adapter_gradients.items()},
        'medium_macro_complete': metrics['medium_f1_interface_macro_direction_complete'],
        'policy_macro_complete': metrics['finger_control_interface_mae_macro_direction_complete'],
        'epoch_metric_key_count': len(epoch_metrics),
        'epoch_medium_f1_interface_macro_direction': epoch_metrics['medium_f1_interface_macro_direction'],
        'epoch_finger_control_interface_mae_macro_direction': epoch_metrics[
            'finger_control_interface_mae_macro_direction'
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/model_phase_m3_validation.json')
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    checks: list[tuple[str, Callable[[Path, Path], dict[str, Any]]]] = [
        ('m2_prerequisite', validate_m2_prerequisite),
        ('config_resolution', validate_config_resolution),
        ('function_tests', validate_function_tests),
        ('real_paired_batch_smoke', validate_real_paired_batch_smoke),
        ('frozen_data_release_integrity', validate_frozen_data_release_integrity),
    ]
    results: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix='cmg_model_phase_m3_') as temp_dir:
        temp_root = Path(temp_dir)
        for name, check in checks:
            try:
                results[name] = check(project_root, temp_root)
            except Exception as error:  # noqa: BLE001
                issues.append({'check': name, 'error_type': type(error).__name__, 'message': str(error)})

    payload = {
        'phase': 'Model Phase M3',
        'project_root': str(project_root),
        'check_count': len(checks),
        'passed_check_count': len(checks) - len(issues),
        'error_count': len(issues),
        'checks': results,
        'issues': issues,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
