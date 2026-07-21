from __future__ import annotations

import argparse
import gc
import importlib.util
import inspect
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import load_yaml, resolve_training_configs, sha256_file, sync_tactile_model_config
from cmg.models import CrossMediumSystem
from cmg.training import load_model_weights
from scripts.validate_model_phase_m1 import validate_frozen_data_release_integrity


DIRECTION_PREFIXES = [
    'direction_embedding.*',
    'medium_direction_adapter.*',
    'policy_direction_adapter.*',
]

IDENTITY_KEYS = [
    'medium_logits',
    'medium_probs',
    'medium_sequence_features',
    'force_pred',
    'force_base',
    'force_interface_delta',
    'finger_force_pred',
    'finger_force_base',
    'finger_force_interface_delta',
]


def synthetic_cached_visual_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    batch_size, windows, tactile_points = 2, 1, 26
    return {
        'visual_features': torch.randn(batch_size, windows, 512),
        'tactile_high': torch.randn(batch_size, windows, tactile_points, 12),
        'tactile_low': torch.randn(batch_size, windows, tactile_points, 12),
        'tactile_mask': torch.ones(batch_size, windows, tactile_points, dtype=torch.bool),
        'window_mask': torch.ones(batch_size, windows, dtype=torch.bool),
        'window_lengths': torch.ones(batch_size, dtype=torch.long),
        'stable_masks': torch.ones(batch_size, windows, dtype=torch.bool),
        'stable_phases': torch.tensor([[0], [2]], dtype=torch.long),
        'finger_reference_forces': torch.tensor([[[30.0, 31.0, 32.0]], [[25.0, 26.0, 27.0]]]),
        'direction_ids': torch.tensor([0, 1], dtype=torch.long),
    }


def validate_canonical_warmstart_identity(project_root: Path, _: Path) -> dict[str, Any]:
    checkpoint_path = project_root / 'runs/stage38j_f20_v3_causal/checkpoints/best.pt'
    run_config = yaml.safe_load(
        (project_root / 'runs/stage38j_f20_v3_causal/run_config.yaml').read_text(encoding='utf-8')
    )
    batch = synthetic_cached_visual_batch()

    legacy_model = CrossMediumSystem(run_config['model']).eval()
    legacy_info = load_model_weights(
        legacy_model,
        checkpoint_path,
        strict=True,
        allow_lora_injection=True,
    )
    with torch.no_grad():
        legacy_full_outputs = legacy_model(batch)
        legacy_outputs = {key: legacy_full_outputs[key].detach().cpu().clone() for key in IDENTITY_KEYS}
    del legacy_full_outputs, legacy_model
    gc.collect()

    data_config = load_yaml(project_root / 'configs/data/policy_20hz_bidirectional_v4.yaml')
    model_config = load_yaml(project_root / 'configs/model/direction_conditioned_v1.yaml')
    model_config = sync_tactile_model_config(data_config, model_config)
    direction_model = CrossMediumSystem(model_config).eval()
    direction_info = load_model_weights(
        direction_model,
        checkpoint_path,
        strict=True,
        allow_lora_injection=True,
        allowed_missing_prefixes=DIRECTION_PREFIXES,
    )
    with torch.no_grad():
        direction_outputs = direction_model(batch)
        flipped_batch = dict(batch)
        flipped_batch['direction_ids'] = 1 - batch['direction_ids']
        flipped_outputs = direction_model(flipped_batch)

    max_legacy_difference = max(
        float((direction_outputs[key].cpu() - legacy_outputs[key]).abs().max().item())
        for key in IDENTITY_KEYS
    )
    max_direction_flip_difference = max(
        float((direction_outputs[key] - flipped_outputs[key]).abs().max().item())
        for key in IDENTITY_KEYS
    )
    embedding = direction_model.direction_embedding(batch['direction_ids'])
    medium_gamma, medium_beta = direction_model.medium_direction_adapter.modulation(embedding)
    policy_gamma, policy_beta = direction_model.policy_direction_adapter.modulation(embedding)
    max_initial_modulation = max(
        float(tensor.abs().max().item())
        for tensor in (medium_gamma, medium_beta, policy_gamma, policy_beta)
    )

    expected_missing = {
        'direction_embedding.embedding.weight',
        'medium_direction_adapter.input_layer.weight',
        'medium_direction_adapter.input_layer.bias',
        'medium_direction_adapter.output_layer.weight',
        'medium_direction_adapter.output_layer.bias',
        'policy_direction_adapter.input_layer.weight',
        'policy_direction_adapter.input_layer.bias',
        'policy_direction_adapter.output_layer.weight',
        'policy_direction_adapter.output_layer.bias',
    }
    assert set(direction_info['missing_keys']) == expected_missing
    assert not direction_info['unexpected_keys']
    assert not direction_info['disallowed_missing_keys']
    assert not direction_info['disallowed_unexpected_keys']
    assert direction_info['loaded_tensor_count'] == legacy_info['loaded_tensor_count'] == 471
    assert direction_info['target_tensor_count'] == 480
    assert max_legacy_difference == 0.0
    assert max_direction_flip_difference == 0.0
    assert max_initial_modulation == 0.0
    return {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_sha256': direction_info['checkpoint_sha256'],
        'loaded_tensor_count': direction_info['loaded_tensor_count'],
        'target_tensor_count': direction_info['target_tensor_count'],
        'direction_missing_key_count': len(direction_info['missing_keys']),
        'direction_missing_keys': direction_info['missing_keys'],
        'unexpected_key_count': len(direction_info['unexpected_keys']),
        'max_abs_new_vs_legacy': max_legacy_difference,
        'max_abs_direction_flip_at_zero_init': max_direction_flip_difference,
        'max_abs_initial_direction_modulation': max_initial_modulation,
    }


def validate_config_resolution(project_root: Path, temp_root: Path) -> dict[str, Any]:
    stage_path = temp_root / 'm2_config_smoke.yaml'
    stage_path.write_text(
        '\n'.join(
            [
                'name: model_phase_m2_config_smoke',
                'data_config: configs/data/policy_20hz_bidirectional_v4.yaml',
                'model_config: configs/model/direction_conditioned_v1.yaml',
                'train_config: configs/train/bidirectional_v1.yaml',
                'split: data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    data, model, train, _, sources = resolve_training_configs(project_root, stage_path)
    assert data['dataset_version'] == 'bidirectional_v1'
    assert model['direction_conditioning']['enabled'] is True
    assert model['policy']['head_type'] == 'state_residual_per_finger_sign_specific'
    assert train['sampling_mode'] == 'direction_object_aware'
    assert sources['model']['project_path'] == 'configs/model/direction_conditioned_v1.yaml'
    assert len(sources['model']['dependencies']) == 2
    return {
        'model_config_path': sources['model']['path'],
        'model_config_sha256': sources['model']['sha256'],
        'model_config_effective_sha256': sources['model']['effective_sha256'],
        'model_dependency_count': len(sources['model']['dependencies']),
        'dataset_version': data['dataset_version'],
        'sampling_mode': train['sampling_mode'],
    }


def validate_function_tests(project_root: Path, _: Path) -> dict[str, Any]:
    passed: list[str] = []
    failures: list[dict[str, str]] = []
    for path in sorted((project_root / 'tests').glob('test_*.py')):
        if path.name == 'test_model_phase_m1.py':
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            failures.append({'test': path.name, 'message': 'Unable to load test module.'})
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name, function in sorted(vars(module).items()):
            if not name.startswith('test_') or not callable(function):
                continue
            if inspect.signature(function).parameters:
                continue
            test_name = f'{path.name}::{name}'
            try:
                function()
                passed.append(test_name)
            except Exception:  # noqa: BLE001 - validation records complete failures
                failures.append({'test': test_name, 'message': traceback.format_exc()})
    assert not failures, json.dumps(failures, ensure_ascii=False)
    m2_tests = [name for name in passed if name.startswith('test_model_phase_m2.py::')]
    assert len(m2_tests) == 8
    return {
        'passed_count': len(passed),
        'failed_count': len(failures),
        'm2_test_count': len(m2_tests),
        'm2_tests': m2_tests,
    }


def validate_m1_prerequisite(project_root: Path, _: Path) -> dict[str, Any]:
    report_path = project_root / 'data/processed/stats/model_phase_m1_validation.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['error_count'] == 0
    return {
        'report_path': str(report_path.resolve()),
        'report_sha256': sha256_file(report_path),
        'passed_check_count': report['passed_check_count'],
        'error_count': report['error_count'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/model_phase_m2_validation.json')
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    checks: list[tuple[str, Callable[[Path, Path], dict[str, Any]]]] = [
        ('m1_prerequisite', validate_m1_prerequisite),
        ('canonical_warmstart_identity', validate_canonical_warmstart_identity),
        ('config_resolution', validate_config_resolution),
        ('function_tests', validate_function_tests),
        ('frozen_data_release_integrity', validate_frozen_data_release_integrity),
    ]
    results: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix='cmg_model_phase_m2_') as temp_dir:
        temp_root = Path(temp_dir)
        for name, check in checks:
            try:
                results[name] = check(project_root, temp_root)
            except Exception as error:  # noqa: BLE001 - validator reports every check
                issues.append({'check': name, 'error_type': type(error).__name__, 'message': str(error)})

    payload = {
        'phase': 'Model Phase M2',
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
