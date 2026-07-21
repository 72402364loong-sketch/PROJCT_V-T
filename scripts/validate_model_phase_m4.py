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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import (
    deep_update,
    load_yaml,
    resolve_training_configs,
    sha256_file,
    sync_tactile_model_config,
    validate_direction_training_contract,
    validate_initialization_source_contract,
)
from cmg.losses import compute_losses
from cmg.models import CrossMediumSystem
from cmg.training import (
    audit_model_parameters,
    audit_optimizer_groups,
    build_optimizer,
    freeze_modules,
    load_model_weights,
)
from scripts.smoke_bidirectional_sampler import compact_windows
from scripts.train import make_dataset, make_train_loader
from scripts.validate_model_phase_m1 import validate_frozen_data_release_integrity
from scripts.validate_model_phase_m3 import _tiny_direction_model_config


STAGE_FILES = (
    'stage39a_direction_adapter_warmup.yaml',
    'stage39b_bidirectional_medium.yaml',
    'stage39c_bidirectional_policy.yaml',
    'stage39d_bidirectional_joint.yaml',
)
DIRECTION_PREFIXES = (
    'direction_embedding.*',
    'medium_direction_adapter.*',
    'policy_direction_adapter.*',
)
OUTPUT_KEYS = (
    'medium_logits',
    'force_pred',
    'force_base',
    'force_interface_delta',
    'finger_force_pred',
    'finger_force_base',
    'finger_force_interface_delta',
)


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + '.')


def _resolved_stages(project_root: Path) -> list[tuple[dict, dict, dict, dict, dict]]:
    return [
        resolve_training_configs(project_root, project_root / 'configs/stages' / stage_file)
        for stage_file in STAGE_FILES
    ]


def _set_trained_policy_outputs(model: CrossMediumSystem) -> None:
    with torch.no_grad():
        for layer in (
            model.policy_head.finger_residual_pos_output_layer,
            model.policy_head.finger_residual_neg_output_layer,
            model.policy_head.finger_residual_direction_output_layer,
        ):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.02)


def _real_paired_batch(project_root: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
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
    train_config = load_yaml(project_root / 'configs/train/direction_conditioned_v1.yaml')
    train_config.update({'num_workers': 0, 'pin_memory': False, 'persistent_workers': False})
    dataset = make_dataset(
        project_root,
        project_root / data_config['split_path'],
        'train',
        data_config,
        train_config,
        {'tail_mode': 'all_valid'},
    )
    compact_windows(dataset)
    batch = next(iter(make_train_loader(dataset, train_config)))
    assert batch['directions'].count('W2A') == 4
    assert batch['directions'].count('A2W') == 4
    return batch, dataset, train_config


def validate_m3_prerequisite(project_root: Path, _: Path) -> dict[str, Any]:
    report_path = project_root / 'data/processed/stats/model_phase_m3_validation.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['error_count'] == 0
    return {
        'report_path': str(report_path.resolve()),
        'report_sha256': sha256_file(report_path),
        'passed_check_count': report['passed_check_count'],
        'error_count': report['error_count'],
    }


def validate_stage_configs_and_canonical_checkpoint(project_root: Path, _: Path) -> dict[str, Any]:
    resolved = _resolved_stages(project_root)
    canonical_path = project_root / 'runs/stage38j_f20_v3_causal/checkpoints/best.pt'
    first_data, first_model_config, first_train, first_stage, _ = resolved[0]
    first_model_config = sync_tactile_model_config(first_data, first_model_config)
    preflight = validate_direction_training_contract(
        project_root,
        first_data,
        first_model_config,
        first_train,
        first_stage,
        initialization_path=canonical_path,
        allow_pending_e0=True,
    )
    model = CrossMediumSystem(first_model_config)
    load_info = load_model_weights(
        model,
        canonical_path,
        strict=True,
        allow_lora_injection=True,
        allowed_missing_prefixes=DIRECTION_PREFIXES,
    )
    assert load_info['loaded_tensor_count'] == 471
    assert load_info['target_tensor_count'] == 480
    assert len(load_info['missing_keys']) == 9

    stage_audits: dict[str, Any] = {}
    for _, _, train_config, stage_config, sources in resolved:
        for parameter in model.parameters():
            parameter.requires_grad = True
            parameter.grad = None
        model.train()
        freeze_report = freeze_modules(
            model,
            stage_config['freeze_modules'],
            strict=True,
            lock_eval=True,
            train_mode_module_names=stage_config.get('frozen_train_mode_modules', []),
        )
        parameter_audit = audit_model_parameters(model)
        trainable = parameter_audit['trainable_parameter_names']
        expected = stage_config['expected_trainable_prefixes']
        unexpected_trainable = [
            name for name in trainable if not any(_matches_prefix(name, prefix) for prefix in expected)
        ]
        missing_prefixes = [
            prefix for prefix in expected if not any(_matches_prefix(name, prefix) for name in trainable)
        ]
        assert not unexpected_trainable
        assert not missing_prefixes
        optimizer = build_optimizer(model, train_config)
        optimizer_groups = audit_optimizer_groups(model, optimizer)
        stage_audits[stage_config['name']] = {
            'stage_config_sha256': sources['stage']['sha256'],
            'stage_config_effective_sha256': sources['stage']['effective_sha256'],
            'freeze_module_count': len(freeze_report['resolved_module_names']),
            'trainable_tensor_count': parameter_audit['trainable_tensor_count'],
            'trainable_value_count': parameter_audit['trainable_value_count'],
            'trainable_fraction': parameter_audit['trainable_fraction'],
            'optimizer_groups': [
                {
                    'name': group['name'],
                    'learning_rate': group['learning_rate'],
                    'parameter_tensor_count': group['parameter_tensor_count'],
                }
                for group in optimizer_groups
            ],
            'unexpected_trainable': unexpected_trainable,
            'missing_expected_prefixes': missing_prefixes,
        }
        del optimizer

    del model
    gc.collect()
    return {
        'preflight': preflight,
        'canonical_checkpoint_path': str(canonical_path.resolve()),
        'canonical_checkpoint_sha256': load_info['checkpoint_sha256'],
        'loaded_tensor_count': load_info['loaded_tensor_count'],
        'target_tensor_count': load_info['target_tensor_count'],
        'direction_missing_key_count': len(load_info['missing_keys']),
        'stage_audits': stage_audits,
    }


def validate_function_tests(project_root: Path, _: Path) -> dict[str, Any]:
    passed: list[str] = []
    failures: list[dict[str, str]] = []
    for path in sorted((project_root / 'tests').glob('test_*.py')):
        if path.name in {'test_model_phase_m1.py', 'test_model_phase_m2.py', 'test_model_phase_m3.py'}:
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
    m4_tests = [name for name in passed if name.startswith('test_model_phase_m4.py::')]
    assert len(m4_tests) == 9
    return {
        'passed_count': len(passed),
        'failed_count': len(failures),
        'm4_test_count': len(m4_tests),
        'm4_tests': m4_tests,
    }


def validate_four_stage_real_batch_smoke(project_root: Path, temp_root: Path) -> dict[str, Any]:
    torch.manual_seed(49)
    batch, _, shared_train_config = _real_paired_batch(project_root)
    input_dim = int(batch['tactile_high'].shape[-1])
    legacy_config = _tiny_direction_model_config(input_dim)
    legacy_config['direction_conditioning']['enabled'] = False
    legacy = CrossMediumSystem(legacy_config)
    _set_trained_policy_outputs(legacy)
    previous_path = temp_root / 'legacy_w2a.pt'
    torch.save({'model': legacy.state_dict(), 'stage_name': 'stage38j_f20_v3_causal'}, previous_path)
    del legacy

    stage_reports: dict[str, Any] = {}
    previous_outputs: dict[str, torch.Tensor] | None = None
    for stage_file in STAGE_FILES:
        _, _, train_config, stage_config, _ = resolve_training_configs(
            project_root,
            project_root / 'configs/stages' / stage_file,
        )
        model = CrossMediumSystem(_tiny_direction_model_config(input_dim))
        initialization = stage_config['initialization']
        load_info = load_model_weights(
            model,
            previous_path,
            strict=True,
            allow_lora_injection=bool(initialization.get('allow_lora_injection', True)),
            allowed_missing_prefixes=initialization.get('allowed_missing_prefixes', []),
        )
        if initialization.get('expected_source_stage'):
            validate_initialization_source_contract(stage_config, load_info)
        model.eval()
        with torch.no_grad():
            loaded_outputs = {key: value.detach().clone() for key, value in model(batch).items() if key in OUTPUT_KEYS}
        reload_max_difference = 0.0
        if previous_outputs is not None:
            reload_max_difference = max(
                float((loaded_outputs[key] - previous_outputs[key]).abs().max().item())
                for key in OUTPUT_KEYS
            )
            assert reload_max_difference == 0.0

        freeze_report = freeze_modules(
            model,
            stage_config['freeze_modules'],
            strict=True,
            lock_eval=True,
            train_mode_module_names=stage_config.get('frozen_train_mode_modules', []),
        )
        optimizer = build_optimizer(model, train_config)
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for module_name in freeze_report['resolved_module_names']:
            module = model
            for part in module_name.split('.'):
                module = getattr(module, part)
            module.eval()
        outputs = model(batch)
        losses = compute_losses(
            outputs,
            batch,
            loss_weights=stage_config['loss_weights'],
            policy_loss_config=stage_config['policy_loss'],
            temperature_clip=0.07,
            temperature_inv=0.07,
            phase_class_weights=torch.ones(3),
            loss_reduction_config=shared_train_config['loss_reduction'],
        )
        assert torch.isfinite(losses['total'])
        losses['total'].backward()

        trainable_gradient_nonzero: dict[str, int] = {}
        frozen_gradient_tensor_count = 0
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                prefix = name.split('.')[0]
                if name.startswith('policy_head.'):
                    prefix = 'policy_head'
                count = int(torch.count_nonzero(parameter.grad).item()) if parameter.grad is not None else 0
                trainable_gradient_nonzero[prefix] = trainable_gradient_nonzero.get(prefix, 0) + count
            elif parameter.grad is not None:
                frozen_gradient_tensor_count += 1
        assert frozen_gradient_tensor_count == 0
        if float(stage_config['loss_weights'].get('med', 0.0)) > 0.0:
            assert trainable_gradient_nonzero.get('medium_direction_adapter', 0) > 0
        if float(stage_config['loss_weights'].get('pol', 0.0)) > 0.0:
            assert trainable_gradient_nonzero.get('policy_direction_adapter', 0) > 0
        optimizer.step()

        model.eval()
        with torch.no_grad():
            previous_outputs = {key: value.detach().clone() for key, value in model(batch).items() if key in OUTPUT_KEYS}
        stage_checkpoint = temp_root / f'{stage_config["name"]}.pt'
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stage_name': stage_config['name'],
                'metrics': {},
            },
            stage_checkpoint,
        )
        stage_reports[stage_config['name']] = {
            'initialized_from_stage': load_info['stage_name'],
            'loaded_tensor_count': load_info['loaded_tensor_count'],
            'target_tensor_count': load_info['target_tensor_count'],
            'missing_key_count': len(load_info['missing_keys']),
            'loss_total': float(losses['total'].detach().item()),
            'loss_med_w2a': float(losses['med_w2a'].detach().item()),
            'loss_med_a2w': float(losses['med_a2w'].detach().item()),
            'loss_pol_w2a': float(losses['pol_w2a'].detach().item()),
            'loss_pol_a2w': float(losses['pol_a2w'].detach().item()),
            'reload_max_abs_difference': reload_max_difference,
            'frozen_gradient_tensor_count': frozen_gradient_tensor_count,
            'trainable_gradient_nonzero_values': trainable_gradient_nonzero,
            'checkpoint_tensor_count': len(model.state_dict()),
        }
        previous_path = stage_checkpoint
        del model, optimizer, outputs, losses
        gc.collect()

    final_model = CrossMediumSystem(_tiny_direction_model_config(input_dim)).eval()
    final_info = load_model_weights(final_model, previous_path, strict=True)
    with torch.no_grad():
        final_outputs = final_model(batch)
    final_reload_difference = max(
        float((final_outputs[key] - previous_outputs[key]).abs().max().item())
        for key in OUTPUT_KEYS
    )
    assert final_reload_difference == 0.0
    return {
        'batch_size': int(batch['direction_ids'].numel()),
        'direction_counts': {'W2A': batch['directions'].count('W2A'), 'A2W': batch['directions'].count('A2W')},
        'window_counts': [int(value) for value in batch['window_lengths'].tolist()],
        'stage_reports': stage_reports,
        'final_checkpoint_stage': final_info['stage_name'],
        'final_checkpoint_tensor_count': final_info['target_tensor_count'],
        'final_reload_max_abs_difference': final_reload_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/model_phase_m4_validation.json')
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    checks: list[tuple[str, Callable[[Path, Path], dict[str, Any]]]] = [
        ('m3_prerequisite', validate_m3_prerequisite),
        ('stage_configs_and_canonical_checkpoint', validate_stage_configs_and_canonical_checkpoint),
        ('function_tests', validate_function_tests),
        ('four_stage_real_batch_smoke', validate_four_stage_real_batch_smoke),
        ('frozen_data_release_integrity', validate_frozen_data_release_integrity),
    ]
    results: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix='cmg_model_phase_m4_') as temp_dir:
        temp_root = Path(temp_dir)
        for name, check in checks:
            try:
                results[name] = check(project_root, temp_root)
            except Exception as error:  # noqa: BLE001
                issues.append({'check': name, 'error_type': type(error).__name__, 'message': str(error)})

    payload = {
        'phase': 'Model Phase M4',
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
