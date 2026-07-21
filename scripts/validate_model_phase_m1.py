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
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import resolve_training_configs, sha256_file
from cmg.models import CrossMediumSystem
from cmg.training import Trainer, enforce_frozen_modules_eval, freeze_modules, load_model_weights


class LegacyModel(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(width, width)


class DirectionModel(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Linear(width, width)
        self.direction_embedding = nn.Embedding(2, 4)
        self.medium_direction_adapter = nn.Linear(4, width)
        self.policy_direction_adapter = nn.Linear(4, width)


class FreezeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
        self.head = nn.Linear(3, 1)


def validate_checkpoint_audit(_: Path, temp_root: Path) -> dict[str, Any]:
    torch.manual_seed(11)
    legacy = LegacyModel()
    checkpoint_path = temp_root / 'legacy.pt'
    torch.save({'model': legacy.state_dict(), 'stage_name': 'legacy'}, checkpoint_path)
    target = DirectionModel()
    prefixes = [
        'direction_embedding.*',
        'medium_direction_adapter.*',
        'policy_direction_adapter.*',
    ]
    info = load_model_weights(
        target,
        checkpoint_path,
        strict=True,
        allowed_missing_prefixes=prefixes,
    )
    assert torch.equal(target.backbone.weight, legacy.backbone.weight)
    assert info['checkpoint_sha256'] == sha256_file(checkpoint_path)
    assert not info['disallowed_missing_keys']
    assert info['module_load_report']['backbone']['loaded_tensor_count'] == 2
    assert info['module_load_report']['direction_embedding']['loaded_tensor_count'] == 0

    unapproved_rejected = False
    try:
        load_model_weights(DirectionModel(), checkpoint_path, strict=True)
    except RuntimeError as error:
        unapproved_rejected = 'explicitly allowed architecture delta' in str(error)
    assert unapproved_rejected

    shape_rejected = False
    try:
        load_model_weights(LegacyModel(width=4), checkpoint_path, strict=True)
    except RuntimeError as error:
        shape_rejected = 'tensor shape mismatches' in str(error)
    assert shape_rejected
    return {
        'checkpoint_sha256': info['checkpoint_sha256'],
        'loaded_tensor_count': info['loaded_tensor_count'],
        'missing_keys': info['missing_keys'],
        'unapproved_missing_rejected': unapproved_rejected,
        'shape_mismatch_rejected': shape_rejected,
    }


def validate_canonical_checkpoint(project_root: Path, _: Path) -> dict[str, Any]:
    run_config_path = project_root / 'runs' / 'stage38j_f20_v3_causal' / 'run_config.yaml'
    checkpoint_path = project_root / 'runs' / 'stage38j_f20_v3_causal' / 'checkpoints' / 'best.pt'
    run_config = yaml.safe_load(run_config_path.read_text(encoding='utf-8'))
    model = CrossMediumSystem(run_config['model'])
    info = load_model_weights(
        model,
        checkpoint_path,
        strict=True,
        allow_lora_injection=True,
    )
    assert info['loaded_tensor_count'] == info['target_tensor_count']
    assert not info['missing_keys']
    assert not info['unexpected_keys']
    assert not info['disallowed_missing_keys']
    assert not info['disallowed_unexpected_keys']
    return {
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_sha256': info['checkpoint_sha256'],
        'loaded_tensor_count': info['loaded_tensor_count'],
        'target_tensor_count': info['target_tensor_count'],
        'missing_key_count': len(info['missing_keys']),
        'unexpected_key_count': len(info['unexpected_keys']),
    }


def validate_freeze_contract(_: Path, __: Path) -> dict[str, Any]:
    model = FreezeModel()
    invalid_rejected = False
    try:
        freeze_modules(model, ['backbon'])
    except ValueError:
        invalid_rejected = True
    assert invalid_rejected
    assert all(parameter.requires_grad for parameter in model.parameters())

    report = freeze_modules(model, ['backbone'], strict=True, lock_eval=True)
    model.train()
    assert model.backbone.training
    enforce_frozen_modules_eval(model)
    assert not model.backbone.training
    assert model.head.training
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    return {
        'invalid_path_rejected': invalid_rejected,
        'resolved_module_names': report['resolved_module_names'],
        'frozen_parameter_tensor_count': report['parameter_tensor_count'],
        'eval_lock_active': not model.backbone.training and model.head.training,
    }


def validate_config_resolution(project_root: Path, temp_root: Path) -> dict[str, Any]:
    resolved_stage_count = 0
    for stage_path in sorted((project_root / 'configs' / 'stages').glob('*.yaml')):
        resolve_training_configs(project_root, stage_path)
        resolved_stage_count += 1

    stage_path = temp_root / 'bidirectional_stage.yaml'
    stage_path.write_text(
        '\n'.join(
            [
                'name: model_phase_m1_bidirectional_config_smoke',
                'data_config: configs/data/policy_20hz_bidirectional_v4.yaml',
                'model_config: configs/model/default.yaml',
                'train_config: configs/train/bidirectional_v1.yaml',
                'split: data/splits/split_unseen_fixed_test_obj004_obj007_v1.yaml',
                'train:',
                '  epochs: 3',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    data, _, train, stage, sources = resolve_training_configs(project_root, stage_path)
    assert data['dataset_version'] == 'bidirectional_v1'
    assert train['sampling_mode'] == 'direction_object_aware'
    assert train['epochs'] == 3
    assert train['run_name'] == stage['name']
    assert len(sources['train']['dependencies']) == 2
    assert all(len(sources[kind]['sha256']) == 64 for kind in ('data', 'model', 'train', 'stage'))

    mismatch_path = temp_root / 'mismatch_stage.yaml'
    mismatch_path.write_text(
        '\n'.join(
            [
                'name: mismatch',
                'data_config: configs/data/policy_20hz_bidirectional_v4.yaml',
                'model_config: configs/model/default.yaml',
                'train_config: configs/train/bidirectional_v1.yaml',
                'split: data/splits/split_unseen_fold1_v1.yaml',
            ]
        )
        + '\n',
        encoding='utf-8',
    )
    split_mismatch_rejected = False
    try:
        resolve_training_configs(project_root, mismatch_path)
    except ValueError as error:
        split_mismatch_rejected = 'split_path disagree' in str(error)
    assert split_mismatch_rejected
    return {
        'legacy_stage_configs_resolved': resolved_stage_count,
        'bidirectional_data_config': sources['data'],
        'bidirectional_train_config': sources['train'],
        'split_mismatch_rejected': split_mismatch_rejected,
    }


def validate_trainer_provenance(_: Path, temp_root: Path) -> dict[str, Any]:
    run_dir = temp_root / 'run'
    model = FreezeModel()
    config = {
        'device': 'cpu',
        'epochs': 1,
        'weight_decay': 0.0,
        'base_learning_rate': 1e-3,
        'lora_learning_rate': 1e-3,
        'warmup_ratio': 0.0,
        'min_lr_scale': 0.1,
        'amp_enabled': False,
        'grad_clip_norm': 1.0,
        'early_stopping_patience': 0,
        'val_every_n_epochs': 1,
        'log_tensorboard': False,
        'phase_class_weights': {'Water': 1.0, 'Interface': 1.0, 'Air': 1.0},
    }
    stage = {
        'name': 'm1_validation',
        'checkpoint_prefix': 'm1_validation',
        'freeze_modules': ['backbone'],
        'selection_metric': 'loss_total',
        'maximize_metric': False,
    }
    initialization = {'checkpoint_path': 'legacy.pt', 'checkpoint_sha256': 'a' * 64}
    config_sources = {'stage': {'path': 'stage.yaml', 'sha256': 'b' * 64}}
    trainer = Trainer(
        model=model,
        train_loader=[None],
        val_loader=[None],
        config=config,
        stage_config=stage,
        model_config={},
        run_dir=run_dir,
        config_sources=config_sources,
        initialization_info=initialization,
    )
    run_config = yaml.safe_load((run_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    audit_path = Path(run_config['parameter_audit']['path'])
    assert run_config['initialization'] == initialization
    assert run_config['config_sources'] == config_sources
    assert run_config['freeze']['resolved_module_names'] == ['backbone']
    assert run_config['parameter_audit']['sha256'] == sha256_file(audit_path)

    checkpoint_path = run_dir / 'checkpoints' / 'validation.pt'
    trainer.save_checkpoint(checkpoint_path, epoch=0, metrics={})
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['initialization'] == initialization
    assert checkpoint['config_sources'] == config_sources
    assert checkpoint['parameter_audit']['sha256'] == sha256_file(audit_path)
    return {
        'run_config_written': (run_dir / 'run_config.yaml').is_file(),
        'parameter_audit_sha256': run_config['parameter_audit']['sha256'],
        'frozen_tensor_count': run_config['parameter_audit']['frozen_tensor_count'],
        'checkpoint_provenance_persisted': True,
    }


def validate_existing_function_tests(project_root: Path, _: Path) -> dict[str, Any]:
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
            except Exception:  # noqa: BLE001 - validation records the full traceback
                failures.append({'test': test_name, 'message': traceback.format_exc()})
    assert not failures, json.dumps(failures, ensure_ascii=False)
    return {
        'passed_count': len(passed),
        'failed_count': len(failures),
        'passed_tests': passed,
    }


def validate_frozen_data_release_integrity(project_root: Path, _: Path) -> dict[str, Any]:
    manifest_path = (
        project_root
        / 'data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/release_manifest_bidirectional_v1.json'
    )
    checksum_path = manifest_path.with_suffix('.sha256')
    release = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_manifest_sha = checksum_path.read_text(encoding='utf-8').split()[0]
    actual_manifest_sha = sha256_file(manifest_path)
    assert actual_manifest_sha == expected_manifest_sha

    protected_mismatches: list[str] = []
    checked_protected_files = 0
    for group_name in ('artifacts', 'validation_reports'):
        for name, record in release[group_name].items():
            path = project_root / record['path']
            checked_protected_files += 1
            if not path.is_file() or sha256_file(path) != record['sha256']:
                protected_mismatches.append(f'{group_name}.{name}')
    assert not protected_mismatches

    source_code_drift: list[str] = []
    for name, record in release['source_code'].items():
        path = project_root / record['path']
        if not path.is_file() or sha256_file(path) != record['sha256']:
            source_code_drift.append(name)
    assert source_code_drift == ['train_entrypoint']
    return {
        'release_id': release['release_id'],
        'release_manifest_sha256': actual_manifest_sha,
        'checked_data_and_report_file_count': checked_protected_files,
        'protected_mismatch_count': len(protected_mismatches),
        'source_code_drift': source_code_drift,
        'source_code_drift_reason': 'Expected Model Phase M1 training-entrypoint upgrade; frozen data release unchanged.',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output', default='data/processed/stats/model_phase_m1_validation.json')
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path

    checks: list[tuple[str, Callable[[Path, Path], dict[str, Any]]]] = [
        ('checkpoint_audit', validate_checkpoint_audit),
        ('canonical_checkpoint', validate_canonical_checkpoint),
        ('freeze_contract', validate_freeze_contract),
        ('config_resolution', validate_config_resolution),
        ('trainer_provenance', validate_trainer_provenance),
        ('existing_function_tests', validate_existing_function_tests),
        ('frozen_data_release_integrity', validate_frozen_data_release_integrity),
    ]
    results: dict[str, Any] = {}
    issues: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix='cmg_model_phase_m1_') as temp_dir:
        temp_root = Path(temp_dir)
        for name, check in checks:
            try:
                results[name] = check(project_root, temp_root)
            except Exception as error:  # noqa: BLE001 - validation must report every failed check
                issues.append({'check': name, 'error_type': type(error).__name__, 'message': str(error)})

    payload = {
        'phase': 'Model Phase M1',
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
