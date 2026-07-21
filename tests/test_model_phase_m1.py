from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from cmg.config import resolve_training_configs, sha256_file
from cmg.training import (
    Trainer,
    enforce_frozen_modules_eval,
    freeze_modules,
    load_model_weights,
)


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


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    torch.save({'model': model.state_dict(), 'stage_name': 'legacy'}, path)


def test_checkpoint_allows_only_explicit_direction_missing_prefixes(tmp_path: Path) -> None:
    torch.manual_seed(3)
    legacy = LegacyModel()
    checkpoint_path = tmp_path / 'legacy.pt'
    _save_checkpoint(checkpoint_path, legacy)
    target = DirectionModel()

    info = load_model_weights(
        target,
        checkpoint_path,
        strict=True,
        allowed_missing_prefixes=[
            'direction_embedding.*',
            'medium_direction_adapter.*',
            'policy_direction_adapter.*',
        ],
    )

    assert torch.equal(target.backbone.weight, legacy.backbone.weight)
    assert torch.equal(target.backbone.bias, legacy.backbone.bias)
    assert set(info['missing_keys']) == {
        'direction_embedding.weight',
        'medium_direction_adapter.weight',
        'medium_direction_adapter.bias',
        'policy_direction_adapter.weight',
        'policy_direction_adapter.bias',
    }
    assert info['unexpected_keys'] == []
    assert info['disallowed_missing_keys'] == []
    assert info['checkpoint_sha256'] == sha256_file(checkpoint_path)
    assert info['module_load_report']['backbone']['loaded_tensor_count'] == 2
    assert info['module_load_report']['direction_embedding']['loaded_tensor_count'] == 0


def test_checkpoint_rejects_unapproved_missing_and_shape_mismatch(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / 'legacy.pt'
    _save_checkpoint(checkpoint_path, LegacyModel())

    with pytest.raises(RuntimeError, match='explicitly allowed architecture delta'):
        load_model_weights(DirectionModel(), checkpoint_path, strict=True)

    with pytest.raises(RuntimeError, match='tensor shape mismatches') as exc_info:
        load_model_weights(LegacyModel(width=4), checkpoint_path, strict=True)
    assert 'checkpoint_shape' in str(exc_info.value)
    assert 'model_shape' in str(exc_info.value)


def test_freeze_paths_fail_fast_and_eval_lock_survives_model_train() -> None:
    model = FreezeModel()
    with pytest.raises(ValueError, match='Unknown freeze_modules paths'):
        freeze_modules(model, ['backbon'])
    assert all(parameter.requires_grad for parameter in model.parameters())

    report = freeze_modules(model, ['backbone'], strict=True, lock_eval=True)
    assert report['resolved_module_names'] == ['backbone']
    assert report['missing_module_names'] == []
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    model.train()
    assert model.backbone.training
    enforce_frozen_modules_eval(model)
    assert not model.backbone.training
    assert model.head.training


def test_external_config_bases_merge_and_record_dependency_hashes(tmp_path: Path) -> None:
    project_root = tmp_path
    (project_root / 'configs').mkdir()
    (project_root / 'data').mkdir()
    (project_root / 'configs' / 'data_base.yaml').write_text(
        'split_path: data/split.yaml\nvalue: data-base\n',
        encoding='utf-8',
    )
    (project_root / 'configs' / 'model_base.yaml').write_text(
        'width: 3\n',
        encoding='utf-8',
    )
    (project_root / 'configs' / 'train_parent.yaml').write_text(
        'epochs: 8\nbatch_size: 4\n',
        encoding='utf-8',
    )
    (project_root / 'configs' / 'train_base.yaml').write_text(
        'extends: train_parent.yaml\nwarmup_ratio: 0.05\n',
        encoding='utf-8',
    )
    (project_root / 'data' / 'split.yaml').write_text('name: split\n', encoding='utf-8')
    stage_path = project_root / 'stage.yaml'
    stage_path.write_text(
        '\n'.join(
            [
                'name: stage_test',
                'data_config: configs/data_base.yaml',
                'model_config: configs/model_base.yaml',
                'train_config: configs/train_base.yaml',
                'split: data/split.yaml',
                'data:',
                '  value: stage-data',
                'model:',
                '  width: 5',
                'train:',
                '  epochs: 3',
            ]
        )
        + '\n',
        encoding='utf-8',
    )

    data, model, train, stage, sources = resolve_training_configs(project_root, stage_path)
    assert data['value'] == 'stage-data'
    assert model['width'] == 5
    assert train['epochs'] == 3
    assert train['batch_size'] == 4
    assert train['warmup_ratio'] == 0.05
    assert train['run_name'] == stage['name'] == 'stage_test'
    assert len(sources['train']['dependencies']) == 2
    assert sources['train']['dependencies'][0]['project_path'] == 'configs/train_parent.yaml'
    assert len(sources['data']['sha256']) == 64
    assert len(sources['stage']['effective_sha256']) == 64


def test_external_config_rejects_split_disagreement(tmp_path: Path) -> None:
    for directory in ('configs/data', 'configs/model', 'configs/train', 'data'):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    (tmp_path / 'configs/data/default.yaml').write_text(
        'split_path: data/from_data.yaml\n', encoding='utf-8'
    )
    (tmp_path / 'configs/model/default.yaml').write_text('{}\n', encoding='utf-8')
    (tmp_path / 'configs/train/base.yaml').write_text('{}\n', encoding='utf-8')
    stage_path = tmp_path / 'stage.yaml'
    stage_path.write_text('name: mismatch\nsplit: data/from_stage.yaml\n', encoding='utf-8')

    with pytest.raises(ValueError, match='split_path disagree'):
        resolve_training_configs(tmp_path, stage_path)


def test_trainer_records_freeze_optimizer_and_initialization_provenance(tmp_path: Path) -> None:
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
        'name': 'm1_test',
        'checkpoint_prefix': 'm1_test',
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
        run_dir=tmp_path / 'run',
        config_sources=config_sources,
        initialization_info=initialization,
    )

    run_config = yaml.safe_load((tmp_path / 'run' / 'run_config.yaml').read_text(encoding='utf-8'))
    assert run_config['initialization']['checkpoint_sha256'] == 'a' * 64
    assert run_config['config_sources'] == config_sources
    assert run_config['freeze']['resolved_module_names'] == ['backbone']
    assert run_config['parameter_audit']['frozen_tensor_count'] == 2
    audit_path = Path(run_config['parameter_audit']['path'])
    assert run_config['parameter_audit']['sha256'] == sha256_file(audit_path)

    checkpoint_path = tmp_path / 'run' / 'checkpoints' / 'manual.pt'
    trainer.save_checkpoint(checkpoint_path, epoch=0, metrics={})
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['initialization'] == initialization
    assert checkpoint['config_sources'] == config_sources
    assert checkpoint['parameter_audit']['sha256'] == sha256_file(audit_path)
