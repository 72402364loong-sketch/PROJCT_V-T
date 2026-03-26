from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import CrossMediumSequenceDataset, sequence_collate_fn
from cmg.models import CrossMediumSystem
from cmg.training import build_phase_class_weights, load_checkpoint_context, load_model_weights, run_model_epoch


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def apply_archived_run_config(
    data_config: dict,
    model_config: dict,
    train_config: dict,
    stage_config: dict,
    checkpoint_context: dict,
) -> tuple[dict, dict, dict]:
    archived = checkpoint_context.get('run_config')
    if not isinstance(archived, dict):
        return data_config, model_config, train_config

    archived_stage = archived.get('stage') if isinstance(archived.get('stage'), dict) else {}
    archived_stage_name = archived_stage.get('name')
    current_stage_name = stage_config.get('name')
    if archived_stage_name and current_stage_name and str(archived_stage_name) != str(current_stage_name):
        raise ValueError(
            f'Checkpoint stage {archived_stage_name!r} does not match evaluation stage {current_stage_name!r}.'
        )

    if isinstance(archived.get('data'), dict):
        data_config = archived['data']
    elif isinstance(archived_stage.get('data'), dict):
        data_config = deep_update(data_config, archived_stage['data'])
    if isinstance(archived.get('model'), dict):
        model_config = archived['model']
    if isinstance(archived.get('train'), dict):
        train_config = archived['train']
    return data_config, model_config, train_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--only-stable', action='store_true')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    checkpoint_context = load_checkpoint_context(checkpoint_path)

    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    data_config, model_config, train_config = apply_archived_run_config(
        data_config,
        model_config,
        train_config,
        stage_config,
        checkpoint_context,
    )
    if data_config.get('visual_feature_cache_dir') and (
        float(stage_config.get('loss_weights', {}).get('clip', 0.0)) > 0.0
        or float(stage_config.get('loss_weights', {}).get('inv', 0.0)) > 0.0
    ):
        raise ValueError('visual feature cache only supports stages with clip/inv loss weights disabled.')
    split_path = project_root / stage_config['split']

    allowed_trial_results = ['stable'] if args.only_stable else None
    dataset = CrossMediumSequenceDataset(
        project_root=project_root,
        split_path=split_path,
        subset=args.subset,
        num_frames_per_window=int(data_config['num_frames_per_window']),
        image_size=int(data_config['image_size']),
        roi=data_config.get('roi'),
        tactile_dt=float(data_config['tactile_dt']),
        acdc_alpha=float(data_config['ema_alpha_acdc']),
        expert_alpha=float(data_config['ema_alpha_expert']),
        normal_sign_table=data_config['normal_sign_table'],
        clip_mean=data_config.get('clip_mean'),
        clip_std=data_config.get('clip_std'),
        tactile_points_per_window=int(data_config.get('tactile_points_per_window')) if data_config.get('tactile_points_per_window') is not None else None,
        standardize_tactile=bool(data_config.get('standardize_tactile', False)),
        min_valid_ratio_video=float(data_config.get('min_valid_ratio_video', 0.0)),
        min_valid_ratio_tactile=float(data_config.get('min_valid_ratio_tactile', 0.0)),
        normalization_subset='train',
        normalization_trial_results=train_config.get('default_allowed_trial_results', {}).get('train'),
        tail_mode=stage_config.get('tail_mode', data_config.get('valid_tail_mode', 'all_valid')),
        allowed_trial_results=allowed_trial_results,
        visual_feature_cache_dir=data_config.get('visual_feature_cache_dir'),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_config['batch_size']),
        shuffle=False,
        num_workers=int(train_config['num_workers']),
        collate_fn=sequence_collate_fn,
    )

    device = torch.device(train_config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    model = CrossMediumSystem(model_config).to(device)
    load_info = load_model_weights(model, checkpoint_path, strict=True)
    phase_class_weights = build_phase_class_weights(train_config, device)
    metrics = run_model_epoch(
        model,
        loader,
        device=device,
        training=False,
        stage_config=stage_config,
        model_config=model_config,
        phase_class_weights=phase_class_weights,
        amp_enabled=bool(train_config.get('amp_enabled', False)) and device.type == 'cuda',
    )

    if args.output:
        output_path = resolve_path(project_root, args.output)
    else:
        suffix = '__stable' if args.only_stable else ''
        output_dir = project_root / 'evals' / stage_config['name']
        output_path = output_dir / f'{checkpoint_path.stem}__{args.subset}{suffix}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'stage_name': stage_config.get('name'),
        'subset': args.subset,
        'only_stable': bool(args.only_stable),
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_run_config_path': load_info.get('run_config_path'),
        'metrics': metrics,
    }
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print({'output_path': str(output_path), 'metrics': metrics})


if __name__ == '__main__':
    main()

