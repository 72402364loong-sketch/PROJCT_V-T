from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import CrossMediumSequenceDataset, ObjectAwareBatchSampler, sequence_collate_fn
from cmg.losses import compute_losses
from cmg.models import CrossMediumSystem
from cmg.training import build_phase_class_weights


def load_configs(project_root: Path) -> tuple[dict, dict, dict, dict]:
    stage_path = project_root / 'configs' / 'stages' / 'stage1_perception_fold1.yaml'
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    return data_config, model_config, train_config, stage_config


def make_dataset(project_root: Path, split_path: Path, subset: str, data_config: dict, train_config: dict, stage_config: dict) -> CrossMediumSequenceDataset:
    allowed_trial_results = train_config.get('default_allowed_trial_results', {}).get(subset)
    return CrossMediumSequenceDataset(
        project_root=project_root,
        split_path=split_path,
        subset=subset,
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
    )


def main() -> None:
    project_root = ROOT
    data_config, model_config, train_config, stage_config = load_configs(project_root)
    split_path = project_root / stage_config['split']
    train_dataset = make_dataset(project_root, split_path, 'train', data_config, train_config, stage_config)
    batch_sampler = ObjectAwareBatchSampler(
        train_dataset,
        batch_size=int(train_config['batch_size']),
        samples_per_object=int(train_config.get('samples_per_object', 2)),
        seed=int(train_config.get('seed', 42)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
        collate_fn=sequence_collate_fn,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CrossMediumSystem(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config['learning_rate']))
    phase_class_weights = build_phase_class_weights(train_config, device)

    print({'device': str(device), 'dataset_len': len(train_dataset), 'batch_size': int(train_config['batch_size'])}, flush=True)

    for step_idx, batch in enumerate(train_loader):
        sample_ids = batch['sample_ids']
        print({'step': step_idx, 'sample_ids': sample_ids}, flush=True)
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=True):
            outputs = model(batch)
            losses = compute_losses(
                outputs,
                batch,
                loss_weights=stage_config.get('loss_weights', {}),
                temperature_clip=float(model_config['losses']['temperature_clip']),
                temperature_inv=float(model_config['losses']['temperature_inv']),
                phase_class_weights=phase_class_weights,
            )
        losses['total'].backward()
        optimizer.step()
        print({'step': step_idx, 'loss_total': float(losses['total'].detach().cpu().item())}, flush=True)
        if step_idx >= 2:
            break


if __name__ == '__main__':
    main()
