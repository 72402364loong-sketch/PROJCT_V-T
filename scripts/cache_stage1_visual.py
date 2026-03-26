from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import resolve_visual_feature_cache_dir, write_sample_visual_feature_cache
from cmg.data.splits import resolve_sample_ids
from cmg.data.video import load_frames_by_indices, sample_frame_indices
from cmg.models.modules import VisualEncoder


def resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_configs(project_root: Path, stage_path: Path) -> tuple[dict, dict, dict]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    model_config = load_yaml(project_root / 'configs' / 'model' / 'default.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    model_config = deep_update(model_config, stage_config.get('model', {}))
    return data_config, model_config, stage_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='all', choices=['all', 'train', 'val', 'test'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--windows-per-batch', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    if stage_path is None:
        raise ValueError('stage path is required')
    data_config, model_config, stage_config = load_configs(project_root, stage_path)

    output_dir = resolve_visual_feature_cache_dir(
        project_root,
        args.output_dir if args.output_dir is not None else data_config.get('visual_feature_cache_dir'),
    )
    if output_dir is None:
        raise ValueError('No visual feature cache directory is configured. Pass --output-dir or set data.visual_feature_cache_dir.')
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')
    if args.subset == 'all':
        sample_ids = sorted(windows['sample_id'].unique().tolist())
    else:
        split_path = project_root / stage_config['split']
        sample_ids = resolve_sample_ids(samples, split_path, subset=args.subset)

    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    encoder = VisualEncoder(model_config['visual']).to(device)
    encoder.eval()

    num_frames_per_window = int(data_config['num_frames_per_window'])
    image_size = int(data_config['image_size'])
    roi = data_config.get('roi')
    clip_mean = None if data_config.get('clip_mean') is None else torch.tensor(data_config['clip_mean'], dtype=torch.float32).view(1, 3, 1, 1)
    clip_std = None if data_config.get('clip_std') is None else torch.tensor(data_config['clip_std'], dtype=torch.float32).view(1, 3, 1, 1)
    windows_per_batch = max(1, int(args.windows_per_batch or model_config['visual'].get('max_windows_per_encode', 8)))

    processed = 0
    skipped = 0
    with torch.no_grad():
        for sample_id in sample_ids:
            output_path = output_dir / f'{sample_id}.npz'
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue

            sample_rows = samples.loc[samples['sample_id'] == sample_id]
            if sample_rows.empty:
                continue
            sample = sample_rows.iloc[0]
            sample_windows = windows.loc[windows['sample_id'] == sample_id].sort_values('window_start').reset_index(drop=True)
            if sample_windows.empty:
                continue

            video_path = Path(str(sample['video_path']))
            if not video_path.is_absolute():
                if video_path.parts and video_path.parts[0] == 'data':
                    video_path = project_root / video_path
                else:
                    video_path = project_root / 'data' / video_path

            window_specs: list[tuple[str, list[int], list[bool]]] = []
            sampled_frame_indices: list[int] = []
            for window in sample_windows.to_dict('records'):
                all_frame_indices = json.loads(window['video_frame_indices_json'])
                sampled_indices, frame_mask = sample_frame_indices(all_frame_indices, num_frames_per_window)
                window_specs.append((str(window['window_id']), sampled_indices, frame_mask))
                sampled_frame_indices.extend(sampled_indices)

            frame_map = load_frames_by_indices(
                video_path=video_path,
                frame_indices=sampled_frame_indices,
                image_size=image_size,
                roi=roi,
            )

            feature_chunks: list[np.ndarray] = []
            for start in range(0, len(window_specs), windows_per_batch):
                stop = min(len(window_specs), start + windows_per_batch)
                chunk_specs = window_specs[start:stop]
                video_batch: list[torch.Tensor] = []
                mask_batch: list[torch.Tensor] = []
                for _, sampled_indices, frame_mask in chunk_specs:
                    frame_array = np.stack([frame_map[int(frame_index)] for frame_index in sampled_indices], axis=0)
                    frame_tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2)
                    if clip_mean is not None and clip_std is not None:
                        frame_tensor = (frame_tensor - clip_mean) / clip_std
                    video_batch.append(frame_tensor)
                    mask_batch.append(torch.tensor(frame_mask, dtype=torch.bool))
                video_tensor = torch.stack(video_batch, dim=0).to(device)
                frame_mask_tensor = torch.stack(mask_batch, dim=0).to(device)
                window_features, _ = encoder(video_tensor, frame_mask_tensor)
                feature_chunks.append(window_features.cpu().numpy().astype(np.float32))

            visual_features = np.concatenate(feature_chunks, axis=0)
            write_sample_visual_feature_cache(
                project_root,
                output_dir,
                str(sample_id),
                [window_id for window_id, _, _ in window_specs],
                visual_features,
            )
            processed += 1
            print({'sample_id': str(sample_id), 'windows': int(len(window_specs)), 'output_path': str(output_path)})

    print({'processed_samples': processed, 'skipped_samples': skipped, 'output_dir': str(output_dir), 'device': str(device)})


if __name__ == '__main__':
    main()