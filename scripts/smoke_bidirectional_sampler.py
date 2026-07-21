from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.data import DirectionAwareObjectBatchSampler
from scripts.train import make_dataset, make_train_loader


def compact_windows(dataset) -> None:
    sample_by_id = {str(sample['sample_id']): sample for sample in dataset.sample_records}
    compacted: dict[str, list[dict]] = {}
    for sample_id, windows in dataset.windows_by_sample.items():
        sample = sample_by_id[str(sample_id)]
        reference_start = float(sample['reference_start_time'])
        reference_end = float(sample['reference_end_time'])
        reference = [
            window for window in windows
            if reference_start <= float(window['policy_timestamp']) < reference_end
        ][-2:]
        interface = [window for window in windows if str(window['phase_label']) == 'Interface'][:2]
        selected = [*reference, *interface, windows[-1]]
        deduplicated = {str(window['window_id']): window for window in selected}
        compacted[str(sample_id)] = sorted(deduplicated.values(), key=lambda row: float(row['window_start']))
    dataset.windows_by_sample = compacted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument(
        '--output',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_d_sampler_smoke.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    data_config = deep_update(
        load_yaml(root / 'configs' / 'data' / 'default.yaml'),
        load_yaml(root / 'configs' / 'data' / 'policy_20hz_bidirectional_v4.yaml'),
    )
    data_config.update({
        'num_frames_per_window': 2,
        'image_size': 64,
        'roi': None,
        'tactile_points_per_window': 8,
        'visual_feature_cache_dir': None,
    })
    train_config = load_yaml(root / 'configs' / 'train' / 'bidirectional_v1.yaml')
    train_config.update({'num_workers': 0, 'pin_memory': False, 'persistent_workers': False})
    stage_config = {'tail_mode': 'all_valid'}
    split_path = root / data_config['split_path']

    dataset = make_dataset(root, split_path, 'train', data_config, train_config, stage_config)
    tactile_cache_path = dataset._stats_cache_path()
    compact_windows(dataset)
    loader = make_train_loader(dataset, train_config)
    batch = next(iter(loader))

    audit_sampler = DirectionAwareObjectBatchSampler(
        dataset,
        batch_size=int(train_config['batch_size']),
        direction_values=tuple(train_config['direction_values']),
        balance_mode=train_config['direction_balance_mode'],
        interface_alpha=float(train_config.get('interface_sampler_alpha', 0.0)),
        min_interface_expert_windows=int(train_config.get('interface_sampler_min_expert_windows', 1)),
        seed=int(train_config.get('seed', 42)),
    )
    list(audit_sampler)

    direction_counts = {
        direction: batch['directions'].count(direction)
        for direction in ['W2A', 'A2W']
    }
    issues: list[dict[str, str]] = []
    if direction_counts != {'W2A': 4, 'A2W': 4}:
        issues.append({'field': 'batch.direction', 'message': f'Unexpected batch counts: {direction_counts}.'})
    if sorted(batch['direction_ids'].tolist()) != [0, 0, 0, 0, 1, 1, 1, 1]:
        issues.append({'field': 'batch.direction_ids', 'message': 'direction_ids are not balanced 4/4.'})
    if not tactile_cache_path.exists():
        issues.append({'field': 'tactile_cache', 'message': f'Missing cache: {tactile_cache_path}.'})
    else:
        with tactile_cache_path.open('r', encoding='utf-8') as handle:
            tactile_cache = json.load(handle)
        if tactile_cache.get('annotations_sha256') != dataset.annotations_sha256:
            issues.append({'field': 'tactile_cache.annotations_sha256', 'message': 'Annotation hash mismatch.'})
        if tactile_cache.get('samples_sha256') != dataset.samples_sha256:
            issues.append({'field': 'tactile_cache.samples_sha256', 'message': 'Processed samples hash mismatch.'})
    if not Path(data_config['physical_attribute_norm_stats_path']).is_absolute():
        physical_cache_path = root / data_config['physical_attribute_norm_stats_path']
    else:
        physical_cache_path = Path(data_config['physical_attribute_norm_stats_path'])
    if not physical_cache_path.exists():
        issues.append({'field': 'physical_cache', 'message': f'Missing cache: {physical_cache_path}.'})

    report = {
        'dataset_version': data_config['dataset_version'],
        'schema_version': data_config['schema_version'],
        'split_version': data_config['split_version'],
        'annotations_sha256': dataset.annotations_sha256,
        'samples_sha256': dataset.samples_sha256,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'train_sample_count': int(len(dataset)),
        'batch_size': int(len(batch['directions'])),
        'batch_direction_counts': direction_counts,
        'batch_object_ids': batch['object_ids'],
        'batch_directions': batch['directions'],
        'batch_direction_ids': batch['direction_ids'].tolist(),
        'epoch_sampler_summary': audit_sampler.last_epoch_summary,
        'tactile_stats_cache_path': str(tactile_cache_path),
        'physical_stats_cache_path': str(physical_cache_path),
        'error_count': len(issues),
        'issues': issues,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(output_path), **report}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
