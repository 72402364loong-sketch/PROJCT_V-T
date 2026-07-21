from __future__ import annotations

import argparse
import gc
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
from scripts.train import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/data/policy_20hz_bidirectional_v4.yaml')
    parser.add_argument('--processed-dir', default=None)
    parser.add_argument(
        '--output',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_c_dataset_smoke.json',
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    data_config = deep_update(
        load_yaml(project_root / 'configs' / 'data' / 'default.yaml'),
        load_yaml(config_path),
    )
    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
        if not processed_dir.is_absolute():
            processed_dir = project_root / processed_dir
        data_config['processed_dir'] = str(processed_dir)
        data_config['samples_path'] = str(processed_dir / 'samples.csv')
        data_config['windows_path'] = str(processed_dir / 'windows.csv')
        data_config['physical_attribute_norm_stats_path'] = str(
            processed_dir / 'stats' / 'physical_attribute_norm_stats.json'
        )
    split_path = project_root / data_config['split_path']
    train_config = {
        'default_allowed_trial_results': {
            'train': ['stable'],
            'val': ['stable'],
            'test': ['stable'],
        }
    }
    stage_config = {'tail_mode': 'all_valid'}

    subsets: dict[str, dict[str, object]] = {}
    for subset in ['train', 'val', 'test']:
        dataset = make_dataset(project_root, split_path, subset, data_config, train_config, stage_config)
        subsets[subset] = {
            'sample_count': int(len(dataset)),
            'window_count': int(len(dataset.windows)),
            'direction_counts': {
                str(key): int(value)
                for key, value in dataset.samples['direction'].value_counts().to_dict().items()
            },
            'split_values': sorted(dataset.samples['split'].astype(str).unique().tolist()),
            'trial_result_values': sorted(dataset.samples['trial_result'].astype(str).unique().tolist()),
        }
        del dataset
        gc.collect()

    report = {
        'dataset_version': data_config['dataset_version'],
        'schema_version': data_config['schema_version'],
        'split_version': data_config['split_version'],
        'processed_dir': str(data_config['processed_dir']),
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'subsets': subsets,
        'error_count': 0,
        'issues': [],
    }
    expected_samples = {
        'train': {'W2A': 143, 'A2W': 150},
        'val': {'W2A': 44, 'A2W': 39},
        'test': {'W2A': 75, 'A2W': 73},
    }
    for subset, expected in expected_samples.items():
        if subsets[subset]['direction_counts'] != expected:
            report['issues'].append({
                'subset': subset,
                'message': f'Expected direction counts {expected}, got {subsets[subset]["direction_counts"]}.',
            })
        if subsets[subset]['split_values'] != [subset]:
            report['issues'].append({
                'subset': subset,
                'message': f'Unexpected split values: {subsets[subset]["split_values"]}.',
            })
        if subsets[subset]['trial_result_values'] != ['stable']:
            report['issues'].append({
                'subset': subset,
                'message': f'Unexpected trial results: {subsets[subset]["trial_result_values"]}.',
            })
    report['error_count'] = len(report['issues'])

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(output_path), **report}, ensure_ascii=False, indent=2))
    if report['error_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
