from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.evaluation import resolve_path
from cmg.data.splits import resolve_sample_ids


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(parsed):
        return None
    return parsed


def smooth_ramp(progress: float, ramp: str) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    if str(ramp).lower() in {'cosine', 'cos'}:
        return float(0.5 - 0.5 * np.cos(np.pi * progress))
    return progress


def compute_soft_gate_target(
    *,
    window: dict[str, Any],
    sample: dict[str, Any],
    pre_sec: float,
    post_sec: float,
    ramp: str,
) -> float:
    t_if_enter = parse_optional_float(sample.get('t_if_enter'))
    t_if_exit = parse_optional_float(sample.get('t_if_exit'))
    if t_if_enter is None or t_if_exit is None or t_if_exit <= t_if_enter:
        return 1.0 if str(window.get('phase_label')) == 'Interface' else 0.0

    center = parse_optional_float(window.get('window_center'))
    if center is None:
        start = parse_optional_float(window.get('window_start'))
        end = parse_optional_float(window.get('window_end'))
        if start is None or end is None:
            return 1.0 if str(window.get('phase_label')) == 'Interface' else 0.0
        center = 0.5 * (start + end)

    rise_start = t_if_enter - pre_sec
    decay_end = t_if_exit + post_sec
    if center < rise_start:
        return 0.0
    if center < t_if_enter:
        return 1.0 if pre_sec <= 0.0 else smooth_ramp((center - rise_start) / pre_sec, ramp)
    if center <= t_if_exit:
        return 1.0
    if center <= decay_end:
        return 0.0 if post_sec <= 0.0 else 1.0 - smooth_ramp((center - t_if_exit) / post_sec, ramp)
    return 0.0


def load_stage_configs(project_root: Path, stage_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    return data_config, stage_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='train', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--pre-sec', type=float, default=None)
    parser.add_argument('--post-sec', type=float, default=None)
    parser.add_argument('--ramp', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    data_config, stage_config = load_stage_configs(project_root, stage_path)

    samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')
    if args.subset != 'all':
        sample_ids = set(resolve_sample_ids(samples, project_root / stage_config['split'], subset=args.subset))
        samples = samples.loc[samples['sample_id'].isin(sample_ids)].copy()
        windows = windows.loc[windows['sample_id'].isin(sample_ids)].copy()

    sample_by_id = {str(row['sample_id']): row for row in samples.to_dict('records')}
    pre_sec = float(data_config.get('soft_gate_pre_sec', 0.0) if args.pre_sec is None else args.pre_sec)
    post_sec = float(data_config.get('soft_gate_post_sec', 0.0) if args.post_sec is None else args.post_sec)
    ramp = str(data_config.get('soft_gate_ramp', 'linear') if args.ramp is None else args.ramp)

    rows: list[dict[str, Any]] = []
    for window in windows.sort_values(['sample_id', 'window_start']).to_dict('records'):
        sample = sample_by_id.get(str(window['sample_id']))
        if sample is None:
            continue
        target = compute_soft_gate_target(
            window=window,
            sample=sample,
            pre_sec=max(0.0, pre_sec),
            post_sec=max(0.0, post_sec),
            ramp=ramp,
        )
        rows.append(
            {
                'sample_id': str(window['sample_id']),
                'object_id': str(window['object_id']),
                'window_id': str(window['window_id']),
                'window_start': float(window['window_start']),
                'window_end': float(window['window_end']),
                'window_center': float(window['window_center']),
                'phase_label': str(window['phase_label']),
                'is_stable_mask': int(window['is_stable_mask']),
                't_if_enter': parse_optional_float(sample.get('t_if_enter')),
                't_if_exit': parse_optional_float(sample.get('t_if_exit')),
                'soft_gate_target': float(target),
            }
        )

    output_dir = resolve_path(project_root, args.output_dir) if args.output_dir else project_root / 'data' / 'processed' / 'soft_gate_targets'
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_label = args.subset
    output_path = output_dir / f"{stage_config['name']}__{subset_label}.csv"
    summary_path = output_dir / f"{stage_config['name']}__{subset_label}.json"

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False, encoding='utf-8-sig')
    summary = {
        'stage_name': str(stage_config['name']),
        'subset': args.subset,
        'pre_sec': max(0.0, pre_sec),
        'post_sec': max(0.0, post_sec),
        'ramp': ramp,
        'row_count': int(len(frame)),
        'nonzero_count': int((frame['soft_gate_target'] > 0.0).sum()) if not frame.empty else 0,
        'full_gate_count': int((frame['soft_gate_target'] >= 1.0).sum()) if not frame.empty else 0,
        'output_path': str(output_path),
    }
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print({'output_path': str(output_path), 'summary_path': str(summary_path), **summary})


if __name__ == '__main__':
    main()
