from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.data.tactile import (
    build_tactile_time_axis,
    compute_clean_force_curve,
    load_tactile_array,
    normalize_normal_sign_table,
)


AUDIT_OBJECTS = {
    'train': 'OBJ001',
    'val': 'OBJ005',
    'test': 'OBJ007',
}


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'data':
        return root / path
    return root / 'data' / path


def compressed_phases(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if not result or result[-1] != value:
            result.append(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/data/policy_20hz_bidirectional_v4.yaml')
    parser.add_argument(
        '--output-image',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_e_a2w_phase_reference_spotcheck.png',
    )
    parser.add_argument(
        '--output-report',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/phase_e_a2w_phase_reference_spotcheck.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}

    samples = pd.read_csv(resolve_path(root, config['samples_path']))
    selected_rows: list[pd.Series] = []
    for subset, object_id in AUDIT_OBJECTS.items():
        candidates = samples.loc[
            samples['split'].eq(subset)
            & samples['object_id'].eq(object_id)
            & samples['direction'].eq('A2W')
            & samples['trial_result'].eq('stable')
            & samples['t_if_enter'].notna()
            & samples['t_if_exit'].notna()
        ].sort_values('sample_id')
        if candidates.empty:
            raise RuntimeError(f'No complete stable A2W sample for {subset}/{object_id}.')
        selected_rows.append(candidates.iloc[0])

    selected_ids = {str(row['sample_id']) for row in selected_rows}
    window_chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        resolve_path(root, config['windows_path']),
        usecols=['sample_id', 'policy_timestamp', 'phase_label'],
        chunksize=50000,
    ):
        selected = chunk.loc[chunk['sample_id'].astype(str).isin(selected_ids)]
        if not selected.empty:
            window_chunks.append(selected)
    selected_windows = pd.concat(window_chunks, ignore_index=True)

    sign_table = normalize_normal_sign_table(config['normal_sign_table'])
    tactile_dt = float(config['tactile_dt'])
    reference_duration = float(config['reference']['duration_sec'])
    reference_statistic = str(config['reference']['statistic']).strip().lower()
    figure, axes = plt.subplots(len(selected_rows), 1, figsize=(13, 10), constrained_layout=False)
    colors = ['#0072B2', '#D55E00', '#009E73']
    issues: list[dict[str, str]] = []
    records: list[dict[str, object]] = []

    for axis, sample in zip(axes, selected_rows):
        sample_id = str(sample['sample_id'])
        enter = float(sample['t_if_enter'])
        exit_time = float(sample['t_if_exit'])
        start = float(sample['t_start'])
        end = float(sample['t_end'])
        reference_start = float(sample['reference_start_time'])
        reference_end = float(sample['reference_end_time'])
        tactile = load_tactile_array(resolve_path(root, str(sample['tactile_path'])))
        force_curve = compute_clean_force_curve(
            tactile,
            dt=tactile_dt,
            sync_offset_sec=float(sample['sync_offset_sec']),
            contact_time=None,
            alpha=float(config.get('ema_alpha_expert', 0.1)),
            normal_sign_table=sign_table,
            smoothing_mode='none',
            baseline_mode='none',
            baseline_window_sec=float(config.get('expert_force_baseline_window_sec', 0.5)),
            force_target_mode='per_finger_z_abs_mean',
        )
        time_axis = build_tactile_time_axis(
            len(force_curve),
            dt=tactile_dt,
            offset_sec=float(sample['sync_offset_sec']),
        )
        reference_mask = (time_axis >= reference_start) & (time_axis < reference_end)
        if reference_statistic == 'median':
            reference = np.median(force_curve[reference_mask], axis=0)
        else:
            reference = np.mean(force_curve[reference_mask], axis=0)

        phases = compressed_phases(
            selected_windows.loc[selected_windows['sample_id'].astype(str).eq(sample_id)]
            .sort_values('policy_timestamp')['phase_label'].astype(str).tolist()
        )
        expected_reference_start = max(start, enter - reference_duration)
        checks = {
            'direction_is_a2w': str(sample['direction']) == 'A2W',
            'source_and_reference_are_air': (
                str(sample['source_medium']) == 'air' and str(sample['reference_medium']) == 'air'
            ),
            'baseline_is_none': str(sample['force_baseline_mode']) == 'none',
            'reference_ends_at_interface_enter': bool(np.isclose(reference_end, enter, atol=1e-6)),
            'reference_duration_contract': bool(np.isclose(reference_start, expected_reference_start, atol=1e-6)),
            'reference_has_tactile_points': bool(reference_mask.any()),
            'phase_order_air_interface_water': phases == ['Air', 'Interface', 'Water'],
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            issues.append({'field': sample_id, 'message': f'Failed checks: {failed}.'})

        view_start = max(start, reference_start - 2.0)
        view_end = min(end, exit_time + 3.0)
        view_mask = (time_axis >= view_start) & (time_axis <= view_end)
        axis.axvspan(view_start, enter, color='#E8E8E8', alpha=0.65, label='Air')
        axis.axvspan(enter, exit_time, color='#F0E442', alpha=0.20, label='Interface')
        axis.axvspan(exit_time, view_end, color='#56B4E9', alpha=0.18, label='Water')
        axis.axvspan(reference_start, reference_end, color='#E69F00', alpha=0.35, label='Reference interval')
        for finger, color in enumerate(colors):
            axis.plot(time_axis[view_mask], force_curve[view_mask, finger], color=color, linewidth=1.2, label=f'finger{finger}')
            axis.hlines(
                float(reference[finger]),
                reference_start,
                reference_end,
                color=color,
                linewidth=2.2,
                linestyle='--',
            )
        axis.axvline(enter, color='#CC7900', linewidth=1.0)
        axis.axvline(exit_time, color='#0072B2', linewidth=1.0)
        axis.set_xlim(view_start, view_end)
        axis.set_ylabel('Raw force')
        axis.set_title(f'{sample_id} | {sample["object_id"]} | {sample["split"]} | A2W')
        axis.grid(True, alpha=0.2)

        records.append({
            'sample_id': sample_id,
            'object_id': str(sample['object_id']),
            'split': str(sample['split']),
            'direction': str(sample['direction']),
            'source_medium': str(sample['source_medium']),
            'target_medium': str(sample['target_medium']),
            'reference_medium': str(sample['reference_medium']),
            'reference_start_time': reference_start,
            'reference_end_time': reference_end,
            'interface_enter_time': enter,
            'interface_exit_time': exit_time,
            'reference_tactile_point_count': int(reference_mask.sum()),
            'reference_force_per_finger': np.asarray(reference, dtype=np.float32).tolist(),
            'observed_phase_order': phases,
            'checks': checks,
        })

    axes[-1].set_xlabel('Time (s)')
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.subplots_adjust(top=0.89, bottom=0.07, hspace=0.30)
    figure.legend(
        unique.values(),
        unique.keys(),
        loc='upper center',
        bbox_to_anchor=(0.5, 0.955),
        ncol=8,
        frameon=False,
    )
    figure.suptitle('A2W phase and source-medium Reference audit', fontsize=15, y=0.99)

    image_path = resolve_path(root, args.output_image)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(image_path, dpi=160)
    plt.close(figure)

    report = {
        'release': 'bidirectional_v1__bidirectional-causal-v4',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'selection_rule': AUDIT_OBJECTS,
        'sample_count': len(records),
        'samples': records,
        'image_path': str(image_path),
        'error_count': len(issues),
        'issues': issues,
    }
    report_path = resolve_path(root, args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output_path': str(report_path), **report}, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
