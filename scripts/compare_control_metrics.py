from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.evaluation import evaluate_checkpoint, resolve_path


SUMMARY_KEYS = [
    'overall_mae',
    'stable_mae',
    'interface_mae',
    'interface_bias',
    'interface_hit_rate_100',
    'interface_hit_rate_200',
    'interface_hit_rate_300',
    'medium_f1_interface',
    'medium_macro_f1',
    'joint_score',
]


def build_summary_row(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        'label': label,
        'stage_name': payload['stage_name'],
        'subset': payload['subset'],
        'only_stable': payload['only_stable'],
        'checkpoint_path': payload['checkpoint_path'],
    }
    metrics = payload['metrics']
    for key in SUMMARY_KEYS:
        row[key] = metrics.get(key)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--only-stable', action='store_true')
    parser.add_argument('--output', default=None)
    parser.add_argument(
        '--entry',
        action='append',
        nargs=3,
        metavar=('LABEL', 'STAGE', 'CHECKPOINT'),
        required=True,
        help='One comparison entry: label, stage yaml path, checkpoint path.',
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    payloads: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for label, stage_path, checkpoint_path in args.entry:
        payload = evaluate_checkpoint(
            project_root=project_root,
            stage=stage_path,
            checkpoint=checkpoint_path,
            subset=args.subset,
            only_stable=bool(args.only_stable),
        )
        payload['label'] = label
        payloads.append(payload)
        summary_rows.append(build_summary_row(label, payload))

    if args.output:
        output_path = resolve_path(project_root, args.output)
    else:
        suffix = '__stable' if args.only_stable else ''
        output_path = project_root / 'evals' / 'comparisons' / f'control_compare__{args.subset}{suffix}.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == '.json':
        with output_path.open('w', encoding='utf-8') as handle:
            json.dump(payloads, handle, ensure_ascii=False, indent=2)
    else:
        write_csv(output_path, summary_rows)

    print({'output_path': str(output_path), 'rows': summary_rows})


if __name__ == '__main__':
    main()
