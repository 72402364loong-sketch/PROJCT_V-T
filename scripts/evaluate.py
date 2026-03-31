from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.evaluation import evaluate_checkpoint, resolve_path


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
    payload = evaluate_checkpoint(
        project_root=project_root,
        stage=args.stage,
        checkpoint=args.checkpoint,
        subset=args.subset,
        only_stable=bool(args.only_stable),
    )

    checkpoint_path = resolve_path(project_root, args.checkpoint)
    if args.output:
        output_path = resolve_path(project_root, args.output)
    else:
        suffix = '__stable' if args.only_stable else ''
        output_dir = project_root / 'evals' / payload['stage_name']
        output_path = output_dir / f'{checkpoint_path.stem}__{args.subset}{suffix}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print({'output_path': str(output_path), 'metrics': payload['metrics']})


if __name__ == '__main__':
    main()
