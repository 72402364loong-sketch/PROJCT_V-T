from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.data import write_preprocessed_outputs



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--config', default='configs/data/default.yaml')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()
    command = f"scripts/preprocess.py --project-root {args.project_root} --config {args.config}"
    if args.output_dir:
        command += f" --output-dir {args.output_dir}"
    samples, windows = write_preprocessed_outputs(
        args.project_root,
        config_path=args.config,
        output_dir=args.output_dir,
        generation_command=command,
    )
    print(f'samples: {len(samples)}')
    print(f'windows: {len(windows)}')


if __name__ == '__main__':
    main()

