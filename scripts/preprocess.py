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
    args = parser.parse_args()
    samples, windows = write_preprocessed_outputs(args.project_root)
    print(f'samples: {len(samples)}')
    print(f'windows: {len(windows)}')


if __name__ == '__main__':
    main()

