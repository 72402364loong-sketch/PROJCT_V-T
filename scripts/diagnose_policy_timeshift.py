from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_timeshift_metrics(rows: pd.DataFrame, max_shift_steps: int, stride_sec: float) -> pd.DataFrame:
    required = {'sample_id', 'finger', 'policy_timestamp', 'target_delta', 'pred_delta'}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f'Missing required prediction columns: {missing}')
    group_columns = ['sample_id', 'finger']
    if 'model' in rows.columns:
        group_columns.insert(0, 'model')
    metric_rows: list[dict[str, float | int]] = []
    for shift_steps in range(-max_shift_steps, max_shift_steps + 1):
        absolute_errors: list[np.ndarray] = []
        wrong_sign: list[np.ndarray] = []
        for _, group in rows.groupby(group_columns, sort=False):
            ordered = group.sort_values('policy_timestamp')
            target = ordered['target_delta'].to_numpy(dtype=float)
            prediction = ordered['pred_delta'].shift(shift_steps).to_numpy(dtype=float)
            valid = np.isfinite(target) & np.isfinite(prediction)
            if 'has_target' in ordered.columns:
                valid &= ordered['has_target'].to_numpy(dtype=bool)
            if not valid.any():
                continue
            absolute_errors.append(np.abs(prediction[valid] - target[valid]))
            directional = valid & (np.abs(target) > 1e-8) & (np.abs(prediction) > 1e-8)
            if directional.any():
                wrong_sign.append((np.sign(prediction[directional]) != np.sign(target[directional])).astype(float))
        errors = np.concatenate(absolute_errors) if absolute_errors else np.zeros(0, dtype=float)
        signs = np.concatenate(wrong_sign) if wrong_sign else np.zeros(0, dtype=float)
        metric_rows.append(
            {
                'shift_steps': shift_steps,
                'shift_sec': shift_steps * stride_sec,
                'shift_ms': shift_steps * stride_sec * 1000.0,
                'valid_count': int(errors.size),
                'delta_mae': float(errors.mean()) if errors.size else float('nan'),
                'wrong_sign_rate': float(signs.mean()) if signs.size else float('nan'),
                'directional_count': int(signs.size),
            }
        )
    return pd.DataFrame(metric_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate causal policy prediction/target time shifts.')
    parser.add_argument('--predictions-csv', required=True)
    parser.add_argument('--output-csv', required=True)
    parser.add_argument('--policy-stride-sec', type=float, default=0.05)
    parser.add_argument('--max-shift-steps', type=int, default=4)
    args = parser.parse_args()

    result = compute_timeshift_metrics(
        pd.read_csv(args.predictions_csv),
        max_shift_steps=max(0, args.max_shift_steps),
        stride_sec=float(args.policy_stride_sec),
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result.to_string(index=False))
    print(f'wrote: {output_path}')


if __name__ == '__main__':
    main()
