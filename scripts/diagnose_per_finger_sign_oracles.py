from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.evaluation import prepare_evaluation_context, resolve_path
from cmg.training import move_to_device


def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().mean().item())


def _safe_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0
    return float(selected.float().sum().item())


def _safe_count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _safe_rate(mask: torch.Tensor, denom_mask: torch.Tensor) -> float:
    denom = _safe_count(denom_mask)
    if denom == 0:
        return 0.0
    return float(mask[denom_mask].float().mean().item())


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), q).item())


def _std(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.float().std(unbiased=False).item())


def _labels_from_delta(delta: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    tau_view = tau.reshape(1, -1).to(device=delta.device, dtype=delta.dtype)
    return torch.where(
        delta > tau_view,
        torch.ones_like(delta, dtype=torch.int64),
        torch.where(delta < -tau_view, -torch.ones_like(delta, dtype=torch.int64), torch.zeros_like(delta, dtype=torch.int64)),
    )


def _estimate_tau(
    tau_source_delta: torch.Tensor,
    stable_mask: torch.Tensor,
    *,
    percentile: float,
    std_multiplier: float,
    minimum: float,
    source: str,
) -> dict[str, Any]:
    finger_count = tau_source_delta.shape[-1]
    q = percentile / 100.0
    tau_values: list[float] = []
    per_finger: list[dict[str, float | int]] = []
    for finger_index in range(finger_count):
        values = tau_source_delta[:, finger_index].abs()
        mask = stable_mask[:, finger_index] & torch.isfinite(values)
        selected = values[mask]
        p_value = _quantile(selected, q)
        std_value = _std(selected)
        tau_value = max(float(minimum), p_value, float(std_multiplier) * std_value)
        tau_values.append(tau_value)
        per_finger.append(
            {
                'finger_index': finger_index,
                'stable_count': int(selected.numel()),
                f'p{percentile:g}_abs_stable_delta': p_value,
                'std_abs_stable_delta': std_value,
                'tau': tau_value,
            }
        )
    return {
        'source': source,
        'percentile': percentile,
        'std_multiplier': std_multiplier,
        'minimum': minimum,
        'per_finger': per_finger,
        'tensor': torch.tensor(tau_values, dtype=tau_source_delta.dtype),
    }


def _classification_metrics(
    *,
    target_delta: torch.Tensor,
    pred_delta: torch.Tensor,
    valid_mask: torch.Tensor,
    tau: torch.Tensor,
    large_delta_threshold: float,
) -> dict[str, Any]:
    target_label = _labels_from_delta(target_delta, tau)
    pred_label = _labels_from_delta(pred_delta, tau)

    directional = valid_mask & (target_label != 0)
    same = directional & (pred_label == target_label)
    opposite = directional & (pred_label == -target_label)
    pred_near_zero = directional & (pred_label == 0)
    miss = directional & (pred_label != target_label)

    large = valid_mask & (target_delta.abs() >= float(large_delta_threshold))
    large_directional = large & (target_label != 0)
    large_same = large_directional & (pred_label == target_label)
    large_opposite = large_directional & (pred_label == -target_label)
    large_pred_near_zero = large_directional & (pred_label == 0)
    large_miss = large_directional & (pred_label != target_label)

    result: dict[str, Any] = {
        'directional_count': _safe_count(directional),
        'directional_hit_rate': _safe_rate(same, directional),
        'opposite_sign_rate': _safe_rate(opposite, directional),
        'near_zero_pred_rate': _safe_rate(pred_near_zero, directional),
        'directional_miss_rate': _safe_rate(miss, directional),
        'near_zero_target_count': _safe_count(valid_mask & (target_label == 0)),
        'large_delta_threshold': float(large_delta_threshold),
        'large_directional_count': _safe_count(large_directional),
        'large_directional_hit_rate': _safe_rate(large_same, large_directional),
        'large_opposite_sign_rate': _safe_rate(large_opposite, large_directional),
        'large_near_zero_pred_rate': _safe_rate(large_pred_near_zero, large_directional),
        'large_directional_miss_rate': _safe_rate(large_miss, large_directional),
    }

    pos_mask = large_directional & (target_label > 0)
    neg_mask = large_directional & (target_label < 0)
    pos_opposite = large_opposite & (target_label > 0)
    neg_opposite = large_opposite & (target_label < 0)
    pos_miss = large_miss & (target_label > 0)
    neg_miss = large_miss & (target_label < 0)
    pos_opposite_rate = _safe_rate(pos_opposite, pos_mask)
    neg_opposite_rate = _safe_rate(neg_opposite, neg_mask)
    pos_miss_rate = _safe_rate(pos_miss, pos_mask)
    neg_miss_rate = _safe_rate(neg_miss, neg_mask)
    balance_denominator = max(1, int(_safe_count(pos_mask) > 0) + int(_safe_count(neg_mask) > 0))
    result.update(
        {
            'large_pos_count': _safe_count(pos_mask),
            'large_neg_count': _safe_count(neg_mask),
            'large_pos_opposite_sign_rate': pos_opposite_rate,
            'large_neg_opposite_sign_rate': neg_opposite_rate,
            'large_balanced_opposite_sign_rate': (
                (pos_opposite_rate if _safe_count(pos_mask) > 0 else 0.0)
                + (neg_opposite_rate if _safe_count(neg_mask) > 0 else 0.0)
            )
            / balance_denominator,
            'large_pos_directional_miss_rate': pos_miss_rate,
            'large_neg_directional_miss_rate': neg_miss_rate,
            'large_balanced_directional_miss_rate': (
                (pos_miss_rate if _safe_count(pos_mask) > 0 else 0.0)
                + (neg_miss_rate if _safe_count(neg_mask) > 0 else 0.0)
            )
            / balance_denominator,
        }
    )

    per_finger: list[dict[str, Any]] = []
    for finger_index in range(target_delta.shape[-1]):
        finger_mask = torch.zeros_like(valid_mask)
        finger_mask[:, finger_index] = True
        directional_i = directional & finger_mask
        large_directional_i = large_directional & finger_mask
        per_finger.append(
            {
                'finger_index': finger_index,
                'directional_count': _safe_count(directional_i),
                'directional_hit_rate': _safe_rate(same, directional_i),
                'opposite_sign_rate': _safe_rate(opposite, directional_i),
                'near_zero_pred_rate': _safe_rate(pred_near_zero, directional_i),
                'directional_miss_rate': _safe_rate(miss, directional_i),
                'large_directional_count': _safe_count(large_directional_i),
                'large_directional_hit_rate': _safe_rate(large_same, large_directional_i),
                'large_opposite_sign_rate': _safe_rate(large_opposite, large_directional_i),
                'large_near_zero_pred_rate': _safe_rate(large_pred_near_zero, large_directional_i),
                'large_directional_miss_rate': _safe_rate(large_miss, large_directional_i),
            }
        )
    result['per_finger'] = per_finger
    return result


def _force_metrics(
    *,
    pred_force: torch.Tensor,
    pred_delta: torch.Tensor,
    gate: torch.Tensor,
    target_force: torch.Tensor,
    target_delta: torch.Tensor,
    valid_mask: torch.Tensor,
    interface_mask: torch.Tensor,
    stable_window_mask: torch.Tensor,
    tau: torch.Tensor,
    large_delta_threshold: float,
) -> dict[str, Any]:
    overall_mask = valid_mask
    interface_valid = valid_mask & interface_mask
    stable_valid = valid_mask & stable_window_mask
    large_valid = interface_valid & (target_delta.abs() >= float(large_delta_threshold))
    error = pred_force - target_force
    delta_error = pred_delta - target_delta
    gated_delta = gate * pred_delta
    leakage_mask = stable_window_mask & torch.isfinite(gated_delta)
    return {
        'overall_mae': _safe_mean(error.abs(), overall_mask),
        'overall_count': _safe_count(overall_mask),
        'interface_mae': _safe_mean(error.abs(), interface_valid),
        'interface_count': _safe_count(interface_valid),
        'stable_mae': _safe_mean(error.abs(), stable_valid),
        'stable_count': _safe_count(stable_valid),
        'delta_interface_mae': _safe_mean(delta_error.abs(), interface_valid),
        'delta_interface_bias': _safe_mean(delta_error, interface_valid),
        'pred_delta_abs_interface_mean': _safe_mean(pred_delta.abs(), interface_valid),
        'target_delta_abs_interface_mean': _safe_mean(target_delta.abs(), interface_valid),
        'large_delta_interface_mae': _safe_mean(error.abs(), large_valid),
        'large_delta_count': _safe_count(large_valid),
        'large_pred_delta_abs_mean': _safe_mean(pred_delta.abs(), large_valid),
        'large_target_delta_abs_mean': _safe_mean(target_delta.abs(), large_valid),
        'stable_leakage_mean': _safe_mean(gated_delta.abs(), leakage_mask),
        'stable_leakage_p95': _quantile(gated_delta.abs()[leakage_mask], 0.95) if _safe_count(leakage_mask) > 0 else 0.0,
        'gate_interface_mean': _safe_mean(gate, interface_valid),
        'gate_stable_mean': _safe_mean(gate, stable_window_mask),
        'sign': _classification_metrics(
            target_delta=target_delta,
            pred_delta=pred_delta,
            valid_mask=interface_valid,
            tau=tau,
            large_delta_threshold=large_delta_threshold,
        ),
    }


def _flatten(prefix: str, data: dict[str, Any], out: dict[str, Any]) -> None:
    for key, value in data.items():
        next_key = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            _flatten(next_key, value, out)
        elif isinstance(value, list):
            continue
        else:
            out[next_key] = value


def _variant_rows(variants: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in variants.items():
        row: dict[str, Any] = {'variant': name}
        _flatten('', metrics, row)
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items() if key != 'tensor'}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description='Per-finger dead-zone sign diagnostics and oracle residual experiments.')
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--tau-percentile', type=float, default=90.0)
    parser.add_argument('--tau-std-multiplier', type=float, default=2.0)
    parser.add_argument('--tau-minimum', type=float, default=0.0)
    parser.add_argument('--large-delta-threshold', type=float, default=None)
    parser.add_argument('--oracle-gate-source', choices=['hard_phase', 'soft_target'], default='hard_phase')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=args.stage,
        checkpoint=args.checkpoint,
        subset=args.subset,
        num_workers=args.num_workers,
    )
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    stage_config = context['stage_config']
    policy_loss = stage_config.get('policy_loss', {}) if isinstance(stage_config.get('policy_loss', {}), dict) else {}
    large_delta_threshold = (
        float(args.large_delta_threshold)
        if args.large_delta_threshold is not None
        else float(policy_loss.get('finger_large_delta_threshold', policy_loss.get('residual_large_delta_threshold', 8.0)))
    )
    output_dir = (
        resolve_path(project_root, args.output_dir)
        if args.output_dir
        else project_root
        / 'evals'
        / 'per_finger_oracles'
        / str(stage_config.get('name'))
        / f'{checkpoint_path.stem}__{args.subset}'
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model = context['model']
    device = context['device']
    loader = context['loader']
    train_config = context['train_config']
    amp_enabled = bool(train_config.get('amp_enabled', False)) and device.type == 'cuda'

    targets: list[torch.Tensor] = []
    references: list[torch.Tensor] = []
    target_deltas: list[torch.Tensor] = []
    pred_forces: list[torch.Tensor] = []
    pred_deltas: list[torch.Tensor] = []
    tau_source_deltas: list[torch.Tensor] = []
    tau_source_masks: list[torch.Tensor] = []
    gates: list[torch.Tensor] = []
    oracle_gates: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    interface_masks: list[torch.Tensor] = []
    stable_masks: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(batch)

            target = batch['finger_control_force_targets']
            reference = batch['finger_reference_forces']
            target_delta = batch.get('finger_delta_force_targets', target - reference)
            expert_force = batch.get('finger_expert_forces')
            if 'finger_force_pred' not in outputs or 'finger_force_interface_delta' not in outputs:
                raise RuntimeError('This diagnostic requires per-finger policy outputs.')
            pred_force = outputs['finger_force_pred'].reshape_as(target)
            pred_delta = outputs['finger_force_interface_delta'].reshape_as(target_delta)

            window_mask = batch['window_mask'].bool().unsqueeze(-1).expand_as(target)
            interface_mask = (batch['phase_labels'] == 1).unsqueeze(-1).expand_as(target) & window_mask
            stable_mask = batch['stable_masks'].bool().unsqueeze(-1).expand_as(target) & window_mask
            valid_mask = (
                window_mask
                & batch['has_finger_control_target'].bool()
                & batch['has_finger_reference'].bool()
                & torch.isfinite(target)
                & torch.isfinite(reference)
                & torch.isfinite(target_delta)
            )
            if expert_force is not None:
                tau_delta = expert_force - reference
                tau_mask = (
                    window_mask
                    & stable_mask
                    & batch['has_finger_expert'].bool()
                    & batch['has_finger_reference'].bool()
                    & torch.isfinite(tau_delta)
                    & torch.isfinite(reference)
                )
            else:
                tau_delta = target_delta
                tau_mask = valid_mask & stable_mask
            gate = outputs['interface_gate'].reshape(batch['window_mask'].shape).unsqueeze(-1).expand_as(target).to(target.dtype)
            if args.oracle_gate_source == 'soft_target' and 'soft_gate_targets' in batch:
                oracle_gate = batch['soft_gate_targets'].to(target.dtype).unsqueeze(-1).expand_as(target)
            else:
                oracle_gate = (batch['phase_labels'] == 1).to(target.dtype).unsqueeze(-1).expand_as(target)
            oracle_gate = oracle_gate * window_mask.to(target.dtype)

            targets.append(target.detach().cpu().reshape(-1, target.shape[-1]))
            references.append(reference.detach().cpu().reshape(-1, reference.shape[-1]))
            target_deltas.append(target_delta.detach().cpu().reshape(-1, target_delta.shape[-1]))
            pred_forces.append(pred_force.detach().cpu().reshape(-1, pred_force.shape[-1]))
            pred_deltas.append(pred_delta.detach().cpu().reshape(-1, pred_delta.shape[-1]))
            tau_source_deltas.append(tau_delta.detach().cpu().reshape(-1, tau_delta.shape[-1]))
            tau_source_masks.append(tau_mask.detach().cpu().reshape(-1, tau_mask.shape[-1]))
            gates.append(gate.detach().cpu().reshape(-1, gate.shape[-1]))
            oracle_gates.append(oracle_gate.detach().cpu().reshape(-1, oracle_gate.shape[-1]))
            valid_masks.append(valid_mask.detach().cpu().reshape(-1, valid_mask.shape[-1]))
            interface_masks.append(interface_mask.detach().cpu().reshape(-1, interface_mask.shape[-1]))
            stable_masks.append(stable_mask.detach().cpu().reshape(-1, stable_mask.shape[-1]))

    target = torch.cat(targets, dim=0)
    reference = torch.cat(references, dim=0)
    target_delta = torch.cat(target_deltas, dim=0)
    pred_force = torch.cat(pred_forces, dim=0)
    pred_delta = torch.cat(pred_deltas, dim=0)
    tau_source_delta = torch.cat(tau_source_deltas, dim=0)
    tau_source_mask = torch.cat(tau_source_masks, dim=0)
    gate = torch.cat(gates, dim=0)
    oracle_gate = torch.cat(oracle_gates, dim=0)
    valid_mask = torch.cat(valid_masks, dim=0)
    interface_mask = torch.cat(interface_masks, dim=0)
    stable_mask = torch.cat(stable_masks, dim=0)

    tau_info = _estimate_tau(
        tau_source_delta,
        tau_source_mask,
        percentile=float(args.tau_percentile),
        std_multiplier=float(args.tau_std_multiplier),
        minimum=float(args.tau_minimum),
        source='stable_finger_expert_minus_reference',
    )
    tau = tau_info['tensor']
    target_label = _labels_from_delta(target_delta, tau)
    pred_label = _labels_from_delta(pred_delta, tau)
    oracle_sign_delta = target_label.to(pred_delta.dtype) * pred_delta.abs()
    learned_sign_oracle_magnitude_delta = pred_label.to(pred_delta.dtype) * target_delta.abs()
    zero_delta = torch.zeros_like(pred_delta)

    variants = {
        'model': {
            'pred_force': pred_force,
            'pred_delta': pred_delta,
            'gate': gate,
        },
        'reference_only': {
            'pred_force': reference,
            'pred_delta': zero_delta,
            'gate': gate,
        },
        'oracle_sign_learned_magnitude': {
            'pred_force': reference + gate * oracle_sign_delta,
            'pred_delta': oracle_sign_delta,
            'gate': gate,
        },
        'learned_sign_oracle_magnitude': {
            'pred_force': reference + gate * learned_sign_oracle_magnitude_delta,
            'pred_delta': learned_sign_oracle_magnitude_delta,
            'gate': gate,
        },
        'oracle_gate_learned_residual': {
            'pred_force': reference + oracle_gate * pred_delta,
            'pred_delta': pred_delta,
            'gate': oracle_gate,
        },
    }

    variant_metrics: dict[str, dict[str, Any]] = {}
    for name, tensors in variants.items():
        variant_metrics[name] = _force_metrics(
            pred_force=tensors['pred_force'],
            pred_delta=tensors['pred_delta'],
            gate=tensors['gate'],
            target_force=target,
            target_delta=target_delta,
            valid_mask=valid_mask,
            interface_mask=interface_mask,
            stable_window_mask=stable_mask,
            tau=tau,
            large_delta_threshold=large_delta_threshold,
        )

    summary = {
        'stage_name': stage_config.get('name'),
        'stage_path': str(context['stage_path']),
        'checkpoint_path': str(context['checkpoint_path']),
        'subset': args.subset,
        'row_shape': {
            'window_finger_rows': int(target.numel()),
            'finger_count': int(target.shape[-1]),
        },
        'tau': _json_ready(tau_info),
        'large_delta_threshold': large_delta_threshold,
        'oracle_gate_source': args.oracle_gate_source,
        'variants': _json_ready(variant_metrics),
    }

    summary_path = output_dir / 'per_finger_sign_oracle_summary.json'
    metrics_path = output_dir / 'per_finger_sign_oracle_metrics.csv'
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    _write_csv(metrics_path, _variant_rows(variant_metrics))

    print(
        {
            'summary_path': str(summary_path),
            'metrics_path': str(metrics_path),
            'tau': _json_ready(tau_info),
            'model_interface_mae': variant_metrics['model']['interface_mae'],
            'reference_only_interface_mae': variant_metrics['reference_only']['interface_mae'],
            'oracle_sign_interface_mae': variant_metrics['oracle_sign_learned_magnitude']['interface_mae'],
            'learned_sign_oracle_magnitude_interface_mae': variant_metrics['learned_sign_oracle_magnitude']['interface_mae'],
            'oracle_gate_interface_mae': variant_metrics['oracle_gate_learned_residual']['interface_mae'],
        }
    )


if __name__ == '__main__':
    main()
