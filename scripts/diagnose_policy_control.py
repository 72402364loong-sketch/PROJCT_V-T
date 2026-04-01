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

from cmg.constants import INDEX_TO_PHASE, PHASE_TO_INDEX
from cmg.evaluation import prepare_evaluation_context, resolve_path
from cmg.training import move_to_device



def _safe_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    if not mask.any():
        return {'count': 0, 'mean': 0.0, 'std': 0.0, 'abs_mean': 0.0, 'min': 0.0, 'max': 0.0}
    selected = values[mask]
    return {
        'count': int(selected.numel()),
        'mean': float(selected.mean().item()),
        'std': float(selected.std(unbiased=False).item()) if selected.numel() > 1 else 0.0,
        'abs_mean': float(selected.abs().mean().item()),
        'min': float(selected.min().item()),
        'max': float(selected.max().item()),
    }



def _compute_force_metrics(
    force_pred: torch.Tensor,
    target: torch.Tensor,
    expert_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    phase_labels: torch.Tensor,
) -> dict[str, float]:
    zero_metrics = {
        'overall_mae': 0.0,
        'overall_mse': 0.0,
        'stable_mae': 0.0,
        'stable_mse': 0.0,
        'interface_mae': 0.0,
        'interface_mse': 0.0,
        'interface_bias': 0.0,
        'interface_hit_rate_100': 0.0,
        'interface_hit_rate_200': 0.0,
        'interface_hit_rate_300': 0.0,
    }
    if not expert_mask.any():
        return zero_metrics

    pred = force_pred[expert_mask]
    truth = target[expert_mask]
    error = pred - truth
    abs_error = error.abs()
    sq_error = error ** 2
    metrics = {
        'overall_mae': float(abs_error.mean().item()),
        'overall_mse': float(sq_error.mean().item()),
        'stable_mae': 0.0,
        'stable_mse': 0.0,
        'interface_mae': 0.0,
        'interface_mse': 0.0,
        'interface_bias': 0.0,
        'interface_hit_rate_100': 0.0,
        'interface_hit_rate_200': 0.0,
        'interface_hit_rate_300': 0.0,
    }

    stable_expert = expert_mask & stable_mask
    if stable_expert.any():
        stable_error = force_pred[stable_expert] - target[stable_expert]
        metrics['stable_mae'] = float(stable_error.abs().mean().item())
        metrics['stable_mse'] = float((stable_error ** 2).mean().item())

    interface_expert = expert_mask & (phase_labels == int(PHASE_TO_INDEX['Interface']))
    if interface_expert.any():
        interface_error = force_pred[interface_expert] - target[interface_expert]
        abs_interface_error = interface_error.abs()
        metrics['interface_mae'] = float(abs_interface_error.mean().item())
        metrics['interface_mse'] = float((interface_error ** 2).mean().item())
        metrics['interface_bias'] = float(interface_error.mean().item())
        metrics['interface_hit_rate_100'] = float((abs_interface_error <= 100.0).float().mean().item())
        metrics['interface_hit_rate_200'] = float((abs_interface_error <= 200.0).float().mean().item())
        metrics['interface_hit_rate_300'] = float((abs_interface_error <= 300.0).float().mean().item())
    return metrics



def _summarize_activity(
    *,
    force_pred: torch.Tensor,
    force_pred_oracle: torch.Tensor,
    force_base: torch.Tensor,
    force_delta: torch.Tensor,
    interface_gate: torch.Tensor,
    oracle_gate: torch.Tensor,
    expert_force: torch.Tensor,
    reference_force: torch.Tensor,
    delta_target: torch.Tensor,
    window_mask: torch.Tensor,
    expert_mask: torch.Tensor,
    reference_mask: torch.Tensor,
    stable_mask: torch.Tensor,
    phase_labels: torch.Tensor,
) -> dict[str, Any]:
    gated_delta = interface_gate * force_delta
    oracle_gated_delta = oracle_gate * force_delta
    slices = {
        'all_valid': window_mask,
        'expert': expert_mask,
        'stable_expert': expert_mask & stable_mask,
        'interface_expert': expert_mask & (phase_labels == int(PHASE_TO_INDEX['Interface'])),
        'water_expert': expert_mask & (phase_labels == int(PHASE_TO_INDEX['Water'])),
        'air_expert': expert_mask & (phase_labels == int(PHASE_TO_INDEX['Air'])),
        'reference_windows': reference_mask,
    }
    summary: dict[str, Any] = {}
    for name, mask in slices.items():
        block: dict[str, Any] = {
            'count': int(mask.sum().item()),
            'force_base': _safe_stats(force_base, mask),
            'force_interface_delta': _safe_stats(force_delta, mask),
            'predicted_gate': _safe_stats(interface_gate, mask),
            'gated_residual': _safe_stats(gated_delta, mask),
            'oracle_gated_residual': _safe_stats(oracle_gated_delta, mask),
            'force_pred': _safe_stats(force_pred, mask),
            'force_pred_oracle': _safe_stats(force_pred_oracle, mask),
        }
        if (mask & expert_mask).any():
            block['expert_force'] = _safe_stats(expert_force, mask & expert_mask)
            block['prediction_error'] = _safe_stats(force_pred - expert_force, mask & expert_mask)
            block['oracle_prediction_error'] = _safe_stats(force_pred_oracle - expert_force, mask & expert_mask)
        if (mask & reference_mask).any():
            block['reference_force'] = _safe_stats(reference_force, mask & reference_mask)
            block['delta_force_target'] = _safe_stats(delta_target, mask & reference_mask)
            block['base_minus_reference'] = _safe_stats(force_base - reference_force, mask & reference_mask)
            block['delta_minus_target'] = _safe_stats(force_delta - delta_target, mask & reference_mask)
        summary[name] = block
    return summary



def _bias_decomposition(
    *,
    force_pred: torch.Tensor,
    force_pred_oracle: torch.Tensor,
    force_base: torch.Tensor,
    force_delta: torch.Tensor,
    interface_gate: torch.Tensor,
    oracle_gate: torch.Tensor,
    expert_force: torch.Tensor,
    expert_mask: torch.Tensor,
    phase_labels: torch.Tensor,
) -> dict[str, float | int]:
    interface_mask = expert_mask & (phase_labels == int(PHASE_TO_INDEX['Interface']))
    if not interface_mask.any():
        return {
            'count': 0,
            'target_mean': 0.0,
            'pred_mean': 0.0,
            'oracle_pred_mean': 0.0,
            'base_mean': 0.0,
            'residual_mean': 0.0,
            'predicted_gated_residual_mean': 0.0,
            'oracle_gated_residual_mean': 0.0,
            'predicted_gate_mean': 0.0,
            'oracle_gate_mean': 0.0,
            'pred_bias_mean': 0.0,
            'oracle_pred_bias_mean': 0.0,
            'base_bias_mean': 0.0,
        }
    target = expert_force[interface_mask]
    pred = force_pred[interface_mask]
    pred_oracle = force_pred_oracle[interface_mask]
    base = force_base[interface_mask]
    delta = force_delta[interface_mask]
    gate = interface_gate[interface_mask]
    oracle = oracle_gate[interface_mask]
    return {
        'count': int(target.numel()),
        'target_mean': float(target.mean().item()),
        'pred_mean': float(pred.mean().item()),
        'oracle_pred_mean': float(pred_oracle.mean().item()),
        'base_mean': float(base.mean().item()),
        'residual_mean': float(delta.mean().item()),
        'predicted_gated_residual_mean': float((gate * delta).mean().item()),
        'oracle_gated_residual_mean': float((oracle * delta).mean().item()),
        'predicted_gate_mean': float(gate.mean().item()),
        'oracle_gate_mean': float(oracle.mean().item()),
        'pred_bias_mean': float((pred - target).mean().item()),
        'oracle_pred_bias_mean': float((pred_oracle - target).mean().item()),
        'base_bias_mean': float((base - target).mean().item()),
    }



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--subset', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--only-stable', action='store_true')
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    context = prepare_evaluation_context(
        project_root=project_root,
        stage=args.stage,
        checkpoint=args.checkpoint,
        subset=args.subset,
        only_stable=bool(args.only_stable),
    )

    stage_name = str(context['stage_config'].get('name'))
    checkpoint_path = resolve_path(project_root, args.checkpoint)
    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        suffix = '__stable' if args.only_stable else ''
        output_dir = project_root / 'evals' / 'diagnostics' / stage_name / f'{checkpoint_path.stem}__{args.subset}{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = context['device']
    model = context['model']
    loader = context['loader']
    amp_enabled = bool(context['train_config'].get('amp_enabled', False)) and device.type == 'cuda'

    phase_rows: list[dict[str, Any]] = []
    force_pred_all: list[torch.Tensor] = []
    force_pred_oracle_all: list[torch.Tensor] = []
    force_base_all: list[torch.Tensor] = []
    force_delta_all: list[torch.Tensor] = []
    interface_gate_all: list[torch.Tensor] = []
    oracle_gate_all: list[torch.Tensor] = []
    expert_force_all: list[torch.Tensor] = []
    reference_force_all: list[torch.Tensor] = []
    delta_target_all: list[torch.Tensor] = []
    window_mask_all: list[torch.Tensor] = []
    expert_mask_all: list[torch.Tensor] = []
    reference_mask_all: list[torch.Tensor] = []
    stable_mask_all: list[torch.Tensor] = []
    phase_labels_all: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.amp.autocast('cuda', enabled=amp_enabled):
                outputs = model(batch)

            window_mask = batch['window_mask']
            phase_labels = batch['phase_labels']
            stable_mask = batch['stable_masks'] & window_mask
            expert_mask = batch['has_expert'] & window_mask
            reference_mask = batch['has_reference'] & expert_mask if 'has_reference' in batch else torch.zeros_like(expert_mask)
            expert_force = batch['expert_forces']
            reference_force = batch['reference_forces'] if 'reference_forces' in batch else torch.full_like(expert_force, float('nan'))
            delta_target = batch['delta_force_targets'] if 'delta_force_targets' in batch else torch.full_like(expert_force, float('nan'))
            force_pred = outputs['force_pred'].reshape_as(expert_force)
            force_base = outputs.get('force_base', outputs['force_pred']).reshape_as(expert_force)
            force_delta = outputs.get('force_interface_delta', torch.zeros_like(outputs['force_pred'])).reshape_as(expert_force)
            interface_gate = outputs.get('interface_gate', torch.zeros_like(outputs['force_pred'])).reshape_as(expert_force)
            oracle_gate = (phase_labels == int(PHASE_TO_INDEX['Interface'])).to(force_pred.dtype) * window_mask.to(force_pred.dtype)
            force_pred_oracle = force_base + oracle_gate * force_delta

            force_pred_all.append(force_pred.detach().cpu())
            force_pred_oracle_all.append(force_pred_oracle.detach().cpu())
            force_base_all.append(force_base.detach().cpu())
            force_delta_all.append(force_delta.detach().cpu())
            interface_gate_all.append(interface_gate.detach().cpu())
            oracle_gate_all.append(oracle_gate.detach().cpu())
            expert_force_all.append(expert_force.detach().cpu())
            reference_force_all.append(reference_force.detach().cpu())
            delta_target_all.append(delta_target.detach().cpu())
            window_mask_all.append(window_mask.detach().cpu())
            expert_mask_all.append(expert_mask.detach().cpu())
            reference_mask_all.append(reference_mask.detach().cpu())
            stable_mask_all.append(stable_mask.detach().cpu())
            phase_labels_all.append(phase_labels.detach().cpu())

            batch_size, max_windows = window_mask.shape
            force_pred_cpu = force_pred.detach().cpu()
            force_pred_oracle_cpu = force_pred_oracle.detach().cpu()
            force_base_cpu = force_base.detach().cpu()
            force_delta_cpu = force_delta.detach().cpu()
            interface_gate_cpu = interface_gate.detach().cpu()
            oracle_gate_cpu = oracle_gate.detach().cpu()
            expert_force_cpu = expert_force.detach().cpu()
            reference_force_cpu = reference_force.detach().cpu()
            delta_target_cpu = delta_target.detach().cpu()
            phase_labels_cpu = phase_labels.detach().cpu()
            stable_mask_cpu = stable_mask.detach().cpu()
            expert_mask_cpu = expert_mask.detach().cpu()
            reference_mask_cpu = reference_mask.detach().cpu()
            window_mask_cpu = window_mask.detach().cpu()
            medium_pred_cpu = outputs['medium_logits'].argmax(dim=-1).detach().cpu()

            for batch_index in range(batch_size):
                sample_id = batch['sample_ids'][batch_index]
                object_id = batch['object_ids'][batch_index]
                for window_index in range(max_windows):
                    if not bool(window_mask_cpu[batch_index, window_index]):
                        continue
                    phase_index = int(phase_labels_cpu[batch_index, window_index].item())
                    row = {
                        'sample_id': sample_id,
                        'object_id': object_id,
                        'window_index': window_index,
                        'phase_index': phase_index,
                        'phase_label': INDEX_TO_PHASE.get(phase_index, str(phase_index)),
                        'medium_pred_index': int(medium_pred_cpu[batch_index, window_index].item()),
                        'medium_pred_label': INDEX_TO_PHASE.get(int(medium_pred_cpu[batch_index, window_index].item()), str(int(medium_pred_cpu[batch_index, window_index].item()))),
                        'stable_mask': int(bool(stable_mask_cpu[batch_index, window_index])),
                        'has_expert': int(bool(expert_mask_cpu[batch_index, window_index])),
                        'has_reference': int(bool(reference_mask_cpu[batch_index, window_index])),
                        'expert_force': float(expert_force_cpu[batch_index, window_index].item()),
                        'reference_force': float(reference_force_cpu[batch_index, window_index].item()),
                        'delta_force_target': float(delta_target_cpu[batch_index, window_index].item()),
                        'force_pred': float(force_pred_cpu[batch_index, window_index].item()),
                        'force_pred_oracle': float(force_pred_oracle_cpu[batch_index, window_index].item()),
                        'force_base': float(force_base_cpu[batch_index, window_index].item()),
                        'force_interface_delta': float(force_delta_cpu[batch_index, window_index].item()),
                        'interface_gate': float(interface_gate_cpu[batch_index, window_index].item()),
                        'oracle_interface_gate': float(oracle_gate_cpu[batch_index, window_index].item()),
                        'abs_error': float(abs(force_pred_cpu[batch_index, window_index].item() - expert_force_cpu[batch_index, window_index].item())) if bool(expert_mask_cpu[batch_index, window_index]) else float('nan'),
                        'abs_error_oracle': float(abs(force_pred_oracle_cpu[batch_index, window_index].item() - expert_force_cpu[batch_index, window_index].item())) if bool(expert_mask_cpu[batch_index, window_index]) else float('nan'),
                    }
                    phase_rows.append(row)

    force_pred = torch.cat([tensor.reshape(-1) for tensor in force_pred_all], dim=0)
    force_pred_oracle = torch.cat([tensor.reshape(-1) for tensor in force_pred_oracle_all], dim=0)
    force_base = torch.cat([tensor.reshape(-1) for tensor in force_base_all], dim=0)
    force_delta = torch.cat([tensor.reshape(-1) for tensor in force_delta_all], dim=0)
    interface_gate = torch.cat([tensor.reshape(-1) for tensor in interface_gate_all], dim=0)
    oracle_gate = torch.cat([tensor.reshape(-1) for tensor in oracle_gate_all], dim=0)
    expert_force = torch.cat([tensor.reshape(-1) for tensor in expert_force_all], dim=0)
    reference_force = torch.cat([tensor.reshape(-1) for tensor in reference_force_all], dim=0)
    delta_target = torch.cat([tensor.reshape(-1) for tensor in delta_target_all], dim=0)
    window_mask = torch.cat([tensor.reshape(-1) for tensor in window_mask_all], dim=0)
    expert_mask = torch.cat([tensor.reshape(-1) for tensor in expert_mask_all], dim=0)
    reference_mask = torch.cat([tensor.reshape(-1) for tensor in reference_mask_all], dim=0)
    stable_mask = torch.cat([tensor.reshape(-1) for tensor in stable_mask_all], dim=0)
    phase_labels = torch.cat([tensor.reshape(-1) for tensor in phase_labels_all], dim=0)

    standard_metrics = _compute_force_metrics(force_pred, expert_force, expert_mask, stable_mask, phase_labels)
    oracle_metrics = _compute_force_metrics(force_pred_oracle, expert_force, expert_mask, stable_mask, phase_labels)
    activity_summary = _summarize_activity(
        force_pred=force_pred,
        force_pred_oracle=force_pred_oracle,
        force_base=force_base,
        force_delta=force_delta,
        interface_gate=interface_gate,
        oracle_gate=oracle_gate,
        expert_force=expert_force,
        reference_force=reference_force,
        delta_target=delta_target,
        window_mask=window_mask,
        expert_mask=expert_mask,
        reference_mask=reference_mask,
        stable_mask=stable_mask,
        phase_labels=phase_labels,
    )
    bias_summary = _bias_decomposition(
        force_pred=force_pred,
        force_pred_oracle=force_pred_oracle,
        force_base=force_base,
        force_delta=force_delta,
        interface_gate=interface_gate,
        oracle_gate=oracle_gate,
        expert_force=expert_force,
        expert_mask=expert_mask,
        phase_labels=phase_labels,
    )

    summary = {
        'stage_name': stage_name,
        'subset': args.subset,
        'only_stable': bool(args.only_stable),
        'checkpoint_path': str(context['checkpoint_path'].resolve()),
        'standard_metrics': standard_metrics,
        'oracle_gate_metrics': oracle_metrics,
        'oracle_gate_delta': {key: float(oracle_metrics[key] - standard_metrics.get(key, 0.0)) for key in oracle_metrics},
        'residual_activity': activity_summary,
        'bias_decomposition': bias_summary,
        'row_count': len(phase_rows),
    }

    rows_path = output_dir / 'policy_windows.csv'
    summary_path = output_dir / 'policy_diagnostics.json'
    with rows_path.open('w', encoding='utf-8', newline='') as handle:
        if phase_rows:
            writer = csv.DictWriter(handle, fieldnames=list(phase_rows[0].keys()))
            writer.writeheader()
            writer.writerows(phase_rows)
        else:
            handle.write('')
    with summary_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print({
        'output_dir': str(output_dir),
        'summary_path': str(summary_path),
        'rows_path': str(rows_path),
        'standard_metrics': standard_metrics,
        'oracle_gate_metrics': oracle_metrics,
    })


if __name__ == '__main__':
    main()
