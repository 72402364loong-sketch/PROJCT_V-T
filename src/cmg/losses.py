from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F



def info_nce_loss(view_a: torch.Tensor, view_b: torch.Tensor, temperature: float) -> torch.Tensor:
    if view_a.shape[0] <= 1:
        return view_a.new_tensor(0.0)
    logits = view_a @ view_b.t() / temperature
    targets = torch.arange(view_a.shape[0], device=view_a.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_a + loss_b)



def supcon_cross_medium_loss(
    features: torch.Tensor,
    object_ids: torch.Tensor,
    sample_ids: torch.Tensor,
    phase_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if features.shape[0] <= 1:
        return features.new_tensor(0.0)
    normalized = F.normalize(features, dim=-1)
    logits = normalized @ normalized.t() / temperature
    identity = torch.eye(features.shape[0], device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(identity, float('-inf'))
    positive_mask = (object_ids[:, None] == object_ids[None, :]) & (sample_ids[:, None] != sample_ids[None, :])
    positive_mask &= (phase_ids[:, None] != phase_ids[None, :])
    positive_mask &= ~identity
    log_denominator = torch.logsumexp(logits, dim=-1)
    losses = []
    for index in range(features.shape[0]):
        positives = positive_mask[index]
        if positives.any():
            log_prob = logits[index, positives] - log_denominator[index]
            losses.append(-log_prob.mean())
    if not losses:
        return features.new_tensor(0.0)
    return torch.stack(losses).mean()



def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    return F.smooth_l1_loss(prediction[mask], target[mask], beta=beta)



def compute_policy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    force_targets: torch.Tensor,
    has_expert: torch.Tensor,
    phase_targets: torch.Tensor,
    stable_mask: torch.Tensor,
    window_mask: torch.Tensor,
    zero: torch.Tensor,
    policy_loss_config: dict[str, Any] | None,
) -> torch.Tensor:
    force_pred = outputs['force_pred'].reshape(-1)
    policy_config = policy_loss_config or {}
    policy_type = str(policy_config.get('type', 'mse'))

    if policy_type == 'decomposed_interface_residual_weighted_smooth_l1':
        all_weight = float(policy_config.get('all_weight', 0.25))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.25))
        base_weight = float(policy_config.get('base_weight', 0.5))
        base_stable_weight = float(policy_config.get('base_stable_weight', 1.0))
        residual_interface_weight = float(policy_config.get('residual_interface_weight', 6.0))
        residual_stable_zero_weight = float(policy_config.get('residual_stable_zero_weight', 1.0))
        residual_non_interface_weight = float(policy_config.get('residual_non_interface_weight', 0.25))
        gated_residual_interface_weight = float(policy_config.get('gated_residual_interface_weight', 0.0))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        beta = float(policy_config.get('beta', 1.0))
        detach_base_target = bool(policy_config.get('detach_base_target', True))

        if 'force_base' not in outputs or 'force_interface_delta' not in outputs:
            raise RuntimeError('Decomposed residual policy loss requires force_base and force_interface_delta outputs.')

        base_force = outputs['force_base'].reshape(-1)
        interface_delta = outputs['force_interface_delta'].reshape(-1)
        interface_gate = outputs['interface_gate'].reshape(-1) if 'interface_gate' in outputs else None
        interface_expert = has_expert & (phase_targets == 1)
        stable_expert = has_expert & stable_mask
        non_interface_expert = has_expert & (phase_targets != 1)

        total = zero
        if all_weight > 0.0:
            total = total + all_weight * masked_smooth_l1(force_pred, force_targets, has_expert, beta=beta, zero=zero)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)
        if base_weight > 0.0:
            total = total + base_weight * masked_smooth_l1(base_force, force_targets, has_expert, beta=beta, zero=zero)
        if base_stable_weight > 0.0:
            total = total + base_stable_weight * masked_smooth_l1(base_force, force_targets, stable_expert, beta=beta, zero=zero)

        base_reference = base_force.detach() if detach_base_target else base_force
        delta_target = force_targets - base_reference
        zero_target = torch.zeros_like(interface_delta)
        if residual_interface_weight > 0.0:
            total = total + residual_interface_weight * masked_smooth_l1(interface_delta, delta_target, interface_expert, beta=beta, zero=zero)
        if residual_stable_zero_weight > 0.0:
            total = total + residual_stable_zero_weight * masked_smooth_l1(interface_delta, zero_target, stable_expert, beta=beta, zero=zero)
        if residual_non_interface_weight > 0.0:
            total = total + residual_non_interface_weight * masked_smooth_l1(interface_delta, zero_target, non_interface_expert, beta=beta, zero=zero)
        if gated_residual_interface_weight > 0.0 and interface_gate is not None:
            total = total + gated_residual_interface_weight * masked_smooth_l1(
                interface_gate * interface_delta,
                delta_target,
                interface_expert,
                beta=beta,
                zero=zero,
            )
        if quiet_weight > 0.0:
            non_interface_windows = window_mask & (phase_targets != 1)
            if non_interface_windows.any():
                total = total + quiet_weight * torch.mean(interface_delta[non_interface_windows] ** 2)
        return total

    if policy_type in {'interface_weighted_smooth_l1', 'interface_residual_weighted_smooth_l1'}:
        all_weight = float(policy_config.get('all_weight', 1.0))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.5))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        beta = float(policy_config.get('beta', 1.0))

        total = zero
        if all_weight > 0.0:
            total = total + masked_smooth_l1(force_pred, force_targets, has_expert, beta=beta, zero=zero)

        interface_expert = has_expert & (phase_targets == 1)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)

        stable_expert = has_expert & stable_mask
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)

        if quiet_weight > 0.0 and 'force_interface_delta' in outputs and 'interface_gate' in outputs:
            interface_delta = outputs['force_interface_delta'].reshape(-1)[window_mask]
            interface_gate = outputs['interface_gate'].reshape(-1)[window_mask]
            total = total + quiet_weight * torch.mean(((1.0 - interface_gate) * interface_delta) ** 2)
        return total

    return F.mse_loss(force_pred[has_expert], force_targets[has_expert])



def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    loss_weights: dict[str, float],
    policy_loss_config: dict[str, Any] | None,
    temperature_clip: float,
    temperature_inv: float,
    phase_class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    window_mask = batch['window_mask'].reshape(-1)
    stable_mask = batch['stable_masks'].reshape(-1) & window_mask
    phase_targets = batch['phase_labels'].reshape(-1)
    medium_logits = outputs['medium_logits'].reshape(-1, outputs['medium_logits'].shape[-1])
    fragility_logits = outputs['fragility_logits']
    geometry_logits = outputs['geometry_logits']
    surface_logits = outputs['surface_logits']

    repeated_fragility = batch['fragility_label'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_geometry = batch['geometry_label'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_surface = batch['surface_label'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_object = batch['object_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_sample = batch['sample_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    stable_phases = batch['stable_phases'].reshape(-1)

    losses: dict[str, torch.Tensor] = {}
    zero = medium_logits.new_tensor(0.0)
    med_weight = float(loss_weights.get('med', 0.0))
    attr_weight = float(loss_weights.get('attr', 0.0))
    clip_weight = float(loss_weights.get('clip', 0.0))
    inv_weight = float(loss_weights.get('inv', 0.0))
    pol_weight = float(loss_weights.get('pol', 0.0))

    losses['med'] = F.cross_entropy(medium_logits[window_mask], phase_targets[window_mask], weight=phase_class_weights) if med_weight > 0.0 else zero
    if attr_weight > 0.0:
        losses['frag'] = F.cross_entropy(fragility_logits[window_mask], repeated_fragility[window_mask])
        losses['geom'] = F.cross_entropy(geometry_logits[window_mask], repeated_geometry[window_mask])
        losses['surf'] = F.cross_entropy(surface_logits[window_mask], repeated_surface[window_mask])
        losses['attr'] = losses['frag'] + losses['geom'] + losses['surf']
    else:
        losses['frag'] = zero
        losses['geom'] = zero
        losses['surf'] = zero
        losses['attr'] = zero

    if clip_weight > 0.0:
        stable_valid = stable_mask.nonzero(as_tuple=False).flatten()
        losses['clip'] = info_nce_loss(outputs['z_v'][stable_valid], outputs['z_t'][stable_valid], temperature=temperature_clip)
    else:
        losses['clip'] = zero

    if inv_weight > 0.0:
        stable_water_air = stable_mask & (stable_phases >= 0)
        stable_indices = stable_water_air.nonzero(as_tuple=False).flatten()
        losses['inv'] = supcon_cross_medium_loss(
            outputs['z_content'][stable_indices],
            repeated_object[stable_indices],
            repeated_sample[stable_indices],
            stable_phases[stable_indices],
            temperature=temperature_inv,
        )
    else:
        losses['inv'] = zero

    force_targets = batch['expert_forces'].reshape(-1)
    has_expert = batch['has_expert'].reshape(-1) & window_mask
    if pol_weight > 0.0 and has_expert.any():
        losses['pol'] = compute_policy_loss(
            outputs,
            force_targets=force_targets,
            has_expert=has_expert,
            phase_targets=phase_targets,
            stable_mask=stable_mask,
            window_mask=window_mask,
            zero=zero,
            policy_loss_config=policy_loss_config,
        )
    else:
        losses['pol'] = zero

    total = (
        loss_weights.get('clip', 0.0) * losses['clip']
        + loss_weights.get('inv', 0.0) * losses['inv']
        + loss_weights.get('med', 0.0) * losses['med']
        + loss_weights.get('attr', 0.0) * losses['attr']
        + loss_weights.get('pol', 0.0) * losses['pol']
    )
    losses['total'] = total
    return losses
