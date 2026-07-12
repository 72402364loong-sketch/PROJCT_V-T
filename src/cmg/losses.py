from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from cmg.attribute_metrics import (
    ATTRIBUTE_LABEL_KEYS,
    ATTRIBUTE_LOGIT_KEYS,
    ATTRIBUTE_SAMPLE_LOGIT_KEYS,
    ATTRIBUTE_TASKS,
    build_attribute_window_masks,
)


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


def masked_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    if scale <= 0.0:
        raise ValueError(f'delta normalization scale must be positive, got {scale}.')
    if scale == 1.0:
        return F.smooth_l1_loss(prediction[mask], target[mask], beta=beta)
    return F.smooth_l1_loss(prediction[mask] / scale, target[mask] / scale, beta=beta)


def masked_weighted_scaled_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    weight_alpha: float,
    weight_cap: float,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    if scale <= 0.0:
        raise ValueError(f'delta normalization scale must be positive, got {scale}.')
    pred = prediction[mask] / scale
    tgt = target[mask] / scale
    loss = F.smooth_l1_loss(pred, tgt, beta=beta, reduction='none')
    weights = 1.0 + float(weight_alpha) * torch.clamp(torch.abs(tgt), min=0.0, max=float(weight_cap))
    if sample_weight is not None:
        flat_weight = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
        flat_mask = mask.reshape(-1)
        if flat_weight.numel() != flat_mask.numel():
            raise RuntimeError(
                'sample_weight cannot be broadcast to policy loss mask: '
                f'{flat_weight.numel()} vs {flat_mask.numel()}.'
            )
        selected_weight = flat_weight[flat_mask]
        if not (selected_weight > 0).any():
            return zero
        return (loss * weights * selected_weight).sum() / selected_weight.sum().clamp_min(loss.new_tensor(1e-12))
    return (loss * weights).mean()


def masked_large_delta_sign_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float,
    margin: float,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    if not mask.any():
        return zero
    if scale <= 0.0:
        raise ValueError(f'delta normalization scale must be positive, got {scale}.')
    pred = prediction[mask] / scale
    tgt = target[mask]
    signs = torch.sign(tgt)
    valid = signs != 0
    if not valid.any():
        return zero
    signed_pred = signs[valid] * pred[valid]
    loss = torch.relu(float(margin) - signed_pred)
    if sample_weight is not None:
        flat_weight = sample_weight.reshape(-1).to(device=loss.device, dtype=loss.dtype)
        flat_mask = mask.reshape(-1)
        if flat_weight.numel() != flat_mask.numel():
            raise RuntimeError(
                'sample_weight cannot be broadcast to policy loss mask: '
                f'{flat_weight.numel()} vs {flat_mask.numel()}.'
            )
        selected_weight = flat_weight[flat_mask][valid]
        if not (selected_weight > 0).any():
            return zero
        return (loss * selected_weight).sum() / selected_weight.sum().clamp_min(loss.new_tensor(1e-12))
    return loss.mean()


def sign_balanced_large_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    weight_alpha: float,
    weight_cap: float,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    positive_mask = mask & (target > 0)
    negative_mask = mask & (target < 0)
    parts = []
    weights = []
    positive_weight = float(positive_weight)
    negative_weight = float(negative_weight)
    if positive_weight > 0.0 and positive_mask.any():
        parts.append(
            positive_weight
            * masked_weighted_scaled_smooth_l1(
                prediction,
                target,
                positive_mask,
                beta=beta,
                scale=scale,
                weight_alpha=weight_alpha,
                weight_cap=weight_cap,
                sample_weight=sample_weight,
                zero=zero,
            )
        )
        weights.append(positive_weight)
    if negative_weight > 0.0 and negative_mask.any():
        parts.append(
            negative_weight
            * masked_weighted_scaled_smooth_l1(
                prediction,
                target,
                negative_mask,
                beta=beta,
                scale=scale,
                weight_alpha=weight_alpha,
                weight_cap=weight_cap,
                sample_weight=sample_weight,
                zero=zero,
            )
        )
        weights.append(negative_weight)
    if not parts:
        return zero
    return torch.stack(parts).sum() / prediction.new_tensor(sum(weights))


def sign_balanced_large_delta_sign_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    scale: float,
    margin: float,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    positive_mask = mask & (target > 0)
    negative_mask = mask & (target < 0)
    parts = []
    weights = []
    positive_weight = float(positive_weight)
    negative_weight = float(negative_weight)
    if positive_weight > 0.0 and positive_mask.any():
        parts.append(
            positive_weight
            * masked_large_delta_sign_loss(
                prediction,
                target,
                positive_mask,
                scale=scale,
                margin=margin,
                sample_weight=sample_weight,
                zero=zero,
            )
        )
        weights.append(positive_weight)
    if negative_weight > 0.0 and negative_mask.any():
        parts.append(
            negative_weight
            * masked_large_delta_sign_loss(
                prediction,
                target,
                negative_mask,
                scale=scale,
                margin=margin,
                sample_weight=sample_weight,
                zero=zero,
            )
        )
        weights.append(negative_weight)
    if not parts:
        return zero
    return torch.stack(parts).sum() / prediction.new_tensor(sum(weights))


def sign_balanced_direction_gate_cross_entropy(
    direction_logits: torch.Tensor,
    target_delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    valid_mask = mask & (target_delta != 0)
    if not valid_mask.any():
        return zero
    logits = direction_logits.reshape(-1, 2)
    targets = (target_delta.reshape(-1) > 0).long()
    valid = valid_mask.reshape(-1)
    positive_mask = valid & (targets == 1)
    negative_mask = valid & (targets == 0)
    parts = []
    weights = []
    positive_weight = float(positive_weight)
    negative_weight = float(negative_weight)
    if positive_weight > 0.0 and positive_mask.any():
        positive_loss = F.cross_entropy(logits[positive_mask], targets[positive_mask], reduction='none')
        if sample_weight is not None:
            flat_weight = sample_weight.reshape(-1).to(device=positive_loss.device, dtype=positive_loss.dtype)
            if flat_weight.numel() != valid.numel():
                raise RuntimeError(
                    'sample_weight cannot be broadcast to policy loss mask: '
                    f'{flat_weight.numel()} vs {valid.numel()}.'
                )
            selected_weight = flat_weight[positive_mask]
            positive_loss = (
                (positive_loss * selected_weight).sum()
                / selected_weight.sum().clamp_min(positive_loss.new_tensor(1e-12))
            )
        else:
            positive_loss = positive_loss.mean()
        parts.append(positive_weight * positive_loss)
        weights.append(positive_weight)
    if negative_weight > 0.0 and negative_mask.any():
        negative_loss = F.cross_entropy(logits[negative_mask], targets[negative_mask], reduction='none')
        if sample_weight is not None:
            flat_weight = sample_weight.reshape(-1).to(device=negative_loss.device, dtype=negative_loss.dtype)
            if flat_weight.numel() != valid.numel():
                raise RuntimeError(
                    'sample_weight cannot be broadcast to policy loss mask: '
                    f'{flat_weight.numel()} vs {valid.numel()}.'
                )
            selected_weight = flat_weight[negative_mask]
            negative_loss = (
                (negative_loss * selected_weight).sum()
                / selected_weight.sum().clamp_min(negative_loss.new_tensor(1e-12))
            )
        else:
            negative_loss = negative_loss.mean()
        parts.append(negative_weight * negative_loss)
        weights.append(negative_weight)
    if not parts:
        return zero
    return torch.stack(parts).sum() / direction_logits.new_tensor(sum(weights))


def sign_specific_magnitude_loss(
    outputs: dict[str, torch.Tensor],
    target_delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    positive_scale: float | None = None,
    negative_scale: float | None = None,
    opposite_weight: float = 0.0,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    zero: torch.Tensor,
) -> torch.Tensor:
    pos_magnitude = outputs.get('force_interface_delta_pos_magnitude')
    neg_magnitude = outputs.get('force_interface_delta_neg_magnitude')
    if pos_magnitude is None or neg_magnitude is None:
        raise RuntimeError(
            'Sign-specific magnitude loss requires force_interface_delta_pos_magnitude and '
            'force_interface_delta_neg_magnitude outputs. Use policy.head_type=state_residual_sign_specific.'
        )
    if scale <= 0.0:
        raise ValueError(f'direction magnitude scale must be positive, got {scale}.')
    positive_scale = float(scale if positive_scale is None else positive_scale)
    negative_scale = float(scale if negative_scale is None else negative_scale)
    if positive_scale <= 0.0:
        raise ValueError(f'positive direction magnitude scale must be positive, got {positive_scale}.')
    if negative_scale <= 0.0:
        raise ValueError(f'negative direction magnitude scale must be positive, got {negative_scale}.')

    pos_magnitude = pos_magnitude.reshape(-1)
    neg_magnitude = neg_magnitude.reshape(-1)
    flat_delta = target_delta.reshape(-1)
    flat_mask = mask.reshape(-1)
    zero_magnitude = torch.zeros_like(flat_delta)

    positive_mask = flat_mask & (flat_delta > 0)
    negative_mask = flat_mask & (flat_delta < 0)
    flat_sample_weight = None
    if sample_weight is not None:
        flat_sample_weight = sample_weight.reshape(-1).to(device=flat_delta.device, dtype=flat_delta.dtype)
        if flat_sample_weight.numel() != flat_mask.numel():
            raise RuntimeError(
                'sample_weight cannot be broadcast to policy loss mask: '
                f'{flat_sample_weight.numel()} vs {flat_mask.numel()}.'
            )
    parts = []
    weights = []
    positive_weight = float(positive_weight)
    negative_weight = float(negative_weight)
    opposite_weight = float(opposite_weight)

    if positive_weight > 0.0 and positive_mask.any():
        positive_target_magnitude = torch.abs(flat_delta[positive_mask]) / positive_scale
        active_loss = F.smooth_l1_loss(
            pos_magnitude[positive_mask],
            positive_target_magnitude,
            beta=beta,
            reduction='none',
        )
        if flat_sample_weight is not None:
            selected_weight = flat_sample_weight[positive_mask]
            active_loss = (
                (active_loss * selected_weight).sum()
                / selected_weight.sum().clamp_min(active_loss.new_tensor(1e-12))
            )
        else:
            active_loss = active_loss.mean()
        if opposite_weight > 0.0:
            opposite_loss = F.smooth_l1_loss(
                neg_magnitude[positive_mask],
                zero_magnitude[positive_mask],
                beta=beta,
                reduction='none',
            )
            if flat_sample_weight is not None:
                selected_weight = flat_sample_weight[positive_mask]
                opposite_loss = (
                    (opposite_loss * selected_weight).sum()
                    / selected_weight.sum().clamp_min(opposite_loss.new_tensor(1e-12))
                )
            else:
                opposite_loss = opposite_loss.mean()
        else:
            opposite_loss = zero
        parts.append(positive_weight * (active_loss + opposite_weight * opposite_loss))
        weights.append(positive_weight)

    if negative_weight > 0.0 and negative_mask.any():
        negative_target_magnitude = torch.abs(flat_delta[negative_mask]) / negative_scale
        active_loss = F.smooth_l1_loss(
            neg_magnitude[negative_mask],
            negative_target_magnitude,
            beta=beta,
            reduction='none',
        )
        if flat_sample_weight is not None:
            selected_weight = flat_sample_weight[negative_mask]
            active_loss = (
                (active_loss * selected_weight).sum()
                / selected_weight.sum().clamp_min(active_loss.new_tensor(1e-12))
            )
        else:
            active_loss = active_loss.mean()
        if opposite_weight > 0.0:
            opposite_loss = F.smooth_l1_loss(
                pos_magnitude[negative_mask],
                zero_magnitude[negative_mask],
                beta=beta,
                reduction='none',
            )
            if flat_sample_weight is not None:
                selected_weight = flat_sample_weight[negative_mask]
                opposite_loss = (
                    (opposite_loss * selected_weight).sum()
                    / selected_weight.sum().clamp_min(opposite_loss.new_tensor(1e-12))
                )
            else:
                opposite_loss = opposite_loss.mean()
        else:
            opposite_loss = zero
        parts.append(negative_weight * (active_loss + opposite_weight * opposite_loss))
        weights.append(negative_weight)

    if not parts:
        return zero
    return torch.stack(parts).sum() / pos_magnitude.new_tensor(sum(weights))


def negative_under_magnitude_loss(
    interface_delta: torch.Tensor,
    target_delta: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float,
    scale: float,
    margin: float = 0.0,
    zero: torch.Tensor,
) -> torch.Tensor:
    if scale <= 0.0:
        raise ValueError(f'negative under-magnitude scale must be positive, got {scale}.')

    flat_pred = interface_delta.reshape(-1)
    flat_target = target_delta.reshape(-1)
    flat_mask = mask.reshape(-1)
    negative_mask = flat_mask & (flat_target < 0.0)
    if not negative_mask.any():
        return zero

    target_abs = torch.abs(flat_target[negative_mask])
    pred_negative_abs = torch.relu(-flat_pred[negative_mask])
    under_magnitude = torch.relu(target_abs - pred_negative_abs - float(margin))
    scaled_under_magnitude = under_magnitude / float(scale)
    return F.smooth_l1_loss(
        scaled_under_magnitude,
        torch.zeros_like(scaled_under_magnitude),
        beta=beta,
    )


def residual_direction_logits(outputs: dict[str, torch.Tensor]) -> torch.Tensor | None:
    neg = outputs.get('residual_direction_logit_neg')
    pos = outputs.get('residual_direction_logit_pos')
    if neg is None or pos is None:
        return None
    return torch.stack([neg.reshape(-1), pos.reshape(-1)], dim=-1)



def masked_mean(values: torch.Tensor, mask: torch.Tensor, *, zero: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return zero
    return values[mask].mean()


def compute_physical_attribute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    config: dict[str, Any] | None,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    config = config if isinstance(config, dict) else {}
    required_outputs = (
        'sample_dry_mass_g_normalized_pred',
        'sample_capacity_ratio_normalized_pred',
        'sample_is_open_container_logit',
    )
    missing = [key for key in required_outputs if key not in outputs]
    if missing:
        raise RuntimeError(
            'Physical attribute loss is enabled, but model outputs are missing: '
            + ', '.join(missing)
            + ". Set model.physical_attributes.enabled=true for this stage."
        )

    beta = float(config.get('huber_beta', 1.0))
    weights = config.get('weights', {}) if isinstance(config.get('weights', {}), dict) else {}
    dry_weight = float(weights.get('dry_mass', config.get('dry_mass_weight', 0.2)))
    capacity_weight = float(weights.get('capacity_ratio', config.get('capacity_ratio_weight', 0.2)))
    open_weight = float(weights.get('is_open_container', config.get('is_open_container_weight', 0.2)))

    dry_pred = outputs['sample_dry_mass_g_normalized_pred'].reshape(-1)
    dry_target = batch['dry_mass_g_normalized'].to(device=dry_pred.device, dtype=dry_pred.dtype).reshape(-1)
    dry_loss = F.smooth_l1_loss(dry_pred, dry_target, beta=beta) if dry_weight > 0.0 else zero

    capacity_pred = outputs['sample_capacity_ratio_normalized_pred'].reshape(-1)
    capacity_target = batch['capacity_ratio_normalized'].to(device=capacity_pred.device, dtype=capacity_pred.dtype).reshape(-1)
    capacity_mask = batch['capacity_valid_mask'].to(device=capacity_pred.device).reshape(-1).bool()
    capacity_loss = (
        masked_smooth_l1(capacity_pred, capacity_target, capacity_mask, beta=beta, zero=zero)
        if capacity_weight > 0.0
        else zero
    )

    open_logits = outputs['sample_is_open_container_logit'].reshape(-1)
    open_target = batch['is_open_container'].to(device=open_logits.device, dtype=open_logits.dtype).reshape(-1)
    open_loss = F.binary_cross_entropy_with_logits(open_logits, open_target) if open_weight > 0.0 else zero

    return {
        'physical_dry_mass': dry_loss,
        'physical_capacity_ratio': capacity_loss,
        'physical_is_open_container': open_loss,
        'physical': dry_weight * dry_loss + capacity_weight * capacity_loss + open_weight * open_loss,
    }



def compute_policy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    force_targets: torch.Tensor,
    reference_targets: torch.Tensor | None,
    delta_targets: torch.Tensor | None,
    has_expert: torch.Tensor,
    has_reference: torch.Tensor,
    control_mask: torch.Tensor,
    reference_supervision_mask: torch.Tensor,
    delta_supervision_mask: torch.Tensor,
    quiet_supervision_mask: torch.Tensor,
    phase_targets: torch.Tensor,
    stable_mask: torch.Tensor,
    window_mask: torch.Tensor,
    zero: torch.Tensor,
    policy_loss_config: dict[str, Any] | None,
) -> torch.Tensor:
    force_pred = outputs['force_pred'].reshape(-1)
    policy_config = policy_loss_config or {}
    policy_type = str(policy_config.get('type', 'mse'))

    if policy_type == 'explicit_reference_delta_weighted_smooth_l1':
        all_weight = float(policy_config.get('all_weight', 0.25))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.25))
        base_reference_weight = float(policy_config.get('base_reference_weight', 1.0))
        base_stable_weight = float(policy_config.get('base_stable_weight', 1.0))
        base_normalization_scale = float(policy_config.get('base_normalization_scale', 1.0))
        residual_interface_weight = float(policy_config.get('residual_interface_weight', 6.0))
        residual_stable_zero_weight = float(policy_config.get('residual_stable_zero_weight', 1.0))
        residual_non_interface_weight = float(policy_config.get('residual_non_interface_weight', 0.25))
        gated_residual_interface_weight = float(policy_config.get('gated_residual_interface_weight', 0.0))
        gated_residual_min_gate = float(policy_config.get('gated_residual_min_gate', 0.0))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))
        delta_normalization_scale = float(policy_config.get('delta_normalization_scale', 1.0))
        residual_large_delta_weight = float(policy_config.get('residual_large_delta_weight', 0.0))
        large_delta_threshold = float(
            policy_config.get('residual_large_delta_threshold', policy_config.get('large_delta_threshold', 0.0))
        )
        large_delta_weight_alpha = float(policy_config.get('large_delta_weight_alpha', 0.0))
        large_delta_weight_cap = float(policy_config.get('large_delta_weight_cap', 4.0))
        residual_large_delta_sign_weight = float(policy_config.get('residual_large_delta_sign_weight', 0.0))
        large_delta_sign_margin = float(policy_config.get('large_delta_sign_margin', 0.0))
        sign_balanced_large_delta = bool(policy_config.get('sign_balanced_large_delta', False))
        large_delta_pos_loss_weight = float(policy_config.get('large_delta_pos_loss_weight', 1.0))
        large_delta_neg_loss_weight = float(policy_config.get('large_delta_neg_loss_weight', 1.0))
        residual_direction_large_delta_sign_weight = float(
            policy_config.get(
                'residual_direction_large_delta_sign_weight',
                policy_config.get('direction_gate_large_delta_sign_weight', 0.0),
            )
        )
        direction_gate_pos_loss_weight = float(
            policy_config.get('direction_gate_pos_loss_weight', large_delta_pos_loss_weight)
        )
        direction_gate_neg_loss_weight = float(
            policy_config.get('direction_gate_neg_loss_weight', large_delta_neg_loss_weight)
        )
        residual_sign_magnitude_weight = float(
            policy_config.get(
                'residual_sign_magnitude_weight',
                policy_config.get('residual_direction_magnitude_weight', 0.0),
            )
        )
        sign_magnitude_opposite_weight = float(
            policy_config.get(
                'sign_magnitude_opposite_weight',
                policy_config.get('direction_magnitude_opposite_weight', 0.0),
            )
        )
        sign_magnitude_scale = float(
            policy_config.get(
                'sign_magnitude_scale',
                policy_config.get('direction_magnitude_scale', delta_normalization_scale),
            )
        )
        sign_magnitude_pos_scale = float(
            policy_config.get(
                'sign_magnitude_pos_scale',
                policy_config.get('direction_magnitude_pos_scale', sign_magnitude_scale),
            )
        )
        sign_magnitude_neg_scale = float(
            policy_config.get(
                'sign_magnitude_neg_scale',
                policy_config.get('direction_magnitude_neg_scale', sign_magnitude_scale),
            )
        )
        sign_magnitude_beta = float(
            policy_config.get(
                'sign_magnitude_beta',
                policy_config.get('direction_magnitude_beta', beta),
            )
        )
        sign_magnitude_pos_loss_weight = float(
            policy_config.get('sign_magnitude_pos_loss_weight', large_delta_pos_loss_weight)
        )
        sign_magnitude_neg_loss_weight = float(
            policy_config.get('sign_magnitude_neg_loss_weight', large_delta_neg_loss_weight)
        )
        negative_under_magnitude_weight = float(
            policy_config.get(
                'negative_under_magnitude_weight',
                policy_config.get('residual_negative_under_magnitude_weight', 0.0),
            )
        )
        negative_under_magnitude_scale = float(
            policy_config.get('negative_under_magnitude_scale', delta_normalization_scale)
        )
        negative_under_magnitude_margin = float(policy_config.get('negative_under_magnitude_margin', 0.0))
        negative_under_magnitude_beta = float(policy_config.get('negative_under_magnitude_beta', beta))

        if reference_targets is None or delta_targets is None:
            raise RuntimeError('Explicit reference-delta policy loss requires reference_targets and delta_targets.')
        if 'force_base' not in outputs or 'force_interface_delta' not in outputs:
            raise RuntimeError('Explicit reference-delta policy loss requires force_base and force_interface_delta outputs.')

        base_force = outputs['force_base'].reshape(-1)
        interface_delta = outputs['force_interface_delta'].reshape(-1)
        sample_weight = outputs.get('policy_sample_weight')
        if sample_weight is not None:
            sample_weight = sample_weight.to(device=interface_delta.device, dtype=interface_delta.dtype).reshape(-1)
            if sample_weight.numel() != interface_delta.numel():
                raise RuntimeError(
                    'policy_sample_weight cannot be broadcast to force_interface_delta: '
                    f'{sample_weight.numel()} vs {interface_delta.numel()}.'
                )
        interface_gate = outputs['interface_gate'].reshape(-1) if 'interface_gate' in outputs else None
        if interface_gate is not None and interface_gate.numel() != interface_delta.numel():
            if interface_delta.numel() % max(1, interface_gate.numel()) != 0:
                raise RuntimeError(
                    'interface_gate cannot be broadcast to force_interface_delta: '
                    f'{interface_gate.numel()} vs {interface_delta.numel()}.'
                )
            repeat_factor = interface_delta.numel() // max(1, interface_gate.numel())
            interface_gate = interface_gate.reshape(-1, 1).expand(-1, repeat_factor).reshape(-1)
        interface_expert = delta_supervision_mask
        stable_expert = control_mask & stable_mask
        reference_mask = reference_supervision_mask
        stable_reference = reference_mask & stable_mask
        quiet_mask = quiet_supervision_mask
        zero_target = torch.zeros_like(interface_delta)

        total = zero
        if all_weight > 0.0:
            total = total + all_weight * masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)
        if base_reference_weight > 0.0:
            total = total + base_reference_weight * masked_scaled_smooth_l1(
                base_force,
                reference_targets,
                reference_mask,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if base_stable_weight > 0.0:
            total = total + base_stable_weight * masked_scaled_smooth_l1(
                base_force,
                reference_targets,
                stable_reference,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if residual_interface_weight > 0.0:
            total = total + residual_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                delta_targets,
                interface_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        large_delta_mask = interface_expert & (torch.abs(delta_targets) >= large_delta_threshold)
        if residual_large_delta_weight > 0.0:
            large_delta_loss = (
                sign_balanced_large_delta_loss(
                    interface_delta,
                    delta_targets,
                    large_delta_mask,
                    beta=beta,
                    scale=delta_normalization_scale,
                    weight_alpha=large_delta_weight_alpha,
                    weight_cap=large_delta_weight_cap,
                    positive_weight=large_delta_pos_loss_weight,
                    negative_weight=large_delta_neg_loss_weight,
                    sample_weight=sample_weight,
                    zero=zero,
                )
                if sign_balanced_large_delta
                else masked_weighted_scaled_smooth_l1(
                    interface_delta,
                    delta_targets,
                    large_delta_mask,
                    beta=beta,
                    scale=delta_normalization_scale,
                    weight_alpha=large_delta_weight_alpha,
                    weight_cap=large_delta_weight_cap,
                    sample_weight=sample_weight,
                    zero=zero,
                )
            )
            total = total + residual_large_delta_weight * large_delta_loss
        if residual_large_delta_sign_weight > 0.0:
            large_delta_sign_loss = (
                sign_balanced_large_delta_sign_loss(
                    interface_delta,
                    delta_targets,
                    large_delta_mask,
                    scale=delta_normalization_scale,
                    margin=large_delta_sign_margin,
                    positive_weight=large_delta_pos_loss_weight,
                    negative_weight=large_delta_neg_loss_weight,
                    sample_weight=sample_weight,
                    zero=zero,
                )
                if sign_balanced_large_delta
                else masked_large_delta_sign_loss(
                    interface_delta,
                    delta_targets,
                    large_delta_mask,
                    scale=delta_normalization_scale,
                    margin=large_delta_sign_margin,
                    sample_weight=sample_weight,
                    zero=zero,
                )
            )
            total = total + residual_large_delta_sign_weight * large_delta_sign_loss
        if residual_direction_large_delta_sign_weight > 0.0:
            direction_logits = residual_direction_logits(outputs)
            if direction_logits is None:
                raise RuntimeError(
                    'Direction-gate sign loss requires residual_direction_logit_neg and '
                    'residual_direction_logit_pos outputs. Use policy.head_type=state_residual_sign_specific.'
                )
            total = total + residual_direction_large_delta_sign_weight * sign_balanced_direction_gate_cross_entropy(
                direction_logits,
                delta_targets,
                large_delta_mask,
                positive_weight=direction_gate_pos_loss_weight,
                negative_weight=direction_gate_neg_loss_weight,
                sample_weight=sample_weight,
                zero=zero,
            )
        if residual_sign_magnitude_weight > 0.0:
            total = total + residual_sign_magnitude_weight * sign_specific_magnitude_loss(
                outputs,
                delta_targets,
                large_delta_mask,
                beta=sign_magnitude_beta,
                scale=sign_magnitude_scale,
                positive_scale=sign_magnitude_pos_scale,
                negative_scale=sign_magnitude_neg_scale,
                opposite_weight=sign_magnitude_opposite_weight,
                positive_weight=sign_magnitude_pos_loss_weight,
                negative_weight=sign_magnitude_neg_loss_weight,
                sample_weight=sample_weight,
                zero=zero,
            )
        if negative_under_magnitude_weight > 0.0:
            total = total + negative_under_magnitude_weight * negative_under_magnitude_loss(
                interface_delta,
                delta_targets,
                large_delta_mask,
                beta=negative_under_magnitude_beta,
                scale=negative_under_magnitude_scale,
                margin=negative_under_magnitude_margin,
                zero=zero,
            )
        if residual_stable_zero_weight > 0.0:
            total = total + residual_stable_zero_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                stable_reference,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_non_interface_weight > 0.0:
            total = total + residual_non_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                quiet_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if gated_residual_interface_weight > 0.0 and interface_gate is not None:
            gated_interface_mask = interface_expert & (interface_gate >= gated_residual_min_gate)
            total = total + gated_residual_interface_weight * masked_scaled_smooth_l1(
                interface_gate * interface_delta,
                delta_targets,
                gated_interface_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if quiet_weight > 0.0 and quiet_mask.any():
            quiet_delta = interface_delta[quiet_mask] / delta_normalization_scale
            total = total + quiet_weight * torch.mean(quiet_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    if policy_type == 'decomposed_interface_residual_weighted_smooth_l1':
        all_weight = float(policy_config.get('all_weight', 0.25))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.25))
        base_weight = float(policy_config.get('base_weight', 0.5))
        base_stable_weight = float(policy_config.get('base_stable_weight', 1.0))
        base_normalization_scale = float(policy_config.get('base_normalization_scale', 1.0))
        residual_interface_weight = float(policy_config.get('residual_interface_weight', 6.0))
        residual_stable_zero_weight = float(policy_config.get('residual_stable_zero_weight', 1.0))
        residual_non_interface_weight = float(policy_config.get('residual_non_interface_weight', 0.25))
        gated_residual_interface_weight = float(policy_config.get('gated_residual_interface_weight', 0.0))
        gated_residual_min_gate = float(policy_config.get('gated_residual_min_gate', 0.0))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))
        detach_base_target = bool(policy_config.get('detach_base_target', True))
        delta_normalization_scale = float(policy_config.get('delta_normalization_scale', 1.0))
        residual_large_delta_weight = float(policy_config.get('residual_large_delta_weight', 0.0))
        large_delta_threshold = float(
            policy_config.get('residual_large_delta_threshold', policy_config.get('large_delta_threshold', 0.0))
        )
        large_delta_weight_alpha = float(policy_config.get('large_delta_weight_alpha', 0.0))
        large_delta_weight_cap = float(policy_config.get('large_delta_weight_cap', 4.0))
        residual_large_delta_sign_weight = float(policy_config.get('residual_large_delta_sign_weight', 0.0))
        large_delta_sign_margin = float(policy_config.get('large_delta_sign_margin', 0.0))
        sign_balanced_large_delta = bool(policy_config.get('sign_balanced_large_delta', False))
        large_delta_pos_loss_weight = float(policy_config.get('large_delta_pos_loss_weight', 1.0))
        large_delta_neg_loss_weight = float(policy_config.get('large_delta_neg_loss_weight', 1.0))
        residual_direction_large_delta_sign_weight = float(
            policy_config.get(
                'residual_direction_large_delta_sign_weight',
                policy_config.get('direction_gate_large_delta_sign_weight', 0.0),
            )
        )
        direction_gate_pos_loss_weight = float(
            policy_config.get('direction_gate_pos_loss_weight', large_delta_pos_loss_weight)
        )
        direction_gate_neg_loss_weight = float(
            policy_config.get('direction_gate_neg_loss_weight', large_delta_neg_loss_weight)
        )
        residual_sign_magnitude_weight = float(
            policy_config.get(
                'residual_sign_magnitude_weight',
                policy_config.get('residual_direction_magnitude_weight', 0.0),
            )
        )
        sign_magnitude_opposite_weight = float(
            policy_config.get(
                'sign_magnitude_opposite_weight',
                policy_config.get('direction_magnitude_opposite_weight', 0.0),
            )
        )
        sign_magnitude_scale = float(
            policy_config.get(
                'sign_magnitude_scale',
                policy_config.get('direction_magnitude_scale', delta_normalization_scale),
            )
        )
        sign_magnitude_pos_scale = float(
            policy_config.get(
                'sign_magnitude_pos_scale',
                policy_config.get('direction_magnitude_pos_scale', sign_magnitude_scale),
            )
        )
        sign_magnitude_neg_scale = float(
            policy_config.get(
                'sign_magnitude_neg_scale',
                policy_config.get('direction_magnitude_neg_scale', sign_magnitude_scale),
            )
        )
        sign_magnitude_beta = float(
            policy_config.get(
                'sign_magnitude_beta',
                policy_config.get('direction_magnitude_beta', beta),
            )
        )
        sign_magnitude_pos_loss_weight = float(
            policy_config.get('sign_magnitude_pos_loss_weight', large_delta_pos_loss_weight)
        )
        sign_magnitude_neg_loss_weight = float(
            policy_config.get('sign_magnitude_neg_loss_weight', large_delta_neg_loss_weight)
        )
        negative_under_magnitude_weight = float(
            policy_config.get(
                'negative_under_magnitude_weight',
                policy_config.get('residual_negative_under_magnitude_weight', 0.0),
            )
        )
        negative_under_magnitude_scale = float(
            policy_config.get('negative_under_magnitude_scale', delta_normalization_scale)
        )
        negative_under_magnitude_margin = float(policy_config.get('negative_under_magnitude_margin', 0.0))
        negative_under_magnitude_beta = float(policy_config.get('negative_under_magnitude_beta', beta))

        if 'force_base' not in outputs or 'force_interface_delta' not in outputs:
            raise RuntimeError('Decomposed residual policy loss requires force_base and force_interface_delta outputs.')

        base_force = outputs['force_base'].reshape(-1)
        interface_delta = outputs['force_interface_delta'].reshape(-1)
        interface_gate = outputs['interface_gate'].reshape(-1) if 'interface_gate' in outputs else None
        interface_expert = control_mask & (phase_targets == 1)
        stable_expert = control_mask & stable_mask
        quiet_mask = quiet_supervision_mask

        total = zero
        if all_weight > 0.0:
            total = total + all_weight * masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)
        if base_weight > 0.0:
            total = total + base_weight * masked_scaled_smooth_l1(
                base_force,
                force_targets,
                control_mask,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )
        if base_stable_weight > 0.0:
            total = total + base_stable_weight * masked_scaled_smooth_l1(
                base_force,
                force_targets,
                stable_expert,
                beta=beta,
                scale=base_normalization_scale,
                zero=zero,
            )

        base_reference = base_force.detach() if detach_base_target else base_force
        delta_target = force_targets - base_reference
        zero_target = torch.zeros_like(interface_delta)
        if residual_interface_weight > 0.0:
            total = total + residual_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                delta_target,
                interface_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        large_delta_mask = interface_expert & (torch.abs(delta_target) >= large_delta_threshold)
        if residual_large_delta_weight > 0.0:
            large_delta_loss = (
                sign_balanced_large_delta_loss(
                    interface_delta,
                    delta_target,
                    large_delta_mask,
                    beta=beta,
                    scale=delta_normalization_scale,
                    weight_alpha=large_delta_weight_alpha,
                    weight_cap=large_delta_weight_cap,
                    positive_weight=large_delta_pos_loss_weight,
                    negative_weight=large_delta_neg_loss_weight,
                    zero=zero,
                )
                if sign_balanced_large_delta
                else masked_weighted_scaled_smooth_l1(
                    interface_delta,
                    delta_target,
                    large_delta_mask,
                    beta=beta,
                    scale=delta_normalization_scale,
                    weight_alpha=large_delta_weight_alpha,
                    weight_cap=large_delta_weight_cap,
                    zero=zero,
                )
            )
            total = total + residual_large_delta_weight * large_delta_loss
        if residual_large_delta_sign_weight > 0.0:
            large_delta_sign_loss = (
                sign_balanced_large_delta_sign_loss(
                    interface_delta,
                    delta_target,
                    large_delta_mask,
                    scale=delta_normalization_scale,
                    margin=large_delta_sign_margin,
                    positive_weight=large_delta_pos_loss_weight,
                    negative_weight=large_delta_neg_loss_weight,
                    zero=zero,
                )
                if sign_balanced_large_delta
                else masked_large_delta_sign_loss(
                    interface_delta,
                    delta_target,
                    large_delta_mask,
                    scale=delta_normalization_scale,
                    margin=large_delta_sign_margin,
                    zero=zero,
                )
            )
            total = total + residual_large_delta_sign_weight * large_delta_sign_loss
        if residual_direction_large_delta_sign_weight > 0.0:
            direction_logits = residual_direction_logits(outputs)
            if direction_logits is None:
                raise RuntimeError(
                    'Direction-gate sign loss requires residual_direction_logit_neg and '
                    'residual_direction_logit_pos outputs. Use policy.head_type=state_residual_sign_specific.'
                )
            total = total + residual_direction_large_delta_sign_weight * sign_balanced_direction_gate_cross_entropy(
                direction_logits,
                delta_target,
                large_delta_mask,
                positive_weight=direction_gate_pos_loss_weight,
                negative_weight=direction_gate_neg_loss_weight,
                zero=zero,
            )
        if residual_sign_magnitude_weight > 0.0:
            total = total + residual_sign_magnitude_weight * sign_specific_magnitude_loss(
                outputs,
                delta_target,
                large_delta_mask,
                beta=sign_magnitude_beta,
                scale=sign_magnitude_scale,
                positive_scale=sign_magnitude_pos_scale,
                negative_scale=sign_magnitude_neg_scale,
                opposite_weight=sign_magnitude_opposite_weight,
                positive_weight=sign_magnitude_pos_loss_weight,
                negative_weight=sign_magnitude_neg_loss_weight,
                zero=zero,
            )
        if negative_under_magnitude_weight > 0.0:
            total = total + negative_under_magnitude_weight * negative_under_magnitude_loss(
                interface_delta,
                delta_target,
                large_delta_mask,
                beta=negative_under_magnitude_beta,
                scale=negative_under_magnitude_scale,
                margin=negative_under_magnitude_margin,
                zero=zero,
            )
        if residual_stable_zero_weight > 0.0:
            total = total + residual_stable_zero_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                stable_expert,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if residual_non_interface_weight > 0.0:
            total = total + residual_non_interface_weight * masked_scaled_smooth_l1(
                interface_delta,
                zero_target,
                quiet_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if gated_residual_interface_weight > 0.0 and interface_gate is not None:
            gated_interface_mask = interface_expert & (interface_gate >= gated_residual_min_gate)
            total = total + gated_residual_interface_weight * masked_scaled_smooth_l1(
                interface_gate * interface_delta,
                delta_target,
                gated_interface_mask,
                beta=beta,
                scale=delta_normalization_scale,
                zero=zero,
            )
        if quiet_weight > 0.0 and quiet_mask.any():
            quiet_delta = interface_delta[quiet_mask] / delta_normalization_scale
            total = total + quiet_weight * torch.mean(quiet_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    if policy_type in {'interface_weighted_smooth_l1', 'interface_residual_weighted_smooth_l1'}:
        all_weight = float(policy_config.get('all_weight', 1.0))
        interface_weight = float(policy_config.get('interface_weight', 4.0))
        stable_weight = float(policy_config.get('stable_weight', 0.5))
        quiet_weight = float(policy_config.get('quiet_weight', 0.0))
        interface_positive_bias_weight = float(policy_config.get('interface_positive_bias_weight', 0.0))
        interface_bias_margin = float(policy_config.get('interface_bias_margin', 0.0))
        beta = float(policy_config.get('beta', 1.0))

        total = zero
        if all_weight > 0.0:
            total = total + masked_smooth_l1(force_pred, force_targets, control_mask, beta=beta, zero=zero)

        interface_expert = control_mask & (phase_targets == 1)
        if interface_weight > 0.0:
            total = total + interface_weight * masked_smooth_l1(force_pred, force_targets, interface_expert, beta=beta, zero=zero)

        stable_expert = control_mask & stable_mask
        if stable_weight > 0.0:
            total = total + stable_weight * masked_smooth_l1(force_pred, force_targets, stable_expert, beta=beta, zero=zero)

        if quiet_weight > 0.0 and 'force_interface_delta' in outputs and quiet_supervision_mask.any():
            interface_delta = outputs['force_interface_delta'].reshape(-1)[quiet_supervision_mask]
            total = total + quiet_weight * torch.mean(interface_delta ** 2)
        if interface_positive_bias_weight > 0.0 and interface_expert.any():
            interface_error = force_pred - force_targets
            positive_bias = torch.clamp(masked_mean(interface_error, interface_expert, zero=zero) - interface_bias_margin, min=0.0)
            total = total + interface_positive_bias_weight * (positive_bias ** 2)
        return total

    return F.mse_loss(force_pred[control_mask], force_targets[control_mask])



def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    loss_weights: dict[str, float],
    attribute_loss_config: dict[str, Any] | None = None,
    physical_loss_config: dict[str, Any] | None = None,
    policy_loss_config: dict[str, Any] | None,
    temperature_clip: float,
    temperature_inv: float,
    phase_class_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    window_mask = batch['window_mask'].reshape(-1)
    stable_mask = batch['stable_masks'].reshape(-1) & window_mask
    phase_targets = batch['phase_labels'].reshape(-1)
    medium_logits = outputs['medium_logits'].reshape(-1, outputs['medium_logits'].shape[-1])
    attr_logits = {task: outputs[ATTRIBUTE_LOGIT_KEYS[task]] for task in ATTRIBUTE_TASKS}
    sample_attr_logits = {
        task: outputs[ATTRIBUTE_SAMPLE_LOGIT_KEYS[task]]
        for task in ATTRIBUTE_TASKS
        if ATTRIBUTE_SAMPLE_LOGIT_KEYS[task] in outputs
    }

    repeated_attr_targets = {
        task: batch[ATTRIBUTE_LABEL_KEYS[task]][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
        for task in ATTRIBUTE_TASKS
    }
    repeated_object = batch['object_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    repeated_sample = batch['sample_index'][:, None].expand(-1, batch['window_mask'].shape[1]).reshape(-1)
    stable_phases = batch['stable_phases'].reshape(-1)

    losses: dict[str, torch.Tensor] = {}
    zero = medium_logits.new_tensor(0.0)
    med_weight = float(loss_weights.get('med', 0.0))
    attr_weight = float(loss_weights.get('attr', 0.0))
    physical_weight = float(loss_weights.get('physical', 0.0))
    clip_weight = float(loss_weights.get('clip', 0.0))
    inv_weight = float(loss_weights.get('inv', 0.0))
    pol_weight = float(loss_weights.get('pol', 0.0))
    attribute_loss_config = attribute_loss_config if isinstance(attribute_loss_config, dict) else {}
    attr_class_weights = attribute_loss_config.get('class_weights', {})
    attr_task_weights = attribute_loss_config.get('task_weights', {})
    attr_mode = str(
        attribute_loss_config.get(
            'mode',
            'sample_primary' if len(sample_attr_logits) == len(ATTRIBUTE_TASKS) else 'window',
        )
    ).strip().lower()
    if attr_mode in {'legacy', 'legacy_window', 'window', 'window_all'}:
        default_sample_weight = 0.0
        default_window_weight = 1.0
        default_window_selection = 'all_windows'
    else:
        default_sample_weight = 1.0
        default_window_weight = 0.2
        default_window_selection = 'stable_water_air'
    attr_sample_weight = float(attribute_loss_config.get('sample_weight', default_sample_weight))
    attr_window_weight = float(attribute_loss_config.get('window_weight', default_window_weight))
    attr_window_selection = str(attribute_loss_config.get('window_selection', default_window_selection)).strip().lower()

    def _optional_class_weight(task_name: str, logits: torch.Tensor) -> torch.Tensor | None:
        raw_weights = attr_class_weights.get(task_name) if isinstance(attr_class_weights, dict) else None
        if raw_weights is None:
            return None
        weights = torch.tensor(raw_weights, dtype=logits.dtype, device=logits.device)
        if weights.numel() != logits.shape[-1]:
            raise RuntimeError(
                f'attribute_loss.class_weights.{task_name} has {weights.numel()} values, '
                f'but logits have {logits.shape[-1]} classes.'
            )
        return weights

    def _task_weight(task_name: str) -> float:
        if not isinstance(attr_task_weights, dict):
            return 1.0
        return float(attr_task_weights.get(task_name, 1.0))

    def _attribute_window_loss_mask() -> torch.Tensor:
        masks = build_attribute_window_masks(batch)
        aliases = {
            'all': 'all_windows',
            'all_valid': 'all_windows',
            'stable': 'stable_windows',
            'stable_water': 'stable_water_only',
            'stable_air': 'stable_air_only',
            'stable_water_and_air': 'stable_water_air',
            'water': 'water_only',
            'air': 'air_only',
            'interface': 'interface_only',
        }
        mask_name = aliases.get(attr_window_selection, attr_window_selection)
        if mask_name not in masks:
            raise RuntimeError(
                f"Unsupported attribute_loss.window_selection={attr_window_selection!r}. "
                f"Expected one of {sorted(set(masks) | set(aliases))}."
            )
        return masks[mask_name].reshape(-1)

    losses['med'] = F.cross_entropy(medium_logits[window_mask], phase_targets[window_mask], weight=phase_class_weights) if med_weight > 0.0 else zero
    if attr_weight > 0.0:
        window_loss_mask = _attribute_window_loss_mask()
        sample_task_losses: dict[str, torch.Tensor] = {}
        window_task_losses: dict[str, torch.Tensor] = {}
        for task in ATTRIBUTE_TASKS:
            logits = attr_logits[task]
            if attr_window_weight > 0.0 and window_loss_mask.any():
                window_task_losses[task] = F.cross_entropy(
                    logits[window_loss_mask],
                    repeated_attr_targets[task][window_loss_mask],
                    weight=_optional_class_weight(task, logits),
                )
            else:
                window_task_losses[task] = zero

            sample_logits = sample_attr_logits.get(task)
            if attr_sample_weight > 0.0 and sample_logits is not None:
                sample_task_losses[task] = F.cross_entropy(
                    sample_logits,
                    batch[ATTRIBUTE_LABEL_KEYS[task]],
                    weight=_optional_class_weight(task, sample_logits),
                )
            else:
                sample_task_losses[task] = zero

        losses['frag'] = attr_sample_weight * sample_task_losses['fragility'] + attr_window_weight * window_task_losses['fragility']
        losses['geom'] = attr_sample_weight * sample_task_losses['geometry'] + attr_window_weight * window_task_losses['geometry']
        losses['surf'] = attr_sample_weight * sample_task_losses['surface'] + attr_window_weight * window_task_losses['surface']
        losses['attr_sample'] = (
            _task_weight('fragility') * sample_task_losses['fragility']
            + _task_weight('geometry') * sample_task_losses['geometry']
            + _task_weight('surface') * sample_task_losses['surface']
        )
        losses['attr_window'] = (
            _task_weight('fragility') * window_task_losses['fragility']
            + _task_weight('geometry') * window_task_losses['geometry']
            + _task_weight('surface') * window_task_losses['surface']
        )
        losses['attr'] = attr_sample_weight * losses['attr_sample'] + attr_window_weight * losses['attr_window']
    else:
        losses['frag'] = zero
        losses['geom'] = zero
        losses['surf'] = zero
        losses['attr_sample'] = zero
        losses['attr_window'] = zero
        losses['attr'] = zero

    if physical_weight > 0.0:
        losses.update(
            compute_physical_attribute_loss(
                outputs,
                batch,
                config=physical_loss_config,
                zero=zero,
            )
        )
    else:
        losses['physical_dry_mass'] = zero
        losses['physical_capacity_ratio'] = zero
        losses['physical_is_open_container'] = zero
        losses['physical'] = zero

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

    raw_policy_config = policy_loss_config if isinstance(policy_loss_config, dict) else {}
    policy_type = str(raw_policy_config.get('type', 'mse'))
    use_per_finger_policy = policy_type in {
        'explicit_reference_delta_per_finger_smooth_l1',
        'explicit_reference_delta_per_finger_weighted_smooth_l1',
    }

    effective_policy_loss_config = policy_loss_config
    if use_per_finger_policy:
        if 'finger_force_pred' not in outputs:
            raise RuntimeError(
                f"policy_loss.type={policy_type!r} requires a per-finger policy head."
            )
        if not all(
            key in batch
            for key in (
                'finger_control_force_targets',
                'finger_reference_forces',
                'finger_delta_force_targets',
                'has_finger_expert',
                'has_finger_control_target',
                'has_finger_reference',
            )
        ):
            raise RuntimeError('Per-finger policy loss requires per-finger force targets in the batch.')

        finger_window_mask = batch['window_mask'].unsqueeze(-1).expand_as(batch['finger_control_force_targets']).reshape(-1)
        finger_phase_targets = batch['phase_labels'].unsqueeze(-1).expand_as(batch['finger_control_force_targets']).reshape(-1)
        finger_stable_mask = batch['stable_masks'].unsqueeze(-1).expand_as(batch['finger_control_force_targets']).reshape(-1)
        force_targets = batch['finger_control_force_targets'].reshape(-1)
        reference_targets = batch['finger_reference_forces'].reshape(-1)
        delta_targets = batch['finger_delta_force_targets'].reshape(-1)
        has_expert = batch['has_finger_expert'].reshape(-1) & finger_window_mask
        has_reference = batch['has_finger_reference'].reshape(-1) & finger_window_mask
        control_mask = batch['has_finger_control_target'].reshape(-1) & finger_window_mask
        reference_supervision_mask = has_reference
        delta_supervision_mask = (
            batch['delta_supervision_masks'].unsqueeze(-1).expand_as(batch['finger_control_force_targets']).reshape(-1)
            & control_mask
        )
        quiet_supervision_mask = (
            batch['quiet_supervision_masks'].unsqueeze(-1).expand_as(batch['finger_control_force_targets']).reshape(-1)
            & control_mask
        )
        phase_targets_for_policy = finger_phase_targets
        stable_mask_for_policy = finger_stable_mask
        window_mask_for_policy = finger_window_mask
        effective_policy_loss_config = dict(raw_policy_config)
        effective_policy_loss_config['type'] = 'explicit_reference_delta_weighted_smooth_l1'
        policy_outputs = dict(outputs)
        policy_outputs['force_pred'] = outputs['finger_force_pred']
        policy_outputs['force_base'] = outputs['finger_force_base']
        policy_outputs['force_interface_delta'] = outputs['finger_force_interface_delta']
        if 'finger_force_interface_delta_pos_magnitude' in outputs:
            policy_outputs['force_interface_delta_pos_magnitude'] = outputs[
                'finger_force_interface_delta_pos_magnitude'
            ]
        if 'finger_force_interface_delta_neg_magnitude' in outputs:
            policy_outputs['force_interface_delta_neg_magnitude'] = outputs[
                'finger_force_interface_delta_neg_magnitude'
            ]
        if 'finger_residual_direction_logit_neg' in outputs:
            policy_outputs['residual_direction_logit_neg'] = outputs['finger_residual_direction_logit_neg']
        if 'finger_residual_direction_logit_pos' in outputs:
            policy_outputs['residual_direction_logit_pos'] = outputs['finger_residual_direction_logit_pos']
        finger_loss_weights = raw_policy_config.get('finger_loss_weights')
        if finger_loss_weights is not None:
            finger_count = int(batch['finger_control_force_targets'].shape[-1])
            if len(finger_loss_weights) != finger_count:
                raise RuntimeError(
                    'policy_loss.finger_loss_weights must match finger count: '
                    f'{len(finger_loss_weights)} vs {finger_count}.'
                )
            weight_tensor = torch.tensor(
                [float(weight) for weight in finger_loss_weights],
                device=batch['finger_control_force_targets'].device,
                dtype=batch['finger_control_force_targets'].dtype,
            )
            if (weight_tensor < 0).any():
                raise RuntimeError('policy_loss.finger_loss_weights must be non-negative.')
            view_shape = [1] * batch['finger_control_force_targets'].ndim
            view_shape[-1] = finger_count
            policy_outputs['policy_sample_weight'] = (
                weight_tensor.reshape(view_shape).expand_as(batch['finger_control_force_targets']).reshape(-1)
            )
    else:
        force_targets = batch['control_force_targets'].reshape(-1) if 'control_force_targets' in batch else batch['expert_forces'].reshape(-1)
        reference_targets = batch['reference_forces'].reshape(-1) if 'reference_forces' in batch else None
        delta_targets = batch['delta_force_targets'].reshape(-1) if 'delta_force_targets' in batch else None
        has_expert = batch['has_expert'].reshape(-1) & window_mask
        has_reference = batch['has_reference'].reshape(-1) & window_mask if 'has_reference' in batch else has_expert.new_zeros(has_expert.shape)
        control_mask = batch['has_control_target'].reshape(-1) & window_mask if 'has_control_target' in batch else has_expert
        reference_supervision_mask = batch['reference_supervision_masks'].reshape(-1) & window_mask if 'reference_supervision_masks' in batch else has_reference
        delta_supervision_mask = batch['delta_supervision_masks'].reshape(-1) & window_mask if 'delta_supervision_masks' in batch else (has_reference & (phase_targets == 1))
        quiet_supervision_mask = batch['quiet_supervision_masks'].reshape(-1) & window_mask if 'quiet_supervision_masks' in batch else (control_mask & (phase_targets != 1))
        phase_targets_for_policy = phase_targets
        stable_mask_for_policy = stable_mask
        window_mask_for_policy = window_mask
        policy_outputs = outputs
    if pol_weight > 0.0 and control_mask.any():
        losses['pol'] = compute_policy_loss(
            policy_outputs,
            force_targets=force_targets,
            reference_targets=reference_targets,
            delta_targets=delta_targets,
            has_expert=has_expert,
            has_reference=has_reference,
            control_mask=control_mask,
            reference_supervision_mask=reference_supervision_mask,
            delta_supervision_mask=delta_supervision_mask,
            quiet_supervision_mask=quiet_supervision_mask,
            phase_targets=phase_targets_for_policy,
            stable_mask=stable_mask_for_policy,
            window_mask=window_mask_for_policy,
            zero=zero,
            policy_loss_config=effective_policy_loss_config,
        )
    else:
        losses['pol'] = zero

    total = (
        loss_weights.get('clip', 0.0) * losses['clip']
        + loss_weights.get('inv', 0.0) * losses['inv']
        + loss_weights.get('med', 0.0) * losses['med']
        + loss_weights.get('attr', 0.0) * losses['attr']
        + loss_weights.get('physical', 0.0) * losses['physical']
        + loss_weights.get('pol', 0.0) * losses['pol']
    )
    losses['total'] = total
    return losses
