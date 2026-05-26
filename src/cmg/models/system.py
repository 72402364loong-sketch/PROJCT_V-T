from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .modules import (
    MediumBeliefHead,
    MultiAttributeHead,
    PhaseAwareGateStabilizer,
    PolicyHead,
    TactileContentEncoder,
    TactileEvidenceEncoder,
    VisualEncoder,
)


class CrossMediumSystem(nn.Module):
    def __init__(self, model_config: dict) -> None:
        super().__init__()
        visual_config = model_config['visual']
        tactile_config = model_config['tactile']
        attribute_config = model_config['attributes']
        medium_config = model_config['medium']
        policy_config = model_config['policy']
        proj_dim = int(visual_config['proj_dim'])
        self.visual_proj_dim = proj_dim
        visual_hidden_dim = int(visual_config.get('hidden_dim', visual_config.get('token_dim', proj_dim)))
        self.stop_gradient = bool(attribute_config.get('stop_gradient', True))
        self.policy_base_source = str(policy_config.get('base_source', 'learned')).strip().lower()
        self.policy_gate_source = str(policy_config.get('gate_source', 'medium_prob')).strip().lower()
        self.use_interface_context = bool(policy_config.get('use_interface_context', False))
        self.tactile = SimpleNamespace(
            input_dim=int(tactile_config['input_dim']),
            num_taxels=int(tactile_config.get('num_taxels', 12)),
            axis_dim=int(tactile_config.get('axis_dim', max(1, int(tactile_config['input_dim']) // 12))),
        )
        gate_stabilizer_config = policy_config.get('gate_stabilizer', {}) if isinstance(policy_config.get('gate_stabilizer', {}), dict) else {}
        self.gate_stabilizer_enabled = bool(gate_stabilizer_config.get('enabled', False))
        if self.policy_base_source not in {'learned', 'reference_force'}:
            raise RuntimeError(
                f"Unsupported policy.base_source={self.policy_base_source!r}. Expected 'learned' or 'reference_force'."
            )
        if self.policy_gate_source not in {'medium_prob', 'soft_target', 'hard_phase'}:
            raise RuntimeError(
                f"Unsupported policy.gate_source={self.policy_gate_source!r}. "
                "Expected 'medium_prob', 'soft_target', or 'hard_phase'."
            )
        self.gate_stabilizer = (
            PhaseAwareGateStabilizer(
                on_threshold=float(gate_stabilizer_config.get('on_threshold', 0.25)),
                off_threshold=float(gate_stabilizer_config.get('off_threshold', 0.10)),
                min_on_windows=int(gate_stabilizer_config.get('min_on_windows', 1)),
                min_off_windows=int(gate_stabilizer_config.get('min_off_windows', 1)),
                output_mode=str(gate_stabilizer_config.get('output_mode', 'masked_raw')),
            )
            if self.gate_stabilizer_enabled
            else None
        )

        self.visual_encoder = VisualEncoder(visual_config)
        self.content_encoder = TactileContentEncoder(tactile_config, proj_dim=proj_dim)
        self.evidence_encoder = TactileEvidenceEncoder(tactile_config)
        self.medium_head = MediumBeliefHead(
            input_dim=int(tactile_config['evidence_hidden_dim']),
            hidden_dim=int(medium_config['hidden_dim']),
        )
        attribute_input_dim = visual_hidden_dim + int(tactile_config['content_hidden_dim'])
        self.attribute_head = MultiAttributeHead(
            input_dim=attribute_input_dim,
            hidden_dim=int(attribute_config['hidden_dim']),
            object_feature_dim=int(attribute_config['object_feature_dim']),
            fragility_classes=int(attribute_config.get('fragility_classes', 3)),
            geometry_classes=int(attribute_config.get('geometry_classes', 4)),
            surface_classes=int(attribute_config.get('surface_classes', 2)),
        )
        policy_input_dim = (
            visual_hidden_dim
            + int(tactile_config['content_hidden_dim'])
            + int(attribute_config['object_feature_dim'])
        )
        policy_context_dim = (
            visual_hidden_dim
            + int(tactile_config['content_hidden_dim'])
            + int(tactile_config['evidence_hidden_dim'])
        ) if self.use_interface_context else 0
        policy_state_dim = int(tactile_config['evidence_hidden_dim']) + int(medium_config['hidden_dim']) + 3
        self.policy_head = PolicyHead(
            input_dim=policy_input_dim,
            hidden_dim=int(policy_config['hidden_dim']),
            film_hidden_dim=int(policy_config['film_hidden_dim']),
            head_type=str(policy_config.get('head_type', 'legacy')),
            state_input_dim=policy_state_dim,
            context_input_dim=policy_context_dim,
            context_scale_init=float(policy_config.get('context_scale_init', 0.0)),
        )

    @staticmethod
    def _apply_reference_force_base_override(
        policy_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if 'reference_forces' not in batch:
            return policy_outputs

        learned_base = policy_outputs['force_base']
        reference_force = batch['reference_forces'].reshape(-1).to(device=learned_base.device, dtype=learned_base.dtype)
        finite_reference = torch.isfinite(reference_force)
        if not finite_reference.any():
            return policy_outputs

        oracle_base = torch.where(finite_reference, reference_force, learned_base)
        updated = dict(policy_outputs)
        updated['force_base_learned'] = learned_base
        updated['force_base'] = oracle_base
        updated['force_pred'] = oracle_base + updated['interface_gate'] * updated['force_interface_delta']
        return updated

    @staticmethod
    def _build_sample_attribute_pool_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        valid = batch['window_mask'].bool()
        stable = batch['stable_masks'].bool() & valid
        stable_phases = batch['stable_phases']
        stable_water_air = stable & ((stable_phases == 0) | (stable_phases == 2))

        has_stable_water_air = stable_water_air.any(dim=1, keepdim=True)
        has_stable = stable.any(dim=1, keepdim=True)
        return torch.where(
            has_stable_water_air,
            stable_water_air,
            torch.where(has_stable, stable, valid),
        )

    @staticmethod
    def _masked_mean_windows(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        counts = mask.sum(dim=1, keepdim=True)
        pooled = (features * mask.unsqueeze(-1).to(features.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(features.dtype)
        return pooled.masked_fill(counts == 0, 0.0)

    def _build_task_context(
        self,
        h_v: torch.Tensor,
        z_content: torch.Tensor,
        batch: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        attribute_inputs = torch.cat([h_v, z_content], dim=-1)
        attribute_outputs = self.attribute_head(attribute_inputs)
        sample_attribute_outputs = None
        sample_attribute_pool_mask = None
        if batch is not None:
            batch_size, max_windows = batch['window_mask'].shape
            sample_attribute_pool_mask = self._build_sample_attribute_pool_mask(batch)
            pooled_attribute_inputs = self._masked_mean_windows(
                attribute_inputs.reshape(batch_size, max_windows, -1),
                sample_attribute_pool_mask,
            )
            sample_attribute_outputs = self.attribute_head(pooled_attribute_inputs)
        g_obj_context = attribute_outputs['g_obj_context'].detach() if self.stop_gradient else attribute_outputs['g_obj_context']
        g_obj = self.attribute_head.object_projection(g_obj_context)
        task_context = torch.cat([h_v, z_content, g_obj], dim=-1)
        return attribute_outputs, sample_attribute_outputs, g_obj, task_context, sample_attribute_pool_mask

    @staticmethod
    def _pool_interface_context(
        context_features: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        if context_features.ndim != 3:
            raise RuntimeError(f'Expected context_features to be [B, W, D], got {tuple(context_features.shape)}.')
        if context_mask.ndim != 2:
            raise RuntimeError(f'Expected context_mask to be [B, W], got {tuple(context_mask.shape)}.')

        safe_mask = context_mask.bool()
        counts = safe_mask.sum(dim=1, keepdim=True)
        pooled = (context_features * safe_mask.unsqueeze(-1).to(context_features.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(context_features.dtype)
        pooled = pooled.masked_fill(counts == 0, 0.0)
        return pooled

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size, max_windows = batch['window_mask'].shape
        flat_tactile_high = batch['tactile_high'].reshape(batch_size * max_windows, *batch['tactile_high'].shape[2:])
        flat_tactile_low = batch['tactile_low'].reshape(batch_size * max_windows, *batch['tactile_low'].shape[2:])
        flat_tactile_mask = batch['tactile_mask'].reshape(batch_size * max_windows, batch['tactile_mask'].shape[-1])

        if batch.get('visual_features') is not None:
            h_v = batch['visual_features'].reshape(batch_size * max_windows, batch['visual_features'].shape[-1])
            z_v = h_v.new_zeros((h_v.shape[0], self.visual_proj_dim))
        else:
            flat_video = batch['video'].reshape(batch_size * max_windows, *batch['video'].shape[2:])
            flat_frame_mask = batch['frame_mask'].reshape(batch_size * max_windows, batch['frame_mask'].shape[-1])
            h_v, z_v = self.visual_encoder(flat_video, flat_frame_mask)
        z_content, z_t = self.content_encoder(flat_tactile_high, flat_tactile_mask)
        z_med_window = self.evidence_encoder(flat_tactile_low, flat_tactile_mask)
        z_med_sequence = z_med_window.reshape(batch_size, max_windows, -1)
        medium_logits, p_medium, medium_hidden, medium_sequence_features = self.medium_head(z_med_sequence, batch['window_lengths'])
        flat_medium = p_medium.reshape(batch_size * max_windows, -1)
        flat_medium_features = medium_sequence_features.reshape(batch_size * max_windows, -1)
        state_context = torch.cat([z_med_window, flat_medium_features, flat_medium], dim=-1)
        residual_context = None
        pooled_context = None
        if self.use_interface_context:
            context_mask = batch.get('context_window_mask')
            if context_mask is None:
                context_mask = torch.zeros_like(batch['window_mask'])
            context_features = torch.cat(
                [
                    h_v.reshape(batch_size, max_windows, -1),
                    z_content.reshape(batch_size, max_windows, -1),
                    z_med_window.reshape(batch_size, max_windows, -1),
                ],
                dim=-1,
            )
            effective_context_mask = context_mask.bool() & batch['window_mask'].bool()
            pooled_context = self._pool_interface_context(context_features, effective_context_mask)
            residual_context = pooled_context[:, None, :].expand(-1, max_windows, -1).reshape(batch_size * max_windows, -1)

        attribute_outputs, sample_attribute_outputs, g_obj, task_context, sample_attribute_pool_mask = self._build_task_context(
            h_v,
            z_content,
            batch,
        )
        interface_gate_override = None
        if self.policy_gate_source == 'soft_target':
            if 'soft_gate_targets' not in batch:
                raise RuntimeError("policy.gate_source='soft_target' requires soft_gate_targets in the batch.")
            interface_gate_override = batch['soft_gate_targets'].reshape(batch_size * max_windows)
        elif self.policy_gate_source == 'hard_phase':
            interface_gate_override = (batch['phase_labels'].reshape(batch_size * max_windows) == 1).to(flat_medium.dtype)
        elif self.gate_stabilizer is not None:
            raw_gate = p_medium[..., 1] if p_medium.shape[-1] > 1 else torch.zeros_like(batch['window_mask'], dtype=flat_medium.dtype)
            stabilized_gate = self.gate_stabilizer(raw_gate, batch['window_mask'])
            interface_gate_override = stabilized_gate.reshape(batch_size * max_windows)
        policy_outputs = self.policy_head(
            task_context,
            flat_medium,
            state_context=state_context,
            residual_context=residual_context,
            interface_gate_override=interface_gate_override,
        )
        if self.policy_base_source == 'reference_force':
            policy_outputs = self._apply_reference_force_base_override(policy_outputs, batch)

        outputs = {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': medium_hidden,
            'medium_sequence_features': medium_sequence_features,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': policy_outputs['force_pred'],
            'force_base': policy_outputs['force_base'],
            'force_interface_delta': policy_outputs['force_interface_delta'],
            'interface_gate': policy_outputs['interface_gate'],
        }
        if sample_attribute_outputs is not None:
            outputs.update(
                {
                    'sample_fragility_logits': sample_attribute_outputs['fragility_logits'],
                    'sample_geometry_logits': sample_attribute_outputs['geometry_logits'],
                    'sample_surface_logits': sample_attribute_outputs['surface_logits'],
                    'sample_fragility_entropy': sample_attribute_outputs['fragility_entropy'],
                    'sample_geometry_entropy': sample_attribute_outputs['geometry_entropy'],
                    'sample_surface_entropy': sample_attribute_outputs['surface_entropy'],
                }
            )
        if sample_attribute_pool_mask is not None:
            outputs['sample_attribute_pool_mask'] = sample_attribute_pool_mask
        if self.gate_stabilizer is not None and self.policy_gate_source == 'medium_prob':
            outputs['raw_interface_gate'] = flat_medium[..., 1] if flat_medium.shape[-1] > 1 else torch.zeros_like(policy_outputs['interface_gate'])
        if pooled_context is not None:
            outputs['interface_context_embedding'] = pooled_context
        if 'force_base_learned' in policy_outputs:
            outputs['force_base_learned'] = policy_outputs['force_base_learned']
        return outputs

    def forward_online_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        medium_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_v, z_v = self.visual_encoder(batch['video'], batch['frame_mask'])
        z_content, z_t = self.content_encoder(batch['tactile_high'], batch['tactile_mask'])
        z_med_window = self.evidence_encoder(batch['tactile_low'], batch['tactile_mask'])
        medium_logits, p_medium, next_hidden, medium_step_features = self.medium_head.step(z_med_window, hidden_state=medium_hidden)
        state_context = torch.cat([z_med_window, medium_step_features, p_medium], dim=-1)
        attribute_outputs, _, g_obj, task_context, _ = self._build_task_context(h_v, z_content)
        policy_outputs = self.policy_head(task_context, p_medium, state_context=state_context, residual_context=None)
        if self.policy_base_source == 'reference_force':
            policy_outputs = self._apply_reference_force_base_override(policy_outputs, batch)
        outputs = {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': next_hidden,
            'medium_sequence_features': medium_step_features,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': policy_outputs['force_pred'],
            'force_base': policy_outputs['force_base'],
            'force_interface_delta': policy_outputs['force_interface_delta'],
            'interface_gate': policy_outputs['interface_gate'],
        }
        if 'force_base_learned' in policy_outputs:
            outputs['force_base_learned'] = policy_outputs['force_base_learned']
        return outputs
