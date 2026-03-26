from __future__ import annotations

import torch
from torch import nn

from .modules import (
    MediumBeliefHead,
    MultiAttributeHead,
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
        )
        policy_input_dim = (
            visual_hidden_dim
            + int(tactile_config['content_hidden_dim'])
            + int(attribute_config['object_feature_dim'])
        )
        self.policy_head = PolicyHead(
            input_dim=policy_input_dim,
            hidden_dim=int(policy_config['hidden_dim']),
            film_hidden_dim=int(policy_config['film_hidden_dim']),
        )

    def _build_task_context(
        self,
        h_v: torch.Tensor,
        z_content: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        attribute_inputs = torch.cat([h_v, z_content], dim=-1)
        attribute_outputs = self.attribute_head(attribute_inputs)
        g_obj_context = attribute_outputs['g_obj_context'].detach() if self.stop_gradient else attribute_outputs['g_obj_context']
        g_obj = self.attribute_head.object_projection(g_obj_context)
        task_context = torch.cat([h_v, z_content, g_obj], dim=-1)
        return attribute_outputs, g_obj, task_context

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
        medium_logits, p_medium, medium_hidden = self.medium_head(z_med_sequence, batch['window_lengths'])
        flat_medium = p_medium.reshape(batch_size * max_windows, -1)

        attribute_outputs, g_obj, task_context = self._build_task_context(h_v, z_content)
        force_pred = self.policy_head(task_context, flat_medium)

        return {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': medium_hidden,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': force_pred,
        }

    def forward_online_step(
        self,
        batch: dict[str, torch.Tensor],
        *,
        medium_hidden: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_v, z_v = self.visual_encoder(batch['video'], batch['frame_mask'])
        z_content, z_t = self.content_encoder(batch['tactile_high'], batch['tactile_mask'])
        z_med_window = self.evidence_encoder(batch['tactile_low'], batch['tactile_mask'])
        medium_logits, p_medium, next_hidden = self.medium_head.step(z_med_window, hidden_state=medium_hidden)
        attribute_outputs, g_obj, task_context = self._build_task_context(h_v, z_content)
        force_pred = self.policy_head(task_context, p_medium)
        return {
            'h_v': h_v,
            'z_v': z_v,
            'z_t': z_t,
            'z_content': z_content,
            'z_med_window': z_med_window,
            'medium_logits': medium_logits,
            'medium_probs': p_medium,
            'medium_hidden': next_hidden,
            'fragility_logits': attribute_outputs['fragility_logits'],
            'geometry_logits': attribute_outputs['geometry_logits'],
            'surface_logits': attribute_outputs['surface_logits'],
            'fragility_entropy': attribute_outputs['fragility_entropy'],
            'geometry_entropy': attribute_outputs['geometry_entropy'],
            'surface_entropy': attribute_outputs['surface_entropy'],
            'g_obj': g_obj,
            'force_pred': force_pred,
        }
