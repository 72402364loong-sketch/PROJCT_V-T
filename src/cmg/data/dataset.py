from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cmg.constants import FRAGILITY_TO_INDEX, GEOMETRY_TO_INDEX, PHASE_TO_INDEX, SURFACE_TO_INDEX

from .splits import resolve_sample_ids
from .tactile import (
    build_tactile_time_axis,
    compute_clean_force_curve,
    compute_expert_force_curve,
    load_tactile_array,
    normalize_normal_sign_table,
    resample_tactile_window,
    split_ac_dc,
    standardize_tactile_window,
)
from .video import load_frames_by_indices, sample_frame_indices
from .visual_cache import load_sample_visual_feature_cache, resolve_sample_visual_feature_cache_path, resolve_visual_feature_cache_dir


class CrossMediumSequenceDataset(Dataset):
    def __init__(
        self,
        project_root: str | Path,
        split_path: str | Path,
        subset: str,
        *,
        num_frames_per_window: int,
        image_size: int,
        roi: dict[str, int] | None,
        tactile_dt: float,
        acdc_alpha: float,
        expert_alpha: float,
        normal_sign_table: Any,
        clip_mean: list[float] | None = None,
        clip_std: list[float] | None = None,
        tactile_points_per_window: int | None = None,
        standardize_tactile: bool = False,
        min_valid_ratio_video: float = 0.0,
        min_valid_ratio_tactile: float = 0.0,
        normalization_subset: str = 'train',
        normalization_trial_results: Iterable[str] | None = None,
        tail_mode: str = 'all_valid',
        allowed_trial_results: Iterable[str] | None = None,
        visual_feature_cache_dir: str | Path | None = None,
        reference_force_window_count: int = 3,
        reference_force_statistic: str = 'mean',
        expert_force_mode: str = 'measured_force',
        expert_force_smoothing: str = 'ema',
        expert_force_baseline_mode: str = 'none',
        expert_force_baseline_window_sec: float = 0.5,
        expert_force_interface_margin_sec: float = 0.0,
    ) -> None:
        self.project_root = Path(project_root)
        self.split_path = Path(split_path)
        self.subset = subset
        self.num_frames_per_window = num_frames_per_window
        self.image_size = image_size
        self.roi = roi
        self.tactile_dt = tactile_dt
        self.acdc_alpha = acdc_alpha
        self.expert_alpha = expert_alpha
        self.normal_sign_table = normalize_normal_sign_table(normal_sign_table)
        self.clip_mean = None if clip_mean is None else torch.tensor(clip_mean, dtype=torch.float32).view(1, 3, 1, 1)
        self.clip_std = None if clip_std is None else torch.tensor(clip_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.tactile_points_per_window = tactile_points_per_window
        self.standardize_tactile = standardize_tactile
        self.min_valid_ratio_video = float(min_valid_ratio_video)
        self.min_valid_ratio_tactile = float(min_valid_ratio_tactile)
        self.normalization_subset = normalization_subset
        self.allowed_trial_results = None if allowed_trial_results is None else {str(value).strip().lower() for value in allowed_trial_results}
        self.normalization_trial_results = None if normalization_trial_results is None else {str(value).strip().lower() for value in normalization_trial_results}
        self.tail_mode = tail_mode
        self.visual_feature_cache_dir = resolve_visual_feature_cache_dir(self.project_root, visual_feature_cache_dir)
        self.reference_force_window_count = max(1, int(reference_force_window_count))
        self.reference_force_statistic = str(reference_force_statistic or 'mean').strip().lower()
        self.expert_force_mode = str(expert_force_mode or 'measured_force').strip().lower()
        self.expert_force_smoothing = str(expert_force_smoothing or 'ema').strip().lower()
        self.expert_force_baseline_mode = str(expert_force_baseline_mode or 'none').strip().lower()
        self.expert_force_baseline_window_sec = float(expert_force_baseline_window_sec)
        self.expert_force_interface_margin_sec = float(expert_force_interface_margin_sec)

        samples_path = self.project_root / 'data' / 'processed' / 'samples.csv'
        windows_path = self.project_root / 'data' / 'processed' / 'windows.csv'
        self.all_samples = pd.read_csv(samples_path)
        self.samples = self.all_samples.copy()
        self.windows = pd.read_csv(windows_path)
        sample_ids = set(resolve_sample_ids(self.all_samples, self.split_path, subset=subset))
        self.samples = self.samples.loc[self.samples['sample_id'].isin(sample_ids)].copy()
        self.windows = self.windows.loc[self.windows['sample_id'].isin(sample_ids)].copy()

        if self.allowed_trial_results is not None:
            sample_mask = self.samples['trial_result'].astype(str).str.strip().str.lower().isin(self.allowed_trial_results)
            window_mask = self.windows['trial_result'].astype(str).str.strip().str.lower().isin(self.allowed_trial_results)
            self.samples = self.samples.loc[sample_mask].copy()
            self.windows = self.windows.loc[window_mask].copy()

        if self.min_valid_ratio_video > 0.0 and 'valid_ratio_video' in self.windows.columns:
            self.windows = self.windows.loc[self.windows['valid_ratio_video'] >= self.min_valid_ratio_video].copy()
        if self.min_valid_ratio_tactile > 0.0 and 'valid_ratio_tactile' in self.windows.columns:
            self.windows = self.windows.loc[self.windows['valid_ratio_tactile'] >= self.min_valid_ratio_tactile].copy()

        if tail_mode == 'full_tail_only':
            self.windows = self.windows.loc[self.windows['tail_type'] == 'full_tail']
        elif tail_mode == 'short_tail_only':
            self.windows = self.windows.loc[self.windows['tail_type'] == 'short_tail']

        valid_sample_ids = set(self.windows['sample_id'].unique().tolist())
        self.samples = self.samples.loc[self.samples['sample_id'].isin(valid_sample_ids)].copy()
        self.sample_records = self.samples.sort_values('sample_id').to_dict('records')
        self.windows_by_sample = {
            sample_id: group.sort_values('window_start').to_dict('records')
            for sample_id, group in self.windows.groupby('sample_id')
        }
        self._annotate_sample_records_with_interface_stats()
        self.object_index = {
            object_id: index
            for index, object_id in enumerate(sorted(self.samples['object_id'].unique().tolist()))
        }
        self.sample_index = {
            sample_id: index
            for index, sample_id in enumerate(sorted(self.samples['sample_id'].unique().tolist()))
        }

        self.tactile_high_mean = None
        self.tactile_high_std = None
        self.tactile_low_mean = None
        self.tactile_low_std = None
        if self.standardize_tactile:
            stats = self._load_or_compute_tactile_stats()
            self.tactile_high_mean = np.asarray(stats['high_mean'], dtype=np.float32)
            self.tactile_high_std = np.asarray(stats['high_std'], dtype=np.float32)
            self.tactile_low_mean = np.asarray(stats['low_mean'], dtype=np.float32)
            self.tactile_low_std = np.asarray(stats['low_std'], dtype=np.float32)

    def _annotate_sample_records_with_interface_stats(self) -> None:
        interface_index = int(PHASE_TO_INDEX['Interface'])
        for sample in self.sample_records:
            sample_id = str(sample['sample_id'])
            windows = self.windows_by_sample.get(sample_id, [])
            total_windows = len(windows)
            interface_window_count = 0
            trial_result = str(sample.get('trial_result', '')).strip().lower()
            has_expert_supervision = trial_result == 'stable'
            for window in windows:
                phase_label = str(window['phase_label'])
                if int(PHASE_TO_INDEX[phase_label]) == interface_index:
                    interface_window_count += 1
            sample['interface_window_count'] = int(interface_window_count)
            sample['interface_expert_count'] = int(interface_window_count if has_expert_supervision else 0)
            sample['interface_window_ratio'] = float(interface_window_count / max(1, total_windows))

    def __len__(self) -> int:
        return len(self.sample_records)

    def _resolve_data_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            return relative
        if relative.parts and relative.parts[0] == 'data':
            return self.project_root / relative
        return self.project_root / 'data' / relative

    def _stats_cache_path(self) -> Path:
        cache_root = self.project_root / 'data' / 'processed' / 'cache'
        cache_root.mkdir(parents=True, exist_ok=True)
        suffix_parts = [self.split_path.stem, self.normalization_subset, f"alpha_{str(self.acdc_alpha).replace('.', 'p')}"]
        if self.normalization_trial_results:
            suffix_parts.append('trials_' + '_'.join(sorted(self.normalization_trial_results)))
        return cache_root / ('tactile_stats_' + '__'.join(suffix_parts) + '.json')

    def _load_or_compute_tactile_stats(self) -> dict[str, list[float]]:
        cache_path = self._stats_cache_path()
        if cache_path.exists():
            with cache_path.open('r', encoding='utf-8') as handle:
                return json.load(handle)

        train_sample_ids = set(resolve_sample_ids(self.all_samples, self.split_path, subset=self.normalization_subset))
        stat_samples = self.all_samples.loc[self.all_samples['sample_id'].isin(train_sample_ids)].copy()
        if self.normalization_trial_results is not None:
            mask = stat_samples['trial_result'].astype(str).str.strip().str.lower().isin(self.normalization_trial_results)
            stat_samples = stat_samples.loc[mask].copy()

        high_sum = np.zeros(36, dtype=np.float64)
        high_sq_sum = np.zeros(36, dtype=np.float64)
        low_sum = np.zeros(36, dtype=np.float64)
        low_sq_sum = np.zeros(36, dtype=np.float64)
        count = 0
        for sample in stat_samples.to_dict('records'):
            tactile_path = self._resolve_data_path(sample['tactile_path'])
            tactile_array = load_tactile_array(tactile_path)
            tactile_high, tactile_low = split_ac_dc(tactile_array, alpha=self.acdc_alpha)
            high_sum += tactile_high.sum(axis=0)
            high_sq_sum += np.square(tactile_high, dtype=np.float64).sum(axis=0)
            low_sum += tactile_low.sum(axis=0)
            low_sq_sum += np.square(tactile_low, dtype=np.float64).sum(axis=0)
            count += int(tactile_high.shape[0])

        if count == 0:
            stats = {
                'high_mean': [0.0] * 36,
                'high_std': [1.0] * 36,
                'low_mean': [0.0] * 36,
                'low_std': [1.0] * 36,
            }
        else:
            high_mean = high_sum / count
            low_mean = low_sum / count
            high_var = np.maximum(high_sq_sum / count - np.square(high_mean), 1e-6)
            low_var = np.maximum(low_sq_sum / count - np.square(low_mean), 1e-6)
            stats = {
                'high_mean': high_mean.astype(np.float32).tolist(),
                'high_std': np.sqrt(high_var).astype(np.float32).tolist(),
                'low_mean': low_mean.astype(np.float32).tolist(),
                'low_std': np.sqrt(low_var).astype(np.float32).tolist(),
            }

        with cache_path.open('w', encoding='utf-8') as handle:
            json.dump(stats, handle, ensure_ascii=False, indent=2)
        return stats

    def _compute_window_force_value(
        self,
        force_curve: np.ndarray,
        start_idx: int,
        end_idx: int,
        *,
        sample_mask: np.ndarray | None = None,
    ) -> float:
        segment = force_curve[start_idx:end_idx]
        if sample_mask is not None:
            segment = segment[sample_mask]
        if segment.size == 0:
            return float('nan')
        return float(np.mean(segment))

    def _resolve_reference_window_indices(self, windows: list[dict[str, Any]]) -> list[int]:
        first_interface_index = len(windows)
        for window_index, window in enumerate(windows):
            if str(window['phase_label']) == 'Interface':
                first_interface_index = window_index
                break

        stable_water_indices = [
            item_index
            for item_index, window in enumerate(windows[:first_interface_index])
            if bool(window['is_stable_mask']) and str(window['phase_label']) == 'Water'
        ]
        if stable_water_indices:
            return stable_water_indices[-self.reference_force_window_count :]

        stable_indices = [
            item_index
            for item_index, window in enumerate(windows[:first_interface_index])
            if bool(window['is_stable_mask'])
        ]
        return stable_indices[-self.reference_force_window_count :]

    def _aggregate_reference_force(self, values: list[float]) -> tuple[float, bool]:
        finite_values = [float(value) for value in values if np.isfinite(value)]
        if not finite_values:
            return float('nan'), False
        array = np.asarray(finite_values, dtype=np.float32)
        if self.reference_force_statistic == 'median':
            return float(np.median(array)), True
        return float(np.mean(array)), True



    @staticmethod
    def _parse_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if np.isnan(parsed):
            return None
        return parsed

    def _compute_sample_reference_force(
        self,
        windows: list[dict[str, Any]],
        force_curve: np.ndarray | None,
    ) -> tuple[float, bool, set[int]]:
        if force_curve is None:
            return float('nan'), False, set()
        reference_indices = self._resolve_reference_window_indices(windows)
        values = [
            self._compute_window_force_value(
                force_curve,
                int(windows[window_index]['tactile_start_idx']),
                int(windows[window_index]['tactile_end_idx']),
            )
            for window_index in reference_indices
        ]
        reference_force, has_reference = self._aggregate_reference_force(values)
        return reference_force, has_reference, set(reference_indices)

    def _window_interval_mask(
        self,
        time_axis: np.ndarray,
        start_idx: int,
        end_idx: int,
        *,
        interval_start: float,
        interval_end: float,
    ) -> np.ndarray:
        window_times = time_axis[start_idx:end_idx]
        if window_times.size == 0:
            return np.zeros(0, dtype=bool)
        return (window_times >= float(interval_start)) & (window_times < float(interval_end))

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.sample_records[index]
        sample_id = str(sample['sample_id'])
        windows = self.windows_by_sample[sample_id]
        tactile_path = self._resolve_data_path(sample['tactile_path'])
        tactile_array = load_tactile_array(tactile_path)
        tactile_high, tactile_low = split_ac_dc(tactile_array, alpha=self.acdc_alpha)

        expert_curve = compute_expert_force_curve(
            tactile_array=tactile_array,
            dt=self.tactile_dt,
            trial_result=sample['trial_result'],
            contact_time=sample.get('t_contact_all'),
            alpha=self.expert_alpha,
            normal_sign_table=self.normal_sign_table,
        )

        clean_force_curve = None
        tactile_time_axis = None
        interface_interval_start = None
        interface_interval_end = None
        if expert_curve is not None and self.expert_force_mode == 'local_reference_delta':
            sync_offset_sec = self._parse_optional_float(sample.get('sync_offset_sec')) or 0.0
            contact_time = self._parse_optional_float(sample.get('t_contact_all'))
            clean_force_curve = compute_clean_force_curve(
                tactile_array,
                dt=self.tactile_dt,
                sync_offset_sec=sync_offset_sec,
                contact_time=contact_time,
                alpha=self.expert_alpha,
                normal_sign_table=self.normal_sign_table,
                smoothing_mode=self.expert_force_smoothing,
                baseline_mode=self.expert_force_baseline_mode,
                baseline_window_sec=self.expert_force_baseline_window_sec,
            )
            tactile_time_axis = build_tactile_time_axis(
                clean_force_curve.shape[0],
                dt=self.tactile_dt,
                offset_sec=sync_offset_sec,
            )
            t_if_enter = self._parse_optional_float(sample.get('t_if_enter'))
            t_if_exit = self._parse_optional_float(sample.get('t_if_exit'))
            if t_if_enter is not None and t_if_exit is not None and t_if_exit > t_if_enter:
                interface_interval_start = float(t_if_enter) - self.expert_force_interface_margin_sec
                interface_interval_end = float(t_if_exit) + self.expert_force_interface_margin_sec

        reference_curve = clean_force_curve if clean_force_curve is not None else expert_curve
        sample_reference_force, has_sample_reference, reference_window_indices = self._compute_sample_reference_force(
            windows,
            reference_curve,
        )
        use_local_reference_targets = (
            self.expert_force_mode == 'local_reference_delta'
            and clean_force_curve is not None
            and tactile_time_axis is not None
            and has_sample_reference
            and interface_interval_start is not None
            and interface_interval_end is not None
        )

        visual_feature_windows: list[torch.Tensor] | None = None
        video_windows: list[torch.Tensor] | None = None
        frame_masks: list[torch.Tensor] | None = None
        tactile_high_windows: list[torch.Tensor] = []
        tactile_low_windows: list[torch.Tensor] = []
        tactile_window_masks: list[torch.Tensor] = []
        tactile_lengths: list[int] = []
        phase_labels: list[int] = []
        stable_masks: list[int] = []
        stable_phases: list[int] = []
        expert_forces: list[float] = []
        control_force_targets: list[float] = []
        reference_forces: list[float] = []
        delta_force_targets: list[float] = []
        has_expert: list[int] = []
        has_control_target: list[int] = []
        has_reference: list[int] = []
        reference_supervision_masks: list[int] = []
        delta_supervision_masks: list[int] = []
        quiet_supervision_masks: list[int] = []

        cache_path = None
        if self.visual_feature_cache_dir is not None:
            cache_path = resolve_sample_visual_feature_cache_path(self.project_root, self.visual_feature_cache_dir, sample_id)
        if cache_path is not None and cache_path.exists():
            payload = load_sample_visual_feature_cache(cache_path)
            feature_map = {
                str(window_id): payload['visual_features'][feature_index]
                for feature_index, window_id in enumerate(payload['window_ids'].tolist())
            }
            visual_feature_windows = [
                torch.from_numpy(np.asarray(feature_map[str(window['window_id'])], dtype=np.float32))
                for window in windows
            ]
        else:
            video_path = self._resolve_data_path(sample['video_path'])
            video_windows = []
            frame_masks = []
            window_video_specs: list[tuple[list[int], list[bool]]] = []
            sampled_frame_indices: list[int] = []
            for window in windows:
                all_frame_indices = json.loads(window['video_frame_indices_json'])
                sampled_indices, frame_mask = sample_frame_indices(all_frame_indices, self.num_frames_per_window)
                window_video_specs.append((sampled_indices, frame_mask))
                sampled_frame_indices.extend(sampled_indices)

            frame_map = load_frames_by_indices(
                video_path=video_path,
                frame_indices=sampled_frame_indices,
                image_size=self.image_size,
                roi=self.roi,
            )

            for sampled_indices, frame_mask in window_video_specs:
                frame_array = np.stack([frame_map[int(frame_index)] for frame_index in sampled_indices], axis=0)
                frame_tensor = torch.from_numpy(frame_array).permute(0, 3, 1, 2)
                if self.clip_mean is not None and self.clip_std is not None:
                    frame_tensor = (frame_tensor - self.clip_mean) / self.clip_std
                video_windows.append(frame_tensor)
                frame_masks.append(torch.tensor(frame_mask, dtype=torch.bool))

        for window_index, window in enumerate(windows):
            start_idx = int(window['tactile_start_idx'])
            end_idx = int(window['tactile_end_idx'])
            tactile_high_window = tactile_high[start_idx:end_idx]
            tactile_low_window = tactile_low[start_idx:end_idx]
            tactile_mask_window = np.ones(tactile_high_window.shape[0], dtype=bool)
            if self.tactile_points_per_window is not None:
                tactile_high_window, high_mask = resample_tactile_window(tactile_high_window, self.tactile_points_per_window)
                tactile_low_window, low_mask = resample_tactile_window(tactile_low_window, self.tactile_points_per_window)
                tactile_mask_window = high_mask & low_mask
            if self.standardize_tactile and self.tactile_high_mean is not None and self.tactile_low_mean is not None:
                tactile_high_window = standardize_tactile_window(tactile_high_window, self.tactile_high_mean, self.tactile_high_std)
                tactile_low_window = standardize_tactile_window(tactile_low_window, self.tactile_low_mean, self.tactile_low_std)
            tactile_high_windows.append(torch.from_numpy(tactile_high_window.astype(np.float32)))
            tactile_low_windows.append(torch.from_numpy(tactile_low_window.astype(np.float32)))
            tactile_window_masks.append(torch.from_numpy(tactile_mask_window.astype(bool)))
            tactile_lengths.append(int(tactile_mask_window.sum()))

            phase_label = str(window['phase_label'])
            is_interface_window = phase_label == 'Interface'
            phase_labels.append(PHASE_TO_INDEX[phase_label])
            stable_masks.append(int(window['is_stable_mask']))
            stable_phase = str(window.get('stable_phase', '') or '')
            stable_phases.append(PHASE_TO_INDEX[stable_phase] if stable_phase in PHASE_TO_INDEX else -1)

            expert_force = float('nan')
            control_force_target = float('nan')
            reference_force = float('nan')
            delta_force_target = float('nan')
            has_expert_flag = 0
            has_control_target_flag = 0
            has_reference_flag = 0
            reference_supervision_flag = 0
            delta_supervision_flag = 0
            quiet_supervision_flag = 0

            if expert_curve is not None:
                expert_force = self._compute_window_force_value(expert_curve, start_idx, end_idx)
                has_expert_flag = int(np.isfinite(expert_force))

            if has_sample_reference:
                reference_force = float(sample_reference_force)

            is_reference_window = window_index in reference_window_indices
            if use_local_reference_targets:
                local_overlap_mask = self._window_interval_mask(
                    tactile_time_axis,
                    start_idx,
                    end_idx,
                    interval_start=float(interface_interval_start),
                    interval_end=float(interface_interval_end),
                )
                has_local_overlap = bool(local_overlap_mask.any())
                reference_supervision_flag = int(is_reference_window or is_interface_window)
                delta_supervision_flag = int(is_interface_window)
                quiet_supervision_flag = int((not is_interface_window) and (is_reference_window or has_local_overlap))
                has_reference_flag = reference_supervision_flag

                if is_interface_window:
                    target_force = self._compute_window_force_value(
                        clean_force_curve,
                        start_idx,
                        end_idx,
                        sample_mask=local_overlap_mask if has_local_overlap else None,
                    )
                    if not np.isfinite(target_force):
                        target_force = self._compute_window_force_value(clean_force_curve, start_idx, end_idx)
                    if np.isfinite(target_force):
                        control_force_target = float(target_force)
                        delta_force_target = float(target_force - sample_reference_force)
                        has_control_target_flag = 1
                elif quiet_supervision_flag or reference_supervision_flag:
                    control_force_target = float(sample_reference_force)
                    delta_force_target = 0.0
                    has_control_target_flag = 1
            else:
                if has_expert_flag:
                    control_force_target = float(expert_force)
                    has_control_target_flag = 1
                if has_sample_reference and has_expert_flag:
                    reference_supervision_flag = int(is_reference_window or is_interface_window)
                    delta_supervision_flag = int(is_interface_window)
                    quiet_supervision_flag = int(not is_interface_window)
                    has_reference_flag = reference_supervision_flag
                    delta_force_target = float(expert_force - sample_reference_force)

            expert_forces.append(expert_force)
            control_force_targets.append(control_force_target)
            reference_forces.append(reference_force)
            delta_force_targets.append(delta_force_target)
            has_expert.append(has_expert_flag)
            has_control_target.append(has_control_target_flag)
            has_reference.append(has_reference_flag)
            reference_supervision_masks.append(reference_supervision_flag)
            delta_supervision_masks.append(delta_supervision_flag)
            quiet_supervision_masks.append(quiet_supervision_flag)

        return {
            'sample_id': sample_id,
            'sample_index': self.sample_index[sample_id],
            'object_id': sample['object_id'],
            'object_index': self.object_index[sample['object_id']],
            'visual_feature_windows': visual_feature_windows,
            'video_windows': video_windows,
            'frame_masks': frame_masks,
            'tactile_high_windows': tactile_high_windows,
            'tactile_low_windows': tactile_low_windows,
            'tactile_window_masks': tactile_window_masks,
            'tactile_lengths': tactile_lengths,
            'phase_labels': phase_labels,
            'stable_masks': stable_masks,
            'stable_phases': stable_phases,
            'expert_forces': expert_forces,
            'control_force_targets': control_force_targets,
            'reference_forces': reference_forces,
            'delta_force_targets': delta_force_targets,
            'has_expert': has_expert,
            'has_control_target': has_control_target,
            'has_reference': has_reference,
            'reference_supervision_masks': reference_supervision_masks,
            'delta_supervision_masks': delta_supervision_masks,
            'quiet_supervision_masks': quiet_supervision_masks,
            'fragility_label': FRAGILITY_TO_INDEX[str(sample['fragility'])],
            'geometry_label': GEOMETRY_TO_INDEX[str(sample['geometry'])],
            'surface_label': SURFACE_TO_INDEX[str(sample['surface'])],
        }

def sequence_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_windows = max(len(item['tactile_high_windows']) for item in batch)
    max_tactile = max(max(item['tactile_high_windows'][window_index].shape[0] for window_index in range(len(item['tactile_high_windows']))) for item in batch)
    use_visual_features = batch[0].get('visual_feature_windows') is not None

    visual_features = None
    video = None
    frame_mask = None
    if use_visual_features:
        feature_dim = batch[0]['visual_feature_windows'][0].shape[0]
        visual_features = torch.zeros(batch_size, max_windows, feature_dim, dtype=torch.float32)
    else:
        num_frames = batch[0]['video_windows'][0].shape[0]
        channels = batch[0]['video_windows'][0].shape[1]
        height = batch[0]['video_windows'][0].shape[2]
        width = batch[0]['video_windows'][0].shape[3]
        video = torch.zeros(batch_size, max_windows, num_frames, channels, height, width, dtype=torch.float32)
        frame_mask = torch.zeros(batch_size, max_windows, num_frames, dtype=torch.bool)

    tactile_high = torch.zeros(batch_size, max_windows, max_tactile, 36, dtype=torch.float32)
    tactile_low = torch.zeros(batch_size, max_windows, max_tactile, 36, dtype=torch.float32)
    tactile_mask = torch.zeros(batch_size, max_windows, max_tactile, dtype=torch.bool)
    window_mask = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    phase_labels = torch.full((batch_size, max_windows), -100, dtype=torch.long)
    stable_masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    stable_phases = torch.full((batch_size, max_windows), -1, dtype=torch.long)
    expert_forces = torch.full((batch_size, max_windows), float('nan'), dtype=torch.float32)
    control_force_targets = torch.full((batch_size, max_windows), float('nan'), dtype=torch.float32)
    reference_forces = torch.full((batch_size, max_windows), float('nan'), dtype=torch.float32)
    delta_force_targets = torch.full((batch_size, max_windows), float('nan'), dtype=torch.float32)
    has_expert = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    has_control_target = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    has_reference = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    reference_supervision_masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    delta_supervision_masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    quiet_supervision_masks = torch.zeros(batch_size, max_windows, dtype=torch.bool)
    object_index = torch.zeros(batch_size, dtype=torch.long)
    sample_index = torch.zeros(batch_size, dtype=torch.long)
    fragility_label = torch.zeros(batch_size, dtype=torch.long)
    geometry_label = torch.zeros(batch_size, dtype=torch.long)
    surface_label = torch.zeros(batch_size, dtype=torch.long)
    window_lengths = torch.zeros(batch_size, dtype=torch.long)
    sample_ids: list[str] = []
    object_ids: list[str] = []

    for batch_index, item in enumerate(batch):
        sample_ids.append(item['sample_id'])
        object_ids.append(item['object_id'])
        object_index[batch_index] = int(item['object_index'])
        sample_index[batch_index] = int(item['sample_index'])
        fragility_label[batch_index] = int(item['fragility_label'])
        geometry_label[batch_index] = int(item['geometry_label'])
        surface_label[batch_index] = int(item['surface_label'])
        num_windows_item = len(item['tactile_high_windows'])
        window_lengths[batch_index] = num_windows_item
        for window_index in range(num_windows_item):
            if use_visual_features:
                visual_features[batch_index, window_index] = item['visual_feature_windows'][window_index]
            else:
                frames = item['video_windows'][window_index]
                frames_valid = item['frame_masks'][window_index]
                video[batch_index, window_index, :frames.shape[0]] = frames
                frame_mask[batch_index, window_index, :frames_valid.shape[0]] = frames_valid
            high = item['tactile_high_windows'][window_index]
            low = item['tactile_low_windows'][window_index]
            tactile_valid = item['tactile_window_masks'][window_index]
            tactile_high[batch_index, window_index, :high.shape[0]] = high
            tactile_low[batch_index, window_index, :low.shape[0]] = low
            tactile_mask[batch_index, window_index, :tactile_valid.shape[0]] = tactile_valid
            window_mask[batch_index, window_index] = True
            phase_labels[batch_index, window_index] = int(item['phase_labels'][window_index])
            stable_masks[batch_index, window_index] = bool(item['stable_masks'][window_index])
            stable_phases[batch_index, window_index] = int(item['stable_phases'][window_index])
            expert_forces[batch_index, window_index] = float(item['expert_forces'][window_index])
            control_force_targets[batch_index, window_index] = float(item['control_force_targets'][window_index])
            reference_forces[batch_index, window_index] = float(item['reference_forces'][window_index])
            delta_force_targets[batch_index, window_index] = float(item['delta_force_targets'][window_index])
            has_expert[batch_index, window_index] = bool(item['has_expert'][window_index])
            has_control_target[batch_index, window_index] = bool(item['has_control_target'][window_index])
            has_reference[batch_index, window_index] = bool(item['has_reference'][window_index])
            reference_supervision_masks[batch_index, window_index] = bool(item['reference_supervision_masks'][window_index])
            delta_supervision_masks[batch_index, window_index] = bool(item['delta_supervision_masks'][window_index])
            quiet_supervision_masks[batch_index, window_index] = bool(item['quiet_supervision_masks'][window_index])

    return {
        'sample_ids': sample_ids,
        'object_ids': object_ids,
        'object_index': object_index,
        'sample_index': sample_index,
        'visual_features': visual_features,
        'video': video,
        'frame_mask': frame_mask,
        'tactile_high': tactile_high,
        'tactile_low': tactile_low,
        'tactile_mask': tactile_mask,
        'window_mask': window_mask,
        'window_lengths': window_lengths,
        'phase_labels': phase_labels,
        'stable_masks': stable_masks,
        'stable_phases': stable_phases,
        'expert_forces': expert_forces,
        'control_force_targets': control_force_targets,
        'reference_forces': reference_forces,
        'delta_force_targets': delta_force_targets,
        'has_expert': has_expert,
        'has_control_target': has_control_target,
        'has_reference': has_reference,
        'reference_supervision_masks': reference_supervision_masks,
        'delta_supervision_masks': delta_supervision_masks,
        'quiet_supervision_masks': quiet_supervision_masks,
        'fragility_label': fragility_label,
        'geometry_label': geometry_label,
        'surface_label': surface_label,
    }
