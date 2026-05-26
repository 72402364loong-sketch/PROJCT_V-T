from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import deep_update, load_yaml
from cmg.constants import (
    FRAGILITY_TO_INDEX,
    FRAGILITY_TO_V2,
    FRAGILITY_V2_TO_INDEX,
    GEOMETRY_TO_INDEX,
    GEOMETRY_TO_SHAPE_PROFILE_V2,
    INDEX_TO_PHASE,
    PHASE_TO_INDEX,
    SHAPE_PROFILE_V2_TO_INDEX,
    SURFACE_TEXTURE_V2_TO_INDEX,
    SURFACE_TO_INDEX,
    SURFACE_TO_TEXTURE_V2,
)
from cmg.data.splits import resolve_sample_ids


ATTRIBUTES = {
    'fragility': FRAGILITY_TO_INDEX,
    'geometry': GEOMETRY_TO_INDEX,
    'surface': SURFACE_TO_INDEX,
}


def resolve_attribute_mappings(attribute_taxonomy: str) -> dict[str, dict[str, int]]:
    taxonomy = str(attribute_taxonomy or 'legacy').strip().lower()
    if taxonomy == 'coarse_v2':
        return {
            'fragility': FRAGILITY_V2_TO_INDEX,
            'geometry': SHAPE_PROFILE_V2_TO_INDEX,
            'surface': SURFACE_TEXTURE_V2_TO_INDEX,
        }
    return {
        'fragility': FRAGILITY_TO_INDEX,
        'geometry': GEOMETRY_TO_INDEX,
        'surface': SURFACE_TO_INDEX,
    }


def apply_attribute_taxonomy(samples: pd.DataFrame, attribute_taxonomy: str) -> pd.DataFrame:
    taxonomy = str(attribute_taxonomy or 'legacy').strip().lower()
    if taxonomy != 'coarse_v2':
        return samples.copy()
    updated = samples.copy()
    updated['fragility'] = updated['fragility'].astype(str).map(FRAGILITY_TO_V2)
    updated['geometry'] = updated['geometry'].astype(str).map(GEOMETRY_TO_SHAPE_PROFILE_V2)
    updated['surface'] = updated['surface'].astype(str).map(SURFACE_TO_TEXTURE_V2)
    if updated[['fragility', 'geometry', 'surface']].isna().any().any():
        raise RuntimeError('Failed to map at least one sample to coarse_v2 attribute taxonomy.')
    return updated


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path


def load_current_configs(project_root: Path, stage_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data_config = load_yaml(project_root / 'configs' / 'data' / 'default.yaml')
    train_config = load_yaml(project_root / 'configs' / 'train' / 'base.yaml')
    stage_config = load_yaml(stage_path)
    data_config = deep_update(data_config, stage_config.get('data', {}))
    train_config = deep_update(train_config, stage_config.get('train', {}))
    return data_config, train_config, stage_config


def normalize_allowed(values: Any) -> set[str] | None:
    if values is None:
        return None
    return {str(value).strip().lower() for value in values}


def apply_dataset_filters(
    samples: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    stage_config: dict[str, Any],
    subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    allowed = normalize_allowed(train_config.get('default_allowed_trial_results', {}).get(subset))
    if allowed is not None:
        samples = samples.loc[samples['trial_result'].astype(str).str.strip().str.lower().isin(allowed)].copy()
        windows = windows.loc[windows['trial_result'].astype(str).str.strip().str.lower().isin(allowed)].copy()

    min_valid_ratio_video = float(data_config.get('min_valid_ratio_video', 0.0))
    min_valid_ratio_tactile = float(data_config.get('min_valid_ratio_tactile', 0.0))
    if min_valid_ratio_video > 0.0 and 'valid_ratio_video' in windows.columns:
        windows = windows.loc[windows['valid_ratio_video'] >= min_valid_ratio_video].copy()
    if min_valid_ratio_tactile > 0.0 and 'valid_ratio_tactile' in windows.columns:
        windows = windows.loc[windows['valid_ratio_tactile'] >= min_valid_ratio_tactile].copy()

    tail_mode = str(stage_config.get('tail_mode', data_config.get('valid_tail_mode', 'all_valid')))
    if tail_mode == 'full_tail_only':
        windows = windows.loc[windows['tail_type'] == 'full_tail'].copy()
    elif tail_mode == 'short_tail_only':
        windows = windows.loc[windows['tail_type'] == 'short_tail'].copy()

    valid_sample_ids = set(windows['sample_id'].unique().tolist())
    samples = samples.loc[samples['sample_id'].isin(valid_sample_ids)].copy()
    windows = windows.loc[windows['sample_id'].isin(set(samples['sample_id'].tolist()))].copy()
    return samples, windows


def load_subset_frames(
    project_root: Path,
    split_path: Path,
    data_config: dict[str, Any],
    train_config: dict[str, Any],
    stage_config: dict[str, Any],
    subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_samples = pd.read_csv(project_root / 'data' / 'processed' / 'samples.csv')
    all_windows = pd.read_csv(project_root / 'data' / 'processed' / 'windows.csv')
    sample_ids = set(resolve_sample_ids(all_samples, split_path, subset=subset))
    samples = all_samples.loc[all_samples['sample_id'].isin(sample_ids)].copy()
    windows = all_windows.loc[all_windows['sample_id'].isin(sample_ids)].copy()
    return apply_dataset_filters(
        samples,
        windows,
        data_config=data_config,
        train_config=train_config,
        stage_config=stage_config,
        subset=subset,
    )


def entropy_effective_classes(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return float(math.exp(entropy))


def jensen_shannon_divergence(left_counts: dict[str, int], right_counts: dict[str, int]) -> float:
    labels = sorted(set(left_counts) | set(right_counts))
    left_total = sum(left_counts.get(label, 0) for label in labels)
    right_total = sum(right_counts.get(label, 0) for label in labels)
    if left_total <= 0 or right_total <= 0:
        return 0.0
    left = [left_counts.get(label, 0) / left_total for label in labels]
    right = [right_counts.get(label, 0) / right_total for label in labels]
    middle = [(a + b) / 2.0 for a, b in zip(left, right)]

    def kl_divergence(values: list[float], reference: list[float]) -> float:
        total = 0.0
        for value, ref in zip(values, reference):
            if value > 0.0 and ref > 0.0:
                total += value * math.log(value / ref)
        return total

    return float(0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle))


def distribution_rows(
    samples_by_subset: dict[str, pd.DataFrame],
    windows_by_subset: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        windows = windows_by_subset[subset]
        window_attrs = windows.merge(samples[['sample_id', *ATTRIBUTES.keys()]], on='sample_id', how='left')
        for attr_name, mapping in ATTRIBUTES.items():
            for class_name in sorted(mapping, key=mapping.get):
                sample_count = int((samples[attr_name].astype(str) == class_name).sum())
                window_count = int((window_attrs[attr_name].astype(str) == class_name).sum())
                object_count = int(samples.loc[samples[attr_name].astype(str) == class_name, 'object_id'].nunique())
                rows.append(
                    {
                        'subset': subset,
                        'attribute': attr_name,
                        'class_name': class_name,
                        'sample_count': sample_count,
                        'window_count': window_count,
                        'object_count': object_count,
                        'sample_ratio': sample_count / max(1, len(samples)),
                        'window_ratio': window_count / max(1, len(window_attrs)),
                    }
                )
    return rows


def object_attribute_rows(
    samples_by_subset: dict[str, pd.DataFrame],
    windows_by_subset: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        windows = windows_by_subset[subset]
        window_counts = windows.groupby('object_id').size().to_dict()
        sample_counts = samples.groupby('object_id').size().to_dict()
        object_groups = samples.groupby('object_id', sort=True)
        for object_id, group in object_groups:
            first = group.iloc[0]
            rows.append(
                {
                    'subset': subset,
                    'object_id': object_id,
                    'object_name': first.get('object_name', ''),
                    'object_alias': first.get('object_alias', ''),
                    'fragility': first.get('fragility', ''),
                    'geometry': first.get('geometry', ''),
                    'surface': first.get('surface', ''),
                    'sample_count': int(sample_counts.get(object_id, 0)),
                    'window_count': int(window_counts.get(object_id, 0)),
                }
            )
    return rows


def combination_rows(
    samples_by_subset: dict[str, pd.DataFrame],
    windows_by_subset: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attrs = list(ATTRIBUTES.keys())
    for subset, samples in samples_by_subset.items():
        windows = windows_by_subset[subset]
        window_counts = windows.groupby('sample_id').size().to_dict()
        samples_with_windows = samples.copy()
        samples_with_windows['window_count'] = samples_with_windows['sample_id'].map(window_counts).fillna(0).astype(int)
        for combo, group in samples_with_windows.groupby(attrs, sort=True):
            combo_values = dict(zip(attrs, combo))
            rows.append(
                {
                    'subset': subset,
                    **combo_values,
                    'sample_count': int(len(group)),
                    'window_count': int(group['window_count'].sum()),
                    'object_count': int(group['object_id'].nunique()),
                    'object_ids': '|'.join(sorted(group['object_id'].unique().tolist())),
                }
            )
    return rows


def phase_cooccurrence_rows(
    samples_by_subset: dict[str, pd.DataFrame],
    windows_by_subset: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset, samples in samples_by_subset.items():
        windows = windows_by_subset[subset].merge(samples[['sample_id', *ATTRIBUTES.keys()]], on='sample_id', how='left')
        for attr_name, mapping in ATTRIBUTES.items():
            for class_name in sorted(mapping, key=mapping.get):
                class_windows = windows.loc[windows[attr_name].astype(str) == class_name]
                total = len(class_windows)
                phase_counts = class_windows['phase_label'].astype(str).value_counts().to_dict()
                row = {
                    'subset': subset,
                    'attribute': attr_name,
                    'class_name': class_name,
                    'total_windows': int(total),
                }
                for phase_name in sorted(PHASE_TO_INDEX, key=PHASE_TO_INDEX.get):
                    count = int(phase_counts.get(phase_name, 0))
                    row[f'{phase_name.lower()}_windows'] = count
                    row[f'{phase_name.lower()}_ratio'] = count / max(1, total)
                rows.append(row)
    return rows


def summarize_attribute_health(
    distribution: pd.DataFrame,
    object_attrs: pd.DataFrame,
    subset: str,
) -> dict[str, Any]:
    subset_dist = distribution.loc[distribution['subset'] == subset]
    summary: dict[str, Any] = {}
    for attr_name in ATTRIBUTES:
        attr_dist = subset_dist.loc[subset_dist['attribute'] == attr_name].copy()
        sample_counts = {str(row['class_name']): int(row['sample_count']) for _, row in attr_dist.iterrows()}
        window_counts = {str(row['class_name']): int(row['window_count']) for _, row in attr_dist.iterrows()}
        object_counts = {str(row['class_name']): int(row['object_count']) for _, row in attr_dist.iterrows()}
        nonzero = [count for count in sample_counts.values() if count > 0]
        object_nonzero = [count for count in object_counts.values() if count > 0]
        majority_samples = max(sample_counts.values()) if sample_counts else 0
        total_samples = sum(sample_counts.values())
        majority_windows = max(window_counts.values()) if window_counts else 0
        total_windows = sum(window_counts.values())
        summary[attr_name] = {
            'sample_counts': sample_counts,
            'window_counts': window_counts,
            'object_counts': object_counts,
            'majority_sample_accuracy_baseline': majority_samples / max(1, total_samples),
            'majority_window_accuracy_baseline': majority_windows / max(1, total_windows),
            'sample_effective_classes': entropy_effective_classes(list(sample_counts.values())),
            'window_effective_classes': entropy_effective_classes(list(window_counts.values())),
            'min_nonzero_sample_count': min(nonzero) if nonzero else 0,
            'min_nonzero_object_count': min(object_nonzero) if object_nonzero else 0,
            'missing_classes': [label for label, count in sample_counts.items() if count == 0],
            'low_object_classes': [label for label, count in object_counts.items() if 0 < count < 2],
        }

    object_subset = object_attrs.loc[object_attrs['subset'] == subset]
    duplicate_combo_count = int(
        object_subset.groupby(list(ATTRIBUTES.keys()))['object_id'].nunique().reset_index(name='n_objects').query('n_objects > 1').shape[0]
    )
    summary['object_binding'] = {
        'object_count': int(object_subset['object_id'].nunique()),
        'attribute_combo_count': int(object_subset[list(ATTRIBUTES.keys())].drop_duplicates().shape[0]),
        'duplicate_combo_count': duplicate_combo_count,
    }
    return summary


def build_recommendations(
    distribution: pd.DataFrame,
    object_attrs: pd.DataFrame,
    subsets: list[str],
) -> list[dict[str, Any]]:
    recommendations: list[dict[str, Any]] = []
    train_dist = distribution.loc[distribution['subset'] == 'train']
    for attr_name in ATTRIBUTES:
        attr_train = train_dist.loc[train_dist['attribute'] == attr_name]
        for _, row in attr_train.iterrows():
            class_name = str(row['class_name'])
            if int(row['sample_count']) == 0:
                recommendations.append(
                    {
                        'priority': 'high',
                        'attribute': attr_name,
                        'class_name': class_name,
                        'reason': 'Class is absent from train split.',
                        'suggestion': 'Do not report this class as a learnable supervised category under the current split.',
                    }
                )
            elif int(row['object_count']) < 2:
                recommendations.append(
                    {
                        'priority': 'high',
                        'attribute': attr_name,
                        'class_name': class_name,
                        'reason': 'Class is represented by fewer than two train objects.',
                        'suggestion': 'Merge this class or collect/add more train objects before using macro F1 as a core claim.',
                    }
                )
            elif float(row['sample_ratio']) < 0.10:
                recommendations.append(
                    {
                        'priority': 'medium',
                        'attribute': attr_name,
                        'class_name': class_name,
                        'reason': 'Class has less than 10% of train samples.',
                        'suggestion': 'Consider class weighting, resampling, or coarser labels.',
                    }
                )

        train_counts = {str(row['class_name']): int(row['sample_count']) for _, row in attr_train.iterrows()}
        for subset in subsets:
            if subset == 'train':
                continue
            other = distribution.loc[(distribution['subset'] == subset) & (distribution['attribute'] == attr_name)]
            other_counts = {str(row['class_name']): int(row['sample_count']) for _, row in other.iterrows()}
            jsd = jensen_shannon_divergence(train_counts, other_counts)
            if jsd > 0.10:
                recommendations.append(
                    {
                        'priority': 'medium',
                        'attribute': attr_name,
                        'class_name': None,
                        'reason': f'Train/{subset} sample distribution shift is high (JSD={jsd:.3f}).',
                        'suggestion': 'Use per-class reporting and be careful when interpreting object-level unseen generalization.',
                    }
                )

    for subset in subsets:
        subset_objects = object_attrs.loc[object_attrs['subset'] == subset]
        combo_counts = subset_objects.groupby(list(ATTRIBUTES.keys()))['object_id'].nunique()
        single_object_combos = combo_counts[combo_counts == 1]
        if len(single_object_combos) > 0:
            recommendations.append(
                {
                    'priority': 'medium',
                    'attribute': 'combined',
                    'class_name': None,
                    'reason': f'{subset} has {len(single_object_combos)} attribute combinations bound to a single object.',
                    'suggestion': 'Avoid over-claiming combined attribute accuracy unless ablations show object-invariant learning.',
                }
            )
    return recommendations


def confusion_prf_rows(confusion: list[list[int]], labels: list[str], task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        tp = int(confusion[index][index])
        fp = int(sum(confusion[row][index] for row in range(len(labels))) - tp)
        fn = int(sum(confusion[index][col] for col in range(len(labels))) - tp)
        support = tp + fn
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if (2 * tp + fp + fn) == 0 else (2 * tp) / (2 * tp + fp + fn)
        rows.append(
            {
                'task': task,
                'class_name': label,
                'support': support,
                'predicted_count': tp + fp,
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            }
        )
    return rows


def maybe_load_model_metrics(
    project_root: Path,
    stage: str,
    checkpoint: str | None,
    subset: str,
    *,
    eval_num_workers: int,
) -> dict[str, Any] | None:
    if checkpoint is None:
        return None
    from torch.utils.data import DataLoader

    from cmg.data import sequence_collate_fn
    from cmg.evaluation import prepare_evaluation_context, resolve_path as resolve_eval_path
    from cmg.training import run_model_epoch

    context = prepare_evaluation_context(
        project_root=project_root,
        stage=stage,
        checkpoint=checkpoint,
        subset=subset,
        only_stable=False,
    )
    loader = DataLoader(
        context['dataset'],
        batch_size=int(context['train_config']['batch_size']),
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=bool(context['train_config'].get('pin_memory', False)),
        collate_fn=sequence_collate_fn,
    )
    metrics = run_model_epoch(
        context['model'],
        loader,
        device=context['device'],
        training=False,
        stage_config=context['stage_config'],
        model_config=context['model_config'],
        phase_class_weights=context['phase_class_weights'],
        amp_enabled=bool(context['train_config'].get('amp_enabled', False)) and context['device'].type == 'cuda',
    )
    checkpoint_path = resolve_eval_path(project_root, checkpoint)
    return {
        'stage_name': context['stage_config'].get('name'),
        'stage_path': str(context['stage_path'].resolve()),
        'subset': subset,
        'only_stable': False,
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_run_config_path': context['load_info'].get('run_config_path'),
        'metrics': metrics,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    global ATTRIBUTES

    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--stage', required=True)
    parser.add_argument('--subset', default='all', choices=['train', 'val', 'test', 'all'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--eval-num-workers', type=int, default=0)
    parser.add_argument('--output-dir', default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    stage_path = resolve_path(project_root, args.stage)
    data_config, train_config, stage_config = load_current_configs(project_root, stage_path)
    attribute_taxonomy = str(data_config.get('attribute_taxonomy', 'legacy'))
    ATTRIBUTES = resolve_attribute_mappings(attribute_taxonomy)
    split_path = resolve_path(project_root, stage_config['split'])
    stage_name = str(stage_config['name'])
    subsets = ['train', 'val', 'test'] if args.subset == 'all' else [args.subset]

    samples_by_subset: dict[str, pd.DataFrame] = {}
    windows_by_subset: dict[str, pd.DataFrame] = {}
    for subset in subsets:
        samples, windows = load_subset_frames(
            project_root,
            split_path,
            data_config,
            train_config,
            stage_config,
            subset,
        )
        samples = apply_attribute_taxonomy(samples, attribute_taxonomy)
        samples_by_subset[subset] = samples
        windows_by_subset[subset] = windows

    if args.output_dir:
        output_dir = resolve_path(project_root, args.output_dir)
    else:
        output_dir = project_root / 'evals' / 'attribute_label_diagnostics' / stage_name / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = pd.DataFrame(distribution_rows(samples_by_subset, windows_by_subset))
    object_attrs = pd.DataFrame(object_attribute_rows(samples_by_subset, windows_by_subset))
    combinations = pd.DataFrame(combination_rows(samples_by_subset, windows_by_subset))
    phase_cooccurrence = pd.DataFrame(phase_cooccurrence_rows(samples_by_subset, windows_by_subset))

    distribution_path = output_dir / 'attribute_distribution_by_subset.csv'
    object_path = output_dir / 'object_attribute_map.csv'
    combo_path = output_dir / 'attribute_combination_summary.csv'
    phase_path = output_dir / 'attribute_phase_cooccurrence.csv'
    distribution.to_csv(distribution_path, index=False, encoding='utf-8-sig')
    object_attrs.to_csv(object_path, index=False, encoding='utf-8-sig')
    combinations.to_csv(combo_path, index=False, encoding='utf-8-sig')
    phase_cooccurrence.to_csv(phase_path, index=False, encoding='utf-8-sig')

    summary = {
        'stage_name': stage_name,
        'stage_path': str(stage_path),
        'split_path': str(split_path),
        'attribute_taxonomy': attribute_taxonomy,
        'subset': args.subset,
        'subsets': subsets,
        'sample_counts': {subset: int(len(samples_by_subset[subset])) for subset in subsets},
        'window_counts': {subset: int(len(windows_by_subset[subset])) for subset in subsets},
        'attribute_health': {
            subset: summarize_attribute_health(distribution, object_attrs, subset)
            for subset in subsets
        },
        'recommendations': build_recommendations(distribution, object_attrs, subsets),
        'outputs': {
            'distribution_path': str(distribution_path),
            'object_path': str(object_path),
            'combination_path': str(combo_path),
            'phase_cooccurrence_path': str(phase_path),
        },
    }

    if args.checkpoint is not None and args.subset != 'all':
        model_payload = maybe_load_model_metrics(
            project_root,
            str(stage_path),
            args.checkpoint,
            args.subset,
            eval_num_workers=int(args.eval_num_workers),
        )
        if model_payload is not None:
            metrics = model_payload['metrics']
            labels = {
                'medium': [INDEX_TO_PHASE[index] for index in sorted(INDEX_TO_PHASE)],
                'fragility': [label for label, _ in sorted(ATTRIBUTES['fragility'].items(), key=lambda item: item[1])],
                'geometry': [label for label, _ in sorted(ATTRIBUTES['geometry'].items(), key=lambda item: item[1])],
                'surface': [label for label, _ in sorted(ATTRIBUTES['surface'].items(), key=lambda item: item[1])],
            }
            prf_rows: list[dict[str, Any]] = []
            for task, key in [
                ('medium', 'medium_confusion'),
                ('fragility', 'fragility_confusion'),
                ('geometry', 'geometry_confusion'),
                ('surface', 'surface_confusion'),
            ]:
                if key in metrics:
                    prf_rows.extend(confusion_prf_rows(metrics[key], labels[task], task))
            model_metrics_path = output_dir / 'model_per_class_metrics.csv'
            pd.DataFrame(prf_rows).to_csv(model_metrics_path, index=False, encoding='utf-8-sig')
            summary['model_metrics'] = {
                'checkpoint': args.checkpoint,
                'subset': args.subset,
                'metrics': metrics,
                'per_class_metrics_path': str(model_metrics_path),
            }
            summary['outputs']['model_per_class_metrics_path'] = str(model_metrics_path)
    elif args.checkpoint is not None and args.subset == 'all':
        summary['checkpoint_warning'] = 'Checkpoint metrics are skipped when --subset all; pass --subset train/val/test to evaluate one split.'

    summary_path = output_dir / 'attribute_label_diagnostics_summary.json'
    write_json(summary_path, summary)
    print(
        {
            'output_dir': str(output_dir),
            'summary_path': str(summary_path),
            'distribution_path': str(distribution_path),
            'object_path': str(object_path),
            'combination_path': str(combo_path),
            'phase_cooccurrence_path': str(phase_path),
            'recommendation_count': len(summary['recommendations']),
        }
    )


if __name__ == '__main__':
    main()
