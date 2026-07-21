from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATHS = {
    'data': Path('configs/data/default.yaml'),
    'model': Path('configs/model/default.yaml'),
    'train': Path('configs/train/base.yaml'),
}

CONFIG_SELECTOR_KEYS = {
    'data': 'data_config',
    'model': 'model_config',
    'train': 'train_config',
}


def load_yaml(path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = ' -> '.join(str(item) for item in [*seen, config_path])
        raise ValueError(f'Circular YAML extends chain: {chain}')
    seen.add(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    parent_value = config.pop('extends', None)
    if parent_value is None:
        return config
    parent_path = Path(parent_value)
    if not parent_path.is_absolute():
        parent_path = config_path.parent / parent_path
    parent = load_yaml(parent_path, _seen=seen)
    return deep_update(parent, config)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_project_path(project_root: str | Path, raw_path: str | Path) -> Path:
    project_root = Path(project_root).resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def yaml_dependency_paths(path: str | Path, *, _seen: set[Path] | None = None) -> list[Path]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = ' -> '.join(str(item) for item in [*seen, config_path])
        raise ValueError(f'Circular YAML extends chain: {chain}')
    seen.add(config_path)
    with config_path.open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    parent_value = config.get('extends')
    dependencies: list[Path] = []
    if parent_value is not None:
        parent_path = Path(parent_value)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        dependencies.extend(yaml_dependency_paths(parent_path, _seen=seen))
    dependencies.append(config_path)
    return dependencies


def describe_yaml_source(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    config_path = Path(path).resolve()
    dependencies = yaml_dependency_paths(config_path)

    def describe(dependency: Path) -> dict[str, str]:
        try:
            project_path = dependency.relative_to(project_root).as_posix()
        except ValueError:
            project_path = str(dependency)
        return {
            'path': str(dependency),
            'project_path': project_path,
            'sha256': sha256_file(dependency),
        }

    dependency_records = [describe(dependency) for dependency in dependencies]
    effective_digest = hashlib.sha256()
    for record in dependency_records:
        effective_digest.update(record['project_path'].encode('utf-8'))
        effective_digest.update(record['sha256'].encode('ascii'))
    source = describe(config_path)
    source['effective_sha256'] = effective_digest.hexdigest()
    source['dependencies'] = dependency_records
    return source


def resolve_training_configs(
    project_root: str | Path,
    stage_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    project_root = Path(project_root).resolve()
    stage_path = resolve_project_path(project_root, stage_path)
    stage_config = load_yaml(stage_path)

    base_configs: dict[str, dict[str, Any]] = {}
    config_sources: dict[str, Any] = {
        'stage': describe_yaml_source(stage_path, project_root=project_root),
    }
    for kind, default_path in DEFAULT_CONFIG_PATHS.items():
        selector = CONFIG_SELECTOR_KEYS[kind]
        raw_path = stage_config.get(selector, default_path)
        if not isinstance(raw_path, (str, Path)) or not str(raw_path).strip():
            raise ValueError(f'Stage config field {selector!r} must be a non-empty path.')
        config_path = resolve_project_path(project_root, raw_path)
        base_configs[kind] = load_yaml(config_path)
        config_sources[kind] = describe_yaml_source(config_path, project_root=project_root)

    data_config = deep_update(base_configs['data'], stage_config.get('data', {}))
    model_config = deep_update(base_configs['model'], stage_config.get('model', {}))
    train_config = deep_update(base_configs['train'], stage_config.get('train', {}))
    if 'name' not in stage_config:
        raise ValueError(f'Stage config {stage_path} is missing required field "name".')
    train_config = deep_update(train_config, {'run_name': stage_config['name']})

    stage_split = stage_config.get('split')
    data_split = data_config.get('split_path')
    if stage_split and data_split:
        resolved_stage_split = resolve_project_path(project_root, stage_split)
        resolved_data_split = resolve_project_path(project_root, data_split)
        if resolved_stage_split != resolved_data_split:
            raise ValueError(
                'Stage split and data config split_path disagree: '
                f'{resolved_stage_split} != {resolved_data_split}.'
            )

    return data_config, model_config, train_config, stage_config, config_sources


def sync_tactile_model_config(data_config: dict[str, Any], model_config: dict[str, Any]) -> dict[str, Any]:
    from cmg.data.tactile import normalize_tactile_input_axes, tactile_input_dim_for_axes

    axes = normalize_tactile_input_axes(data_config.get('tactile_input_axes'))
    input_dim = tactile_input_dim_for_axes(axes)
    synced = dict(model_config)
    tactile_config = dict(synced.get('tactile', {}))
    tactile_config['input_dim'] = input_dim
    tactile_config['num_taxels'] = 12
    tactile_config['axis_dim'] = max(1, input_dim // 12)
    synced['tactile'] = tactile_config
    return synced


def validate_direction_training_contract(
    project_root: str | Path,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    train_config: dict[str, Any],
    stage_config: dict[str, Any],
    *,
    initialization_path: str | Path | None,
    resuming: bool = False,
    allow_pending_e0: bool = False,
) -> dict[str, Any]:
    direction_config = model_config.get('direction_conditioning', {})
    if not isinstance(direction_config, dict) or not bool(direction_config.get('enabled', False)):
        return {'direction_conditioning_enabled': False, 'validated': False}
    if not bool(direction_config.get('require_explicit_direction', False)):
        raise ValueError('Formal direction-conditioned training requires explicit direction ids.')
    if str(train_config.get('sampling_mode', '')).strip() != 'direction_object_aware':
        raise ValueError('Direction-conditioned training requires sampling_mode=direction_object_aware.')
    if not bool(train_config.get('use_direction_aware_sampler', False)):
        raise ValueError('Direction-conditioned training requires use_direction_aware_sampler=true.')
    if list(train_config.get('direction_values', [])) != ['W2A', 'A2W']:
        raise ValueError('Direction-conditioned training requires canonical direction_values=[W2A, A2W].')
    batch_size = int(train_config.get('batch_size', 0))
    if batch_size <= 0 or batch_size % 2 != 0:
        raise ValueError('Direction-conditioned training requires a positive even batch_size.')
    reduction = train_config.get('loss_reduction', {})
    supported_reductions = {'sample_direction_macro', 'sample_direction_class_macro_blend'}
    if not isinstance(reduction, dict) or reduction.get('mode') not in supported_reductions:
        raise ValueError(
            'Direction-conditioned training requires sample_direction_macro or '
            'sample_direction_class_macro_blend loss reduction.'
        )
    if reduction.get('mode') == 'sample_direction_class_macro_blend':
        original_weight = float(reduction.get('original_direction_macro_weight', 0.5))
        balanced_weight = float(reduction.get('direction_class_macro_weight', 0.5))
        if original_weight < 0.0 or balanced_weight < 0.0 or original_weight + balanced_weight <= 0.0:
            raise ValueError('Warm+Balanced Medium reduction weights must be non-negative with a positive sum.')
    selection_metric = str(stage_config.get('selection_metric', '')).strip()
    if not selection_metric.endswith('_macro_direction'):
        raise ValueError('Direction-conditioned stage selection_metric must be an explicit *_macro_direction metric.')

    initialization = stage_config.get('initialization', {})
    if not isinstance(initialization, dict):
        raise ValueError('Direction-conditioned stage initialization must be a mapping.')
    if bool(initialization.get('required', False)) and initialization_path is None and not resuming:
        raise ValueError('Direction-conditioned stage requires --init-from; cross-stage --resume is not allowed.')
    expected_source = initialization.get('expected_source')
    if expected_source and initialization_path is not None:
        if resolve_project_path(project_root, expected_source) != Path(initialization_path).resolve():
            raise ValueError(
                'Initialization checkpoint does not match stage expected_source: '
                f'{Path(initialization_path).resolve()} != {resolve_project_path(project_root, expected_source)}.'
            )

    readiness = stage_config.get('training_readiness', {})
    if not isinstance(readiness, dict):
        raise ValueError('Direction-conditioned stage training_readiness must be a mapping.')
    readiness_status = str(readiness.get('status', '')).strip()
    if readiness_status != 'ready' and not allow_pending_e0:
        raise RuntimeError(
            f'Direction-conditioned stage is not ready for formal training: {readiness_status or "missing status"}. '
            f'{readiness.get("reason", "Complete E0 and configure the W2A retention guard first.")}'
        )
    guard = stage_config.get('w2a_retention_guard', train_config.get('w2a_retention_guard', {}))
    if readiness_status == 'ready':
        if not isinstance(guard, dict) or not bool(guard.get('enabled', False)):
            raise ValueError('Ready direction-conditioned stages must enable w2a_retention_guard.')
        if 'baseline_value' not in guard:
            raise ValueError('Ready direction-conditioned stages must provide the E0 W2A baseline_value.')
    return {
        'direction_conditioning_enabled': True,
        'validated': True,
        'readiness_status': readiness_status,
        'selection_metric': selection_metric,
        'batch_size': batch_size,
        'initialization_required': bool(initialization.get('required', False)),
        'guard_enabled': bool(guard.get('enabled', False)) if isinstance(guard, dict) else False,
    }


def validate_initialization_source_contract(
    stage_config: dict[str, Any],
    initialization_info: dict[str, Any],
) -> None:
    initialization = stage_config.get('initialization', {})
    if not isinstance(initialization, dict):
        return
    expected_stage = initialization.get('expected_source_stage')
    if expected_stage and str(initialization_info.get('stage_name')) != str(expected_stage):
        raise ValueError(
            f'Initialization checkpoint stage {initialization_info.get("stage_name")!r} does not match '
            f'expected_source_stage={expected_stage!r}.'
        )
