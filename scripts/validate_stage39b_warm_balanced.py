from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cmg.config import (  # noqa: E402
    resolve_training_configs,
    sha256_file,
    sync_tactile_model_config,
    validate_direction_training_contract,
    validate_initialization_source_contract,
)
from cmg.models import CrossMediumSystem  # noqa: E402
from cmg.training import (  # noqa: E402
    audit_model_parameters,
    audit_optimizer_groups,
    build_optimizer,
    evaluate_w2a_retention_guard,
    freeze_modules,
    load_model_weights,
)


STAGE_PATH = Path('configs/stages/stage39b_v2_warm_balanced_medium.yaml')
INIT_PATH = Path('runs/stage39a_direction_adapter_warmup/checkpoints/best.pt')


def _matches_prefix(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + '.')


def validate(project_root: Path) -> dict[str, Any]:
    stage_path = project_root / STAGE_PATH
    init_path = project_root / INIT_PATH
    if not init_path.exists():
        raise FileNotFoundError(init_path)

    data, model_config, train, stage, sources = resolve_training_configs(project_root, stage_path)
    model_config = sync_tactile_model_config(data, model_config)
    contract = validate_direction_training_contract(
        project_root,
        data,
        model_config,
        train,
        stage,
        initialization_path=init_path,
    )
    if train['loss_reduction']['mode'] != 'sample_direction_class_macro_blend':
        raise RuntimeError('Warm+Balanced stage did not resolve the balanced loss mode.')
    if stage['selection_metric'] != 'medium_water_interface_hmean_macro_direction':
        raise RuntimeError('Warm+Balanced stage did not resolve the collapse-resistant selection metric.')

    model = CrossMediumSystem(model_config)
    initialization = stage['initialization']
    load_info = load_model_weights(
        model,
        init_path,
        strict=True,
        allow_lora_injection=bool(initialization.get('allow_lora_injection', True)),
        allowed_missing_prefixes=initialization.get('allowed_missing_prefixes', []),
    )
    validate_initialization_source_contract(stage, load_info)
    if load_info['missing_keys'] or load_info['unexpected_keys']:
        raise RuntimeError('Stage39a checkpoint did not load exactly into the Warm+Balanced model.')

    freeze_report = freeze_modules(
        model,
        stage['freeze_modules'],
        strict=True,
        lock_eval=True,
        train_mode_module_names=stage.get('frozen_train_mode_modules', []),
    )
    parameter_audit = audit_model_parameters(model)
    trainable = parameter_audit['trainable_parameter_names']
    expected = stage['expected_trainable_prefixes']
    unexpected_trainable = [
        name for name in trainable if not any(_matches_prefix(name, prefix) for prefix in expected)
    ]
    missing_expected = [
        prefix for prefix in expected if not any(_matches_prefix(name, prefix) for name in trainable)
    ]
    if unexpected_trainable or missing_expected:
        raise RuntimeError(
            f'Trainable parameter contract failed: unexpected={unexpected_trainable}, missing={missing_expected}.'
        )
    optimizer = build_optimizer(model, train)
    optimizer_groups = audit_optimizer_groups(model, optimizer)

    guard = stage['w2a_retention_guard']
    guard_result = evaluate_w2a_retention_guard(
        {
            guard['metric']: float(guard['baseline_value']),
            'medium_window_count_w2a': 1,
        },
        guard,
    )
    if not guard_result['passed']:
        raise RuntimeError('W2A retention guard rejected its own E0 baseline.')

    return {
        'phase': 'Stage39b-v2 Warm+Balanced preflight',
        'status': 'passed',
        'stage_name': stage['name'],
        'stage_path': str(stage_path.resolve()),
        'stage_sha256': sources['stage']['sha256'],
        'stage_effective_sha256': sources['stage']['effective_sha256'],
        'initialization_path': str(init_path.resolve()),
        'initialization_sha256': sha256_file(init_path),
        'initialization_stage': load_info['stage_name'],
        'loaded_tensor_count': load_info['loaded_tensor_count'],
        'target_tensor_count': load_info['target_tensor_count'],
        'missing_keys': load_info['missing_keys'],
        'unexpected_keys': load_info['unexpected_keys'],
        'training_contract': contract,
        'loss_reduction': train['loss_reduction'],
        'selection_metric': stage['selection_metric'],
        'w2a_guard': guard_result,
        'freeze_module_count': len(freeze_report['resolved_module_names']),
        'trainable_tensor_count': parameter_audit['trainable_tensor_count'],
        'trainable_value_count': parameter_audit['trainable_value_count'],
        'expected_trainable_prefixes': expected,
        'unexpected_trainable': unexpected_trainable,
        'missing_expected_trainable_prefixes': missing_expected,
        'optimizer_groups': optimizer_groups,
        'run_dir': str((project_root / 'runs' / stage['name']).resolve()),
        'run_dir_exists_before_launch': (project_root / 'runs' / stage['name']).exists(),
        'evaluation_acceptance': stage['evaluation_acceptance'],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument(
        '--output',
        default='data/processed/stats/stage39b_v2_warm_balanced_preflight.json',
    )
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    payload = validate(project_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
