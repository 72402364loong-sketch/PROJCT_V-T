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

from cmg.config import load_yaml, resolve_training_configs, sha256_file, validate_direction_training_contract
from cmg.training import evaluate_w2a_retention_guard


STAGES = (
    'stage39a_direction_adapter_warmup.yaml',
    'stage39b_bidirectional_medium.yaml',
    'stage39c_bidirectional_policy.yaml',
    'stage39d_bidirectional_joint.yaml',
)
POLICY_BASELINE = 7.069268751150515
MEDIUM_BASELINE = 0.6551664654131447


def validate(project_root: Path) -> dict[str, Any]:
    summary_path = project_root / 'evals/stage39e0_warmstart_baseline/e0_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    summary_sha256 = sha256_file(summary_path)
    if summary['status'] != 'passed' or summary['test_evaluated']:
        raise RuntimeError('M5-0 E0 evidence is not a passed Val-only evaluation.')

    base = load_yaml(project_root / 'configs/stages/stage39_bidirectional_base.yaml')
    if base.get('training_readiness', {}).get('status') != 'blocked_pending_e0_w2a_baseline':
        raise RuntimeError('Stage39 base safety template must remain blocked.')

    stage_reports: dict[str, Any] = {}
    canonical = project_root / 'runs/stage38j_f20_v3_causal/checkpoints/best.pt'
    for stage_file in STAGES:
        stage_path = project_root / 'configs/stages' / stage_file
        data, model, train, stage, sources = resolve_training_configs(project_root, stage_path)
        initialization_path = canonical if stage_file.startswith('stage39a_') else project_root / 'placeholder_previous_stage.pt'
        preflight = validate_direction_training_contract(
            project_root,
            data,
            model,
            train,
            stage,
            initialization_path=initialization_path,
        )
        readiness = stage['training_readiness']
        if readiness.get('evidence_sha256') != summary_sha256:
            raise RuntimeError(f'{stage_file} does not reference the current E0 summary SHA256.')
        guard = stage['w2a_retention_guard']
        expected_baseline = MEDIUM_BASELINE if stage_file.startswith('stage39b_') else POLICY_BASELINE
        if abs(float(guard['baseline_value']) - expected_baseline) > 1e-12:
            raise RuntimeError(f'{stage_file} has an unexpected E0 baseline.')
        count_metric = (
            'medium_window_count_w2a'
            if guard['metric'] == 'medium_f1_interface_w2a'
            else 'finger_control_interface_count_w2a'
        )
        guard_result = evaluate_w2a_retention_guard(
            {guard['metric']: float(guard['baseline_value']), count_metric: 1},
            guard,
        )
        if not guard_result['passed']:
            raise RuntimeError(f'{stage_file} guard rejects its own E0 baseline.')
        stage_reports[stage['name']] = {
            'stage_config_sha256': sources['stage']['sha256'],
            'stage_config_effective_sha256': sources['stage']['effective_sha256'],
            'readiness': preflight,
            'evidence_path': readiness['evidence'],
            'evidence_sha256': readiness['evidence_sha256'],
            'guard': guard_result,
        }

    return {
        'phase': 'Model Phase M5 post-E0 training unlock',
        'status': 'passed',
        'e0_summary_path': str(summary_path.resolve()),
        'e0_summary_sha256': summary_sha256,
        'base_template_status': base['training_readiness']['status'],
        'formal_stage_count': len(stage_reports),
        'ready_stage_count': sum(
            report['readiness']['readiness_status'] == 'ready' for report in stage_reports.values()
        ),
        'stage_reports': stage_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument(
        '--output',
        default='data/processed/stats/model_phase_m5_training_unlock_validation.json',
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

