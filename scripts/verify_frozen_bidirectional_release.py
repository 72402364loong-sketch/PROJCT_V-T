from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', default='.')
    parser.add_argument(
        '--release-manifest',
        default='data/processed/policy_20hz_bidirectional_v1_fixed_test/stats/release_manifest_bidirectional_v1.json',
    )
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest_path = Path(args.release_manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    checksum_path = manifest_path.with_suffix('.sha256')
    with manifest_path.open('r', encoding='utf-8') as handle:
        release = json.load(handle)

    issues: list[str] = []
    expected_manifest_sha = checksum_path.read_text(encoding='utf-8').split()[0]
    actual_manifest_sha = file_sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        issues.append('release manifest checksum mismatch')

    checked_files = 0
    for group_name in ['artifacts', 'source_code', 'validation_reports']:
        for name, record in release[group_name].items():
            path = root / record['path']
            if not path.exists():
                issues.append(f'{group_name}.{name}: missing {path}')
                continue
            actual = file_sha256(path)
            checked_files += 1
            if actual != record['sha256']:
                issues.append(f'{group_name}.{name}: SHA-256 mismatch')

    result = {
        'release_id': release['release_id'],
        'release_status': release['release_status'],
        'release_manifest_sha256': actual_manifest_sha,
        'checked_file_count': checked_files,
        'error_count': len(issues),
        'issues': issues,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
