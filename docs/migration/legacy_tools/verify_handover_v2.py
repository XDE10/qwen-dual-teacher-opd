"""Verify immutable releases and disclose source deltas without refreshing SHA."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/handover_20261024_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    protected = json.loads((REPORT / 'protected_before.json').read_text(encoding='utf-8'))
    changed = [p for p, digest in protected.items() if not (ROOT / p).is_file() or sha(ROOT / p) != digest]
    migration = json.loads((ROOT / 'migration_manifest.json').read_text(encoding='utf-8'))
    deltas = [r['path'] for r in migration['files'] if (ROOT / r['path']).is_file() and sha(ROOT / r['path']) != r['sha256']]
    allowed = {'README.md', 'scripts/train_project2_opd.py', 'src/data/code_sandbox.py'}
    unknown = [p for p in deltas if p.replace('\\', '/') not in allowed]
    report = {'schema': 'handover-v2-immutable-check', 'protected_files': len(protected),
        'protected_changes': changed, 'intentional_tracked_source_deltas': deltas,
        'unexpected_source_deltas': unknown, 'published_manifests_updated': False,
        'reference_py_unchanged': sha(ROOT / 'src/losses/reference.py') == protected['src\\losses\\reference.py'] if 'src\\losses\\reference.py' in protected else sha(ROOT / 'src/losses/reference.py') == protected['src/losses/reference.py']}
    (REPORT / 'immutable_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if changed or unknown:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
