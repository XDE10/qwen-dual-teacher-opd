"""Offline check of the v5 candidate and, once approved, of the v5 release. Read-only.

  python tools/verify_project2_release_v5.py
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
from src.data.approved_training_inputs import load_approved, model_view  # noqa: E402
from src.data.release_v5 import PATHS, jsonl, raw_lines, read, sha  # noqa: E402


def main():
    base = Path(PATHS['out'])
    candidate = base / 'candidate'
    manifest = read(candidate / 'candidate_manifest.json')
    assert all(sha(candidate / n) == h for n, h in manifest['files'].items()), 'candidate file digest'
    assert sha(manifest['audit_report']) == manifest['audit_report_sha256'], 'audit report digest'
    changes = read(candidate / 'changes.json')
    parent = Path(changes['parent_manifest'])
    assert sha(parent) == changes['parent_manifest_sha256'], 'parent v4 manifest changed'
    added = jsonl(candidate / 'added_math_train.jsonl')
    assert all(r['training_approved'] is False for r in added), 'candidate rows must not be approved'
    result = {'candidate': {'removed': len(changes['removed']), 'added': len(added)}, 'release': None, 'errors': []}
    release = base / 'release'
    if release.exists():
        m = read(release / 'manifest.json')
        assert all(sha(release / n) == h for n, h in m['files'].items()), 'release file digest'
        math = load_approved(release / 'manifest.json', 'project2_math_train.jsonl')
        code = load_approved(release / 'manifest.json', 'project2_code_train.jsonl')
        reserved = set(read(release / m['protected_dev_ids_file'])['sample_ids'])
        removed = set(changes['removed'])
        ids = [r['sample_id'] for r in math + code]
        assert len(ids) == len(set(ids)), 'duplicate IDs'
        assert not set(ids) & reserved and not set(ids) & removed, 'reserved or removed ID in training'
        for r in math + code:
            assert model_view(r, 'opd')['target'] is None
        old = parent.parent
        for name in ['project2_math_dev.jsonl', 'project2_code_dev.jsonl', m['protected_dev_ids_file']]:
            assert (release / name).read_bytes() == (old / name).read_bytes(), name + ' must be byte-identical'
        for name in ['project2_math_train.jsonl', 'project2_code_train.jsonl']:
            kept = [l for l in raw_lines(old / name) if json.loads(l)['sample_id'] not in removed]
            assert raw_lines(release / name)[:len(kept)] == kept, name + ' retained lines must be byte-identical'
        assert {r['sample_id'] for r in math} - {json.loads(l)['sample_id'] for l in raw_lines(old / 'project2_math_train.jsonl')} \
            == {r['sample_id'] for r in added}, 'added rows mismatch'
        answers = {r['sample_id'] for r in jsonl(release / 'project2_math_train_answers.jsonl')}
        assert answers == {r['sample_id'] for r in math}, 'answer sidecar must cover math train exactly'
        result['release'] = {'math_train': len(math), 'code_train': dict(Counter(r['interface'] for r in code)),
                             'approver': m['approval']['approver'], 'approved_at_utc': m['approval']['approved_at_utc']}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
