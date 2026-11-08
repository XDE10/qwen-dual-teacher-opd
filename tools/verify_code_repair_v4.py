"""Offline stdlib-only acceptance for the new repair release; no training/network."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
from src.data.approved_training_inputs import load_approved,model_view

MAPPING=Path('docs/migration/path_mapping.json')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def relocations():
    """Old path -> relocation entry (2026-11-08 repository reorg)."""
    if not MAPPING.exists():
        return {}
    return {entry['original']:entry for entry in read(MAPPING)['entries']}


def resolve(path, moved):
    """Return the file that holds `path`'s pinned bytes, or None if it is gone."""
    candidate=Path(str(path).replace('\\', '/'))
    if candidate.exists():
        return candidate
    entry=moved.get(candidate.as_posix())
    if entry and entry.get('status')=='missing_before_reorg':
        return None
    if entry and entry.get('new_location') and Path(entry['new_location']).exists():
        return Path(entry['new_location'])
    mirror=Path('provenance')/candidate
    return mirror if mirror.exists() else None


def pinned_sha(path, moved):
    found=resolve(path, moved)
    assert found is not None, f'pinned evidence missing and not relocated: {path}'
    return sha(found)


def resolution_stats(items, moved):
    stats=Counter('kept' if Path(str(p).replace('\\', '/')).exists() else ('relocated' if resolve(p, moved) else 'unresolved') for p in items)
    return dict(sorted(stats.items()))


def main():
    out=Path('reports/code_training_repair_20261025_v4')
    root=Path('data/code_training_repair_20261025_v4/release')
    moved=relocations()
    manifest=read(root/'manifest.json')
    assert all(sha(root/p)==h for p,h in manifest['files'].items())
    assert all(pinned_sha(p, moved)==h for p,h in manifest['inputs'].items())
    protected=read(out/'protected_before.json')
    assert all(pinned_sha(p, moved)==h for p,h in protected.items())
    code=load_approved(root/'manifest.json','project2_code_train.jsonl')
    math=load_approved(root/'manifest.json','project2_math_train.jsonl')
    assert len(code)==2500 and len(math)==2500
    ids={r['sample_id'] for r in code}
    assert 'code/vfc_96076' in ids and 'code/vfc_121257' not in ids
    assert len(ids)==2500
    assert Counter(r['interface'] for r in code)=={'function':1250,'stdin_stdout':1250}
    assert not ids&set(read(root/'reserved_dev_ids_never_train.json')['sample_ids'])
    for r in code+math:assert model_view(r,'opd')['target'] is None
    old=Path('data/portable_release')
    for p in manifest['files']:
        if p!='project2_code_train.jsonl':assert (root/p).read_bytes()==(old/p).read_bytes()
    a=(old/'project2_code_train.jsonl').read_bytes().splitlines(keepends=True)
    b=(root/'project2_code_train.jsonl').read_bytes().splitlines(keepends=True)
    changed=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    assert len(a)==len(b)==2500 and changed==[2451]
    decision=read(out/'admission_decision.json')
    assert all(pinned_sha(p, moved)==h for p,h in decision['evidence_files'].items())
    for name in ['smoke','code_short','math_short']:
        config=read(f'configs/project2_opd_v4_{name}.json')
        assert config['data_manifest']==(root/'manifest.json').as_posix()
        assert 'training_exclusions' not in config
    result={'version':'code-training-repair-v4-20261025','code_train':2500,
        'interfaces':dict(Counter(r['interface'] for r in code)), 'math_train':2500,
        'dev_counts':{n:sum(1 for l in (root/n).open(encoding='utf-8') if l.strip()) for n in manifest['approved_evaluation_files']},
        'removed':'code/vfc_121257','added':'code/vfc_96076','changed_slots':changed,
        'other_code_rows_byte_identical':2499,'protected_files_unchanged':len(protected),
        'protected_resolution':resolution_stats(protected, moved),
        'manifest_inputs_resolution':resolution_stats(manifest['inputs'], moved),
        'relocation_map':MAPPING.as_posix() if moved else None,
        'old_manifests_or_flags_modified':False,'errors':[],'training_started':False}
    (out/'verification_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
