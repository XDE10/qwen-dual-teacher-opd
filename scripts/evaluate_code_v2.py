"""Internal dev entry: explicit comparator registry, Docker-only execution."""
import argparse
import json
from pathlib import Path
from src.evaluation.code_v2 import evaluate, provisional_registry
from src.training.runtime_v2 import sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/portable_release/project2_code_dev.jsonl')
    p.add_argument('--output-dir', default='reports/handover_20261024_v2/code_evaluation')
    p.add_argument('--mode', choices=['registry', 'reference-selfcheck', 'student'], required=True)
    p.add_argument('--candidates')
    p.add_argument('--registry')
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(a.data).read_text(encoding='utf-8').splitlines() if l.strip()]
    registry = json.loads(Path(a.registry).read_text(encoding='utf-8')) if a.registry else provisional_registry(rows)
    reviewed_v3 = any(s.get('protocol_version') == 'review-v3' for s in registry.values())
    if reviewed_v3:
        from src.evaluation.reviewed_registry_v3 import validate_registry
        validate_registry(rows, registry, sha(a.data))
    if a.mode == 'registry':
        write(out / 'comparator_registry_unreviewed.json', registry)
        return
    if a.mode == 'reference-selfcheck':
        candidates = {r['sample_id']: r['reference_code'] for r in rows}
        if not reviewed_v3:
            registry = {k: {**v, 'status': 'reference_selfcheck_only'} for k, v in registry.items()}
    else:
        if not a.registry or not a.candidates:
            raise ValueError('Student scoring needs reviewed registry and candidates')
        candidate_rows = [json.loads(l) for l in Path(a.candidates).read_text(encoding='utf-8').splitlines() if l.strip()]
        candidates = {r['sample_id']: r['text'] for r in candidate_rows}
        if len(candidates) != len(candidate_rows) or set(candidates) != {r['sample_id'] for r in rows}:
            raise ValueError('Candidate IDs differ from sealed dev')
        if any(v.get('status') not in ({'reviewed','blocked_after_review'} if reviewed_v3 else {'reviewed'}) for v in registry.values()):
            raise ValueError('Student comparator registry requires semantic review')
    try:
        result = evaluate(rows, candidates, registry)
    except RuntimeError as error:
        result = {'status': '未验证', 'category': 'sandbox_infrastructure_unavailable', 'error': str(error)}
    result.update(mode=a.mode, data_sha256=sha(a.data),
                  worker_sha256=sha('scripts/code_eval_worker_v3.py' if reviewed_v3 else 'scripts/code_eval_worker_v2.py'),
                  source_compatibility_is_not_semantic_gold=True)
    write(out / (a.mode + '.json'), result)
    from collections import Counter
    print(json.dumps({'mode': a.mode, 'status': result.get('status'),
                      'counts': dict(Counter(r['status'] for r in result.get('results', [])))}, ensure_ascii=False))


if __name__ == '__main__':
    main()
