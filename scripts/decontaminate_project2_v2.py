"""No deletions, no approval edits; thresholds frozen before scanning train rows."""
import argparse
import collections
import json
from pathlib import Path
import sys

from src.data.decontamination_v2 import Index, calibrate
from src.training.runtime_v2 import sha, write


def rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def mbpp_text(prompt):
    # Official MBPP+ wraps the problem in a docstring plus demonstration assert.
    return prompt.strip().strip('"').split('\nassert ')[0].strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='reports/decontamination_20261024.json')
    p.add_argument('--report-dir', default='reports/handover_20261024_v2')
    a = p.parse_args()
    report_dir = Path(a.report_dir)
    release = Path('data/portable_release')
    mbpp_plus = rows('data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl')
    train_path = report_dir / 'mbpp_train_current.jsonl'
    known, positives, negatives = [], [], []
    if train_path.exists():
        mbpp_train = rows(train_path)
        by_id = {int(str(r['task_id']).split('/')[-1]): r for r in mbpp_plus}
        for r in mbpp_train:
            if r['task_id'] in by_id:
                other = by_id[r['task_id']]
                known.append({'train_id': r['task_id'], 'plus_id': other['task_id']})
                positives.append((r['text'], mbpp_text(other['prompt'])))
        for r in mbpp_train[:min(108, len(mbpp_train))]:
            unrelated = next(x for x in mbpp_plus if int(x['task_id'].split('/')[-1]) != r['task_id'])
            negatives.append((r['text'], mbpp_text(unrelated['prompt'])))
    if positives:
        calibration = calibrate(positives, negatives)
    else:
        calibration = {'threshold': .8, 'status': '未验证: MBPP train source unavailable; provisional threshold only', 'positive_recall': None}
    write(report_dir / 'decontamination_calibration_before_scan.json', calibration)
    threshold = calibration['threshold']
    # Identity-only positive pairs can give threshold=1: use a separate, predeclared
    # sensitivity tier .8, rather than pretend exact recall validates near duplicates.
    sensitivity_threshold = .8
    sources = {
        'math_train': release / 'project2_math_train.jsonl',
        'code_train': release / 'project2_code_train.jsonl',
        'math_dev': release / 'project2_math_dev.jsonl', 'code_dev': release / 'project2_code_dev.jsonl',
        'math500': Path('data/math/math500_test.jsonl'), 'gsm8k': Path('data/math/gsm8k_test.jsonl'),
        'humaneval_plus': Path('data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl'),
        'mbpp_plus': Path('data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl')}
    findings = []
    for domain, target_names in [('math', ['math500', 'gsm8k', 'math_dev']),
                                  ('code', ['humaneval_plus', 'mbpp_plus', 'code_dev'])]:
        for target in target_names:
            target_rows = rows(sources[target])
            indexed = [{'id': str(r.get('sample_id', r.get('task_id'))),
                        'text': r.get('problem', r.get('prompt', r.get('question', '')))} for r in target_rows]
            index = Index(indexed)
            for r in rows(sources[domain + '_train']):
                for hit in index.query(r['problem'], min(threshold, sensitivity_threshold)):
                    findings.append({'train_id': r['sample_id'], 'domain': domain, 'interface': r.get('interface'),
                        'target_dataset': target, **hit,
                        'tier': ('calibrated' if calibration.get('target_reached') else 'calibration_failed_candidate') if hit['exact'] or max(hit['jaccard_5gram'], hit['numeric_template_5gram'], hit['containment_5gram']) >= threshold else 'sensitivity_0.8_unvalidated'})
    code_rows = rows(sources['code_train'])
    impacted = {f['train_id'] for f in findings if f['domain'] == 'code'}
    affected = collections.Counter(r['interface'] for r in code_rows if r['sample_id'] in impacted)
    sensitivity = [f for f in findings if f['exact'] or max(f['jaccard_5gram'], f['numeric_template_5gram'], f['containment_5gram']) >= sensitivity_threshold]
    sensitivity_ids = {f['train_id'] for f in sensitivity if f['domain'] == 'code'}
    sensitivity_affected = collections.Counter(r['interface'] for r in code_rows if r['sample_id'] in sensitivity_ids)
    result = {'schema': 'decontamination-v2', 'status': 'read-only candidate audit; no data removal',
        'sources': {k: {'path': v.as_posix(), 'sha256': sha(v), 'rows': len(rows(v))} for k, v in sources.items()},
        'calibration': calibration, 'sensitivity_threshold': sensitivity_threshold,
        'sensitivity_0_8': {'finding_pairs': len(sensitivity), 'candidate_pairs': [{'train_id': f['train_id'], 'target_dataset': f['target_dataset'], 'target_id': f['target_id']} for f in sensitivity],
                            'hypothetical_code_removals': dict(sensitivity_affected)},
        'mbpp_train_source': json.loads((report_dir / 'mbpp_train_source.json').read_text(encoding='utf-8')) if (report_dir / 'mbpp_train_source.json').exists() else {'status': '未验证'},
        'mbpp_train_plus_id_overlaps': known, 'known_108_reproduced': len(known) == 108 if positives else '未验证',
        'findings': findings,
        'embedding': {'status': '未验证 / 未执行', 'reason': 'No pinned local embedding model; task forbids downloading model weights. Lexical 5-gram/numeric template are not embeddings.'},
        'hypothetical_balance_impact': {k: {'current': 1250, 'candidate_hits': affected[k], 'remaining_if_all_removed': 1250 - affected[k], 'needed_replacements_to_restore_1250': affected[k]} for k in ['function', 'stdin_stdout']},
        'limitations': ['No semantic decontamination guarantee', 'Sensitivity tier recall on free paraphrases unverified', 'Numeric-template matches may be related exercises, not duplicates', 'No rows removed or approval/manifest changed']}
    write(a.output, result)
    write(report_dir / Path(a.output).name, result)
    print(json.dumps({'findings': len(findings), 'known_mbpp_overlaps': len(known), 'calibration': calibration, 'impact': result['hypothetical_balance_impact']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
