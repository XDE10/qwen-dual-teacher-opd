"""Project-2 data release v5: v4 minus overlap/budget failures, refilled from sealed pools.

The v4 release is never modified. The pipeline has two separate steps:

1. ``build_candidate``: deterministic audit and refill. It writes a candidate
   directory (changes, added rows with training_approved=false, answer sidecar)
   and an audit report. A candidate cannot be loaded for training.
2. ``approve_candidate``: an explicit, recorded owner decision. It materialises
   ``release/``. Retained v4 lines are byte-identical; only added rows change
   status; approval evidence is bound to the candidate digests.

Gates for every training row (retained or added):
* prompt budget under the frozen prompt contract (prompt + reserve <= 4096);
* decontamination_v5 against evaluation and dev sets (see that module);
* for added rows also: not reserved, not quarantined, not previously reviewed
  and rejected (reconstructed from the deterministic original P2 pool
  selection), and no internal near-duplicate (3-gram Jaccard 0.85, normalized
  exact) against the rest of the release.
Added MATH rows keep the removed rows' (source, level) mix. GSM8K keeps its source.
"""
import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.data.approved_training_inputs import load_approved
from src.data.decontamination_v5 import (LCS_RATIO, NGRAM, ProtectedIndex, ast_sha, audit,
                                         strip_code_boilerplate)
from src.data.prepare_math_data import SimilarityIndex

VERSION = 'project2-release-v5-20261108'
OFFICIAL_CODE = {
    'humanevalplus': {'path': 'data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl',
                      'sha256': '42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f', 'rows': 164},
    'mbppplus': {'path': 'data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl',
                 'sha256': 'b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad', 'rows': 378}}
PATHS = {
    'parent_manifest': 'data/code_training_repair_20261025_v4/release/manifest.json',
    'math_manifest': 'data/math_manifest.json',
    'tokenizer': 'data/tokenizers/Qwen2.5-1.5B-Instruct',
    'out': 'data/project2_release_v5_20261108',
    'report': 'reports/project2_release_v5_20261108',
}
MATH_POOLS = {'DigitalLearningGmbH/MATH-lighteval': 'math_train', 'openai/gsm8k': 'gsm8k_train'}
MATH_PROTECTED = {'math500_test': 'math500_test', 'gsm8k_test': 'gsm8k_test', 'math_test_5000': 'math_source_test',
                  'original_math_dev': 'math_dev'}
FILES = ['project2_math_train.jsonl', 'project2_code_train.jsonl', 'project2_math_dev.jsonl', 'project2_code_dev.jsonl']
ADDED_REVIEW = 'deterministic_local_gates_v5_no_ai_no_itemwise_human'
APPROVAL_METHOD = 'v5_deterministic_gates_plus_owner_batch_approval'
ANSWERS = 'project2_math_train_answers.jsonl'
EXCLUDED = 'excluded_training_ids_v5.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_lines(path):
    return Path(path).read_bytes().splitlines(keepends=True)


def dump_row(row):
    return (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')


def order_key(sample_id):
    return hashlib.sha256((VERSION + '/' + sample_id).encode()).hexdigest()


def code_inputs():
    """Digests of the audit code itself, relative to this repository."""
    here = Path(__file__).resolve()
    return {'src/data/' + name: sha(here.with_name(name)) for name in ('release_v5.py', 'decontamination_v5.py')}


def verified_math_files(root, manifest_path):
    """{name: path} for the sealed math files, each checked against math_manifest."""
    manifest = read(root / manifest_path)
    out = {}
    for name, meta in manifest['files'].items():
        path = root / meta['path']
        if sha(path) != meta['sha256']:
            raise ValueError('Sealed math file changed: ' + meta['path'])
        out[name] = path
    return out


def official_code(root):
    out = {}
    for name, meta in OFFICIAL_CODE.items():
        path = root / meta['path']
        if not path.exists():
            raise FileNotFoundError(f'{meta["path"]} missing: run python -m scripts.restore_official_benchmarks first')
        if sha(path) != meta['sha256']:
            raise ValueError('Official benchmark digest mismatch: ' + meta['path'])
        out[name] = jsonl(path)
    return out


def original_math_pool_ids(root):
    """IDs of the deterministic original P2 math pool (1500 MATH + 1000 GSM8K).

    Rows of this pool that are absent from the v4 release were not approved.

    ``build_math_pool`` reproduces the original selection exactly (all 2380
    surviving rows match field by field). Its other 120 rows were not approved
    in the delegated review. The reasons were lost in the reorg, so refills
    exclude them conservatively.
    """
    from src.data.screen_code_interfaces import build_math_pool
    picked, _ = build_math_pool(root)
    return {r['sample_id'] for r in picked}


def prompt_lengths(tokenizer_path, contract):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def length(problem, domain):
        system = contract[domain + '_system']
        messages = ([{'role': 'system', 'content': system}] if system is not None else [])
        messages.append({'role': 'user', 'content': problem})
        return len(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    return length


def budget_limit(contract, domain):
    return contract['context_limit'] - contract[domain + '_max_new_tokens']


def prompt_of(row, domain):
    return row['problem'] if domain == 'math' else row.get('prompt', row['problem'])


def math_protected_items(math_files, dev_rows, reserved, pool_by_id):
    items, seen = [], set()
    for set_name, file_key in MATH_PROTECTED.items():
        for r in jsonl(math_files[file_key]):
            if r['sample_id'] not in seen:
                items.append({'id': r['sample_id'], 'text': r['problem'], 'set': set_name})
                seen.add(r['sample_id'])
    for r in dev_rows:
        if r['sample_id'] not in seen:
            items.append({'id': r['sample_id'], 'text': r['problem'], 'set': 'math_dev_v4'})
            seen.add(r['sample_id'])
    for sid in sorted(reserved):
        if sid.startswith('math/') and sid not in seen and sid in pool_by_id:
            items.append({'id': sid, 'text': pool_by_id[sid]['problem'], 'set': 'reserved_math_dev_candidate'})
            seen.add(sid)
    return items


def code_protected_items(official, dev_rows):
    items = [{'id': r['task_id'], 'text': strip_code_boilerplate(r['prompt']), 'set': name}
             for name, rows in official.items() for r in rows]
    items += [{'id': r['sample_id'], 'text': strip_code_boilerplate(r['problem']), 'set': 'code_dev_v4'} for r in dev_rows]
    return items


def benchmark_ast_hashes(official, code_dev):
    hashes = {}
    for name, rows in official.items():
        for r in rows:
            for code in (r['canonical_solution'], r['prompt'] + r['canonical_solution']):
                try:
                    hashes[ast_sha(code)] = name + ':' + r['task_id']
                except (SyntaxError, ValueError):
                    pass
    for r in code_dev:
        if r.get('reference_ast_sha256'):
            hashes[r['reference_ast_sha256']] = 'code_dev_v4:' + r['sample_id']
    return hashes


def added_math_row(pool_row, prompt_tokens, limit):
    row = {k: pool_row[k] for k in ('sample_id', 'source', 'source_revision', 'source_split', 'source_id', 'problem',
                                    'problem_hash', 'family_id', 'raw_problem_sha256') if k in pool_row}
    if pool_row.get('level') is not None:
        row.update(level=pool_row['level'], problem_type=pool_row.get('problem_type'))
    row.update(domain='math', split='train', training_approved=False, review_method=ADDED_REVIEW,
               itemwise_human_review_performed=False, local_review_flags=[], local_admission_version=VERSION,
               prompt_token_count=prompt_tokens, v5_status='candidate_pending_owner_approval',
               admission_gates={'prompt_budget': f'{prompt_tokens}<={limit}',
                                'decontamination_v5': 'no_hit', 'internal_duplicate_jaccard_3gram_0.85': 'no_hit',
                                'not_reserved_or_quarantined_or_previously_rejected': True})
    return row


def stratum(row):
    return (row['source'], row.get('level') if row['source'] in ('DigitalLearningGmbH/MATH-lighteval',) else None)


def build_candidate(root='.', paths=None, length_fn=None, rejected_fn=None, math_target=None):
    """Deterministic audit + refill. Returns the audit report dict.

    ``length_fn(problem, domain)`` and ``rejected_fn(root)`` (returns the original
    pool IDs) are injectable for tests.
    ``math_target`` grows the math train set (extra rows follow the v4 source ratio).
    """
    from src.teachers.opd_contract import load_contract
    root = Path(root)
    paths = {**PATHS, **(paths or {})}
    contract = load_contract(root / 'configs/project2_prompt_contract.json')
    parent_path = root / paths['parent_manifest']
    parent = read(parent_path)
    release = parent_path.parent
    for name, digest in parent['files'].items():
        if sha(release / name) != (digest['sha256'] if isinstance(digest, dict) else digest):
            raise ValueError('Parent release file changed: ' + name)
    math_train = load_approved(parent_path, 'project2_math_train.jsonl')
    code_train = load_approved(parent_path, 'project2_code_train.jsonl')
    math_dev, code_dev = jsonl(release / 'project2_math_dev.jsonl'), jsonl(release / 'project2_code_dev.jsonl')
    reserved = set(read(release / parent['protected_dev_ids_file'])['sample_ids'])
    math_files = verified_math_files(root, paths['math_manifest'])
    pools = {source: jsonl(math_files[key]) for source, key in MATH_POOLS.items()}
    pool_by_id = {r['sample_id']: r for rows in pools.values() for r in rows}
    pool_by_id.update({r['sample_id']: r for r in jsonl(math_files['math_dev'])})
    quarantine = {r['sample_id'] for r in jsonl(math_files['math_quarantine'])}
    official = official_code(root)
    length_fn = length_fn or prompt_lengths(root / paths['tokenizer'], contract)
    rejected_before = (rejected_fn or original_math_pool_ids)(root) - {r['sample_id'] for r in math_train}

    # 1. Budgets for every current train/dev row.
    budget = {}
    for name, domain, rows in [('math_train', 'math', math_train), ('math_dev', 'math', math_dev),
                               ('code_train', 'code', code_train), ('code_dev', 'code', code_dev)]:
        lengths = {r['sample_id']: length_fn(prompt_of(r, domain), domain) for r in rows}
        over = sorted(s for s, n in lengths.items() if n > budget_limit(contract, domain))
        recorded = [r for r in rows if r.get('prompt_token_count') is not None]
        budget[name] = {'rows': len(rows), 'max_prompt_tokens': max(lengths.values()), 'limit': budget_limit(contract, domain),
                        'over_budget': over, 'rows_without_recorded_count': len(rows) - len(recorded),
                        'recorded_count_mismatches': sorted(r['sample_id'] for r in recorded
                                                            if r['prompt_token_count'] != lengths[r['sample_id']])}

    # 2. Overlap audit of retained rows.
    math_index = ProtectedIndex(math_protected_items(math_files, math_dev, reserved, pool_by_id))
    code_index = ProtectedIndex(code_protected_items(official, code_dev))
    math_flags = audit(math_train, math_index, lambda r: r['problem'])
    code_flags = audit(code_train, code_index, lambda r: strip_code_boilerplate(r['problem']))
    ast_hits = benchmark_ast_hashes(official, code_dev)
    for r in code_train:
        hit = ast_hits.get(r.get('reference_ast_sha256'))
        if hit:
            code_flags.setdefault(r['sample_id'], []).append({'rule': 'reference_ast_exact', 'protected_id': hit,
                                                              'protected_set': hit.split(':')[0]})
    removed = {}
    for domain, flags in [('math', math_flags), ('code', code_flags)]:
        for sid, hits in flags.items():
            removed[sid] = {'domain': domain, 'reason': 'decontamination_v5', 'evidence': hits}
    for name in ('math_train', 'code_train'):
        for sid in budget[name]['over_budget']:
            removed.setdefault(sid, {'domain': name.split('_')[0], 'reason': 'prompt_budget', 'evidence': []})

    # 3. Refill math by removed (source, level); optional growth by source ratio.
    kept_math = [r for r in math_train if r['sample_id'] not in removed]
    # Delegated v4 rows carry no level field; read (source, level) from the sealed pool row.
    need = Counter(stratum(pool_by_id[r['sample_id']]) for r in math_train if r['sample_id'] in removed)
    if math_target is not None:
        extra = math_target - len(math_train)
        if extra < 0:
            raise ValueError('math_target below current size')
        ratio = Counter(r['source'] for r in math_train)
        shares = {s: extra * ratio[s] // len(math_train) for s in sorted(ratio)}
        by_remainder = sorted(ratio, key=lambda s: (-(extra * ratio[s] % len(math_train)), s))
        for s in by_remainder[:extra - sum(shares.values())]:
            shares[s] += 1
        for s, n in shares.items():
            need[(s, '*')] += n
    excluded = ({r['sample_id'] for r in math_train} | reserved | quarantine | rejected_before |
                {r['sample_id'] for r in math_dev} | {r['sample_id'] for r in jsonl(math_files['math_dev'])})
    internal = SimilarityIndex(.85)
    for r in kept_math:
        internal.add(r)
    added, skipped = [], Counter()
    candidates = sorted((r for rows in pools.values() for r in rows), key=lambda r: order_key(r['sample_id']))
    for r in candidates:
        if not +need:
            break
        key = stratum(r)
        slot = key if need[key] > 0 else ((r['source'], '*') if need[(r['source'], '*')] > 0 else None)
        if slot is None:
            continue
        if r['sample_id'] in excluded:
            skipped['excluded_id'] += 1
            continue
        if math_index.query(r['problem']):
            skipped['decontamination_v5'] += 1
            continue
        if internal.match(r['problem']):
            skipped['internal_duplicate'] += 1
            continue
        n = length_fn(r['problem'], 'math')
        if n > budget_limit(contract, 'math'):
            skipped['prompt_budget'] += 1
            continue
        added.append(added_math_row(r, n, budget_limit(contract, 'math')))
        internal.add(r)
        need[slot] -= 1
    if +need:
        raise ValueError('Math pools exhausted before refill completed: ' + repr(dict(+need)))

    final_math_ids = [r['sample_id'] for r in kept_math] + [r['sample_id'] for r in added]
    answers = []
    for sid in final_math_ids:
        source_row = pool_by_id[sid]
        answers.append({'sample_id': sid, 'reference_answer': source_row['reference_answer'], 'source': source_row['source'],
                        'use': 'analysis / rejection-sampling baselines only; never part of a model prompt'})

    out = root / paths['out'] / 'candidate'
    if out.exists():
        raise FileExistsError(f'{out} exists; never overwrite a candidate (delete it deliberately to rebuild)')
    out.mkdir(parents=True)
    with (out / 'added_math_train.jsonl').open('wb') as f:
        for row in added:
            f.write(dump_row(row))
    with (out / ANSWERS).open('wb') as f:
        for row in answers:
            f.write(dump_row(row))
    changes = {'version': VERSION, 'parent_manifest': paths['parent_manifest'], 'parent_manifest_sha256': sha(parent_path),
               'removed': dict(sorted(removed.items())), 'added_math_ids': [r['sample_id'] for r in added],
               'code_refill': 'none: removed code rows are not replaced (raw code sources were deleted in the reorg)',
               'previously_rejected_math_excluded_from_refill': sorted(rejected_before)}
    write(out / 'changes.json', changes)
    counts = {'math_train': {'v4': len(math_train), 'removed': sum(v['domain'] == 'math' for v in removed.values()),
                             'added': len(added), 'v5': len(final_math_ids)},
              'code_train': {'v4': len(code_train), 'removed': sum(v['domain'] == 'code' for v in removed.values()),
                             'added': 0, 'v5': len(code_train) - sum(v['domain'] == 'code' for v in removed.values())}}
    by_set = defaultdict(Counter)
    for sid, item in removed.items():
        for hit in item['evidence']:
            by_set[item['domain']][hit['protected_set']] += 1
    report = {'version': VERSION, 'status': 'candidate_pending_owner_approval', 'counts': counts, 'budget': budget,
              'decontamination': {'rules': {'ngram': NGRAM, 'lcs_ratio': LCS_RATIO, 'jaccard_3gram': .85,
                                            'code_boilerplate_removed': True},
                                  'math_protected_items': len(math_index.items), 'code_protected_items': len(code_index.items),
                                  'flagged_math_train_rows': len(math_flags), 'flagged_code_train_rows': len(code_flags),
                                  'hits_by_protected_set': {d: dict(c) for d, c in by_set.items()}},
              'refill': {'need_by_stratum': {str(k): v for k, v in Counter(
                             stratum(pool_by_id[r['sample_id']]) for r in math_train if r['sample_id'] in removed).items()},
                         'skipped_candidates': dict(skipped), 'math_target': math_target,
                         'added_by_source': dict(Counter(r['source'] for r in added))},
              'code_interfaces_v5': dict(Counter(r['interface'] for r in code_train if r['sample_id'] not in removed)),
              'limits': ['Lexical overlap rules cannot prove semantic or pretraining decontamination.',
                         'Added rows passed deterministic gates only: no AI review and no itemwise human review.',
                         'IDs of 40 MATH top-up candidates that were not admitted in 2026-10 were lost; '
                         'they cannot be excluded by ID.',
                         'Removed code rows are not refilled.']}
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    report['candidate_files'] = files
    report_dir = root / paths['report']
    write(report_dir / 'audit.json', report)
    inputs = {paths['parent_manifest']: sha(parent_path), paths['math_manifest']: sha(root / paths['math_manifest']),
              **{OFFICIAL_CODE[k]['path']: OFFICIAL_CODE[k]['sha256'] for k in OFFICIAL_CODE}}
    write(out / 'candidate_manifest.json', {'version': VERSION, 'files': files, 'inputs': inputs,
                                            'code_inputs': code_inputs(),
                                            'audit_report': (Path(paths['report']) / 'audit.json').as_posix(),
                                            'audit_report_sha256': sha(report_dir / 'audit.json'),
                                            'status': 'candidate_pending_owner_approval'})
    return report


def approve_candidate(root, approver, note='', paths=None):
    """Owner decision: materialise release/ from v4 + candidate. Never overwrites."""
    root = Path(root)
    paths = {**PATHS, **(paths or {})}
    if not approver or not approver.strip():
        raise ValueError('An explicit approver name is required')
    base = root / paths['out']
    candidate, release = base / 'candidate', base / 'release'
    if release.exists():
        raise FileExistsError('release/ already exists; never overwrite an approved release')
    manifest = read(candidate / 'candidate_manifest.json')
    for name, digest in manifest['files'].items():
        if sha(candidate / name) != digest:
            raise ValueError('Candidate file changed after audit: ' + name)
    if sha(root / manifest['audit_report']) != manifest['audit_report_sha256']:
        raise ValueError('Audit report changed after the candidate was built')
    for path, digest in manifest['inputs'].items():
        if sha(root / path) != digest:
            raise ValueError('Candidate input changed: ' + path)
    if manifest['code_inputs'] != code_inputs():
        raise ValueError('Audit code changed since the candidate was built; rebuild the candidate')
    changes = read(candidate / 'changes.json')
    parent_path = root / changes['parent_manifest']
    parent_release = parent_path.parent
    removed = set(changes['removed'])
    approved_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    record = {'approver': approver.strip(), 'approved_at_utc': approved_at, 'note': note,
              'candidate_manifest_sha256': sha(candidate / 'candidate_manifest.json'),
              'audit_report_sha256': manifest['audit_report_sha256'],
              'statement': 'Owner batch approval of deterministic-gate candidates; not an itemwise human review.'}
    release.mkdir(parents=True)
    added = jsonl(candidate / 'added_math_train.jsonl')
    with (release / 'project2_math_train.jsonl').open('wb') as f:
        for line in raw_lines(parent_release / 'project2_math_train.jsonl'):
            if json.loads(line)['sample_id'] not in removed:
                f.write(line)
        for row in added:
            row = {**row, 'training_approved': True, 'approval_method': APPROVAL_METHOD, 'v5_status': 'approved',
                   'approval_record': {'approver': record['approver'], 'approved_at_utc': approved_at}}
            f.write(dump_row(row))
    with (release / 'project2_code_train.jsonl').open('wb') as f:
        for line in raw_lines(parent_release / 'project2_code_train.jsonl'):
            if json.loads(line)['sample_id'] not in removed:
                f.write(line)
    parent = read(parent_path)
    for name in ['project2_math_dev.jsonl', 'project2_code_dev.jsonl', parent['protected_dev_ids_file']]:
        shutil.copyfile(parent_release / name, release / name)
    shutil.copyfile(candidate / ANSWERS, release / ANSWERS)
    write(release / EXCLUDED, {'version': VERSION, 'rule': 'Removed from v5 training for overlap or budget; never re-add.',
                               'removed': changes['removed'],
                               'previously_rejected_math': changes['previously_rejected_math_excluded_from_refill']})
    files = {p.name: sha(p) for p in sorted(release.iterdir()) if p.is_file()}
    math_rows = jsonl(release / 'project2_math_train.jsonl')
    code_rows = jsonl(release / 'project2_code_train.jsonl')
    write(release / 'manifest.json', {
        'version': VERSION, 'project': 2,
        'summary': {'project2_math_train': len(math_rows), 'project2_math_dev': len(jsonl(release / 'project2_math_dev.jsonl')),
                    'project2_code_train': dict(Counter(r['interface'] for r in code_rows)),
                    'project2_code_dev': dict(Counter(r['interface'] for r in jsonl(release / 'project2_code_dev.jsonl'))),
                    'removed_rows': len(removed), 'added_rows': len(added), 'models_trained': 0,
                    'approval': 'v4 rows keep their v4 approval; added rows: ' + APPROVAL_METHOD},
        'parent_manifest': changes['parent_manifest'], 'parent_manifest_sha256': changes['parent_manifest_sha256'],
        'approved_training_files': ['project2_math_train.jsonl', 'project2_code_train.jsonl'],
        'approved_evaluation_files': ['project2_math_dev.jsonl', 'project2_code_dev.jsonl'],
        'protected_dev_ids_file': parent['protected_dev_ids_file'],
        'excluded_training_ids_file': EXCLUDED,
        'sidecar_files': {ANSWERS: 'reference answers for analysis / RFT baselines; not a training input'},
        'files': files,
        'inputs': {'candidate/candidate_manifest.json': record['candidate_manifest_sha256'],
                   manifest['audit_report']: manifest['audit_report_sha256'], **manifest['inputs'],
                   **manifest['code_inputs']},
        'approval': record,
        'approval_scope': 'Only the listed train files. Dev files are evaluation-only. v4 stays unchanged.'})
    return record
