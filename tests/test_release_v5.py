"""Decontamination v5 rules and the v5 candidate/approval flow on synthetic fixtures."""
import json
import random
from pathlib import Path

import pytest

from src.data import release_v5
from src.data.approved_training_inputs import load_approved
from src.data.decontamination_v5 import (ProtectedIndex, lcs_length, lcs_length_dp, strip_code_boilerplate, words)


def test_bit_parallel_lcs_matches_dynamic_program():
    rng = random.Random(0)
    for _ in range(2000):
        a = [rng.choice('abcd') for _ in range(rng.randint(0, 30))]
        b = [rng.choice('abcd') for _ in range(rng.randint(0, 30))]
        assert lcs_length(a, b) == lcs_length_dp(a, b)


def test_rules_fire_and_boilerplate_is_ignored():
    long_text = ' '.join(f'word{i}' for i in range(40))
    index = ProtectedIndex([{'id': 't1', 'text': long_text, 'set': 'test'},
                            {'id': 't2', 'text': 'Compute seven choose three', 'set': 'test'},
                            {'id': 't3', 'text': 'What is 2 plus 2 equal to today', 'set': 'test'}])
    paraphrase = ' '.join(f'word{i}' for i in range(30)) + ' extra tokens appended here'
    assert {h['rule'] for h in index.query(paraphrase)} == {'ngram13_lcs'}
    assert index.query('compute SEVEN choose three!')[0]['rule'] == 'normalized_exact'
    assert index.query('Please compute seven choose three')[0]['rule'] == 'short_containment'
    assert index.query('An unrelated question about prime numbers and their gaps') == []
    wrapped = 'Solve the following coding problem using the programming language python:\nAdd two ints.\n' \
              'The input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.'
    assert words(strip_code_boilerplate(wrapped)) == ['add', 'two', 'ints']


# ------------------------------------------------------------------ fixtures

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')


def problem(tag, n=30):
    return ' '.join(f'{tag}{i}' for i in range(n))


def fixture_root(tmp_path):
    root = tmp_path
    (root / 'configs').mkdir()
    (root / 'configs/project2_prompt_contract.json').write_text(
        Path('configs/project2_prompt_contract.json').read_text(encoding='utf-8'), encoding='utf-8')
    rel = root / 'data/v4/release'
    math_pool = [{'sample_id': f'math/train/{i:05d}', 'source': 'DigitalLearningGmbH/MATH-lighteval', 'source_revision': 'r',
                  'source_split': 'train', 'source_id': str(i), 'problem': problem(f'm{i}x'), 'problem_hash': f'h{i}',
                  'family_id': f'h{i}', 'raw_problem_sha256': f'r{i}', 'level': f'Level {1 + i % 5}', 'problem_type': 'Algebra',
                  'reference_answer': str(i), 'solution': 'x', 'domain': 'math'} for i in range(40)]
    gsm_pool = [{'sample_id': f'gsm8k/train/{i:05d}', 'source': 'openai/gsm8k', 'source_revision': 'g', 'source_split': 'train',
                 'source_id': str(i), 'problem': problem(f'g{i}y'), 'problem_hash': f'gh{i}', 'family_id': f'gh{i}',
                 'reference_answer': str(i), 'solution': 'y', 'domain': 'math'} for i in range(20)]
    test500 = [{'sample_id': 'math500/test/00000', 'problem': problem('m0x', 26) + ' slightly reworded'}]
    files = {'math_train': math_pool, 'gsm8k_train': gsm_pool, 'math500_test': test500,
             'gsm8k_test': [{'sample_id': 'gsm8k/test/00000', 'problem': 'unrelated gsm test problem'}],
             'math_source_test': [{'sample_id': 'math/test/00000', 'problem': 'another unrelated test problem'}],
             'math_dev': [{'sample_id': 'math/train/00039', 'problem': math_pool[39]['problem']}],
             'math_quarantine': [{'sample_id': 'math/train/00038', 'problem': 'q'}]}
    manifest = {'files': {}}
    for name, rows in files.items():
        write_jsonl(root / f'data/math/{name}.jsonl', rows)
        manifest['files'][name] = {'path': f'data/math/{name}.jsonl', 'sha256': release_v5.sha(root / f'data/math/{name}.jsonl')}
    (root / 'data/math_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    approved = {'training_approved': True}
    math_train = [{**{k: v for k, v in math_pool[i].items() if k in ('sample_id', 'source', 'problem', 'domain')}, **approved}
                  for i in range(0, 10)] + [{'sample_id': 'gsm8k/train/00000', 'source': 'openai/gsm8k',
                                             'problem': gsm_pool[0]['problem'], 'domain': 'math', **approved}]
    code_train = [{'sample_id': f'code/vfc_{i}', 'domain': 'code', 'interface': 'stdin_stdout' if i % 2 else 'function',
                   'problem': problem(f'c{i}z'), 'prompt': problem(f'c{i}z'), **approved} for i in range(6)]
    code_dev = [{'sample_id': 'code/vfc_900', 'domain': 'code', 'interface': 'stdin_stdout', 'problem': problem('c1z'),
                 'prompt': problem('c1z'), 'training_approved': False}]
    math_dev = [{'sample_id': 'math/train/00039', 'domain': 'math', 'problem': math_pool[39]['problem'], 'training_approved': False}]
    write_jsonl(rel / 'project2_math_train.jsonl', math_train)
    write_jsonl(rel / 'project2_code_train.jsonl', code_train)
    write_jsonl(rel / 'project2_math_dev.jsonl', math_dev)
    write_jsonl(rel / 'project2_code_dev.jsonl', code_dev)
    (rel / 'reserved.json').write_text(json.dumps({'sample_ids': ['code/vfc_900', 'math/train/00039', 'math/train/00030']}))
    (rel / 'manifest.json').write_text(json.dumps({
        'approved_training_files': ['project2_math_train.jsonl', 'project2_code_train.jsonl'],
        'protected_dev_ids_file': 'reserved.json',
        'files': {n: release_v5.sha(rel / n) for n in [*release_v5.FILES, 'reserved.json']}}), encoding='utf-8')
    he = [{'task_id': 'HumanEval/0', 'prompt': 'def f():\n    """Return one."""\n', 'canonical_solution': '    return 1\n'}]
    mb = [{'task_id': 'Mbpp/2', 'prompt': '"""Write a function that adds."""', 'canonical_solution': 'def g(a, b):\n    return a + b\n'}]
    official = {}
    for name, rows in [('humanevalplus', he), ('mbppplus', mb)]:
        path = root / f'data/official/{name}.jsonl'
        write_jsonl(path, rows)
        official[name] = {'path': f'data/official/{name}.jsonl', 'sha256': release_v5.sha(path), 'rows': len(rows)}
    paths = {'parent_manifest': 'data/v4/release/manifest.json', 'out': 'data/v5', 'report': 'reports/v5'}
    original_pool = {r['sample_id'] for r in math_train} | {'math/train/00011'}  # 00011 was "review-rejected"
    return root, paths, official, original_pool


def test_candidate_removes_overlaps_refills_by_stratum_and_approval_is_explicit(tmp_path, monkeypatch):
    root, paths, official, original_pool = fixture_root(tmp_path)
    monkeypatch.setattr(release_v5, 'OFFICIAL_CODE', official)
    length = lambda text, domain: len(text.split())
    report = release_v5.build_candidate(root, paths, length_fn=length, rejected_fn=lambda _: original_pool)
    changes = json.loads((root / 'data/v5/candidate/changes.json').read_text())
    assert set(changes['removed']) == {'math/train/00000', 'code/vfc_1'}  # MATH-500 and code-dev near duplicates
    added = [json.loads(l) for l in (root / 'data/v5/candidate/added_math_train.jsonl').read_text().splitlines()]
    assert len(added) == 1 and added[0]['level'] == 'Level 1' and added[0]['training_approved'] is False
    forbidden = {'math/train/00011', 'math/train/00030', 'math/train/00038', 'math/train/00039'}
    assert added[0]['sample_id'] not in forbidden and 'reference_answer' not in added[0]
    assert report['counts']['math_train'] == {'v4': 11, 'removed': 1, 'added': 1, 'v5': 11}
    with pytest.raises(FileExistsError):
        release_v5.build_candidate(root, paths, length_fn=length, rejected_fn=lambda _: original_pool)
    assert not (root / 'data/v5/release').exists()
    with pytest.raises(ValueError, match='approver'):
        release_v5.approve_candidate(root, '  ', paths=paths)
    record = release_v5.approve_candidate(root, 'owner', 'checked', paths=paths)
    assert record['approver'] == 'owner'
    manifest = root / 'data/v5/release/manifest.json'
    math = load_approved(manifest, 'project2_math_train.jsonl')
    code = load_approved(manifest, 'project2_code_train.jsonl')
    assert len(math) == 11 and len(code) == 5 and math[-1]['sample_id'] == added[0]['sample_id']
    old = (root / 'data/v4/release/project2_math_train.jsonl').read_bytes().splitlines(keepends=True)
    new = (root / 'data/v5/release/project2_math_train.jsonl').read_bytes().splitlines(keepends=True)
    assert new[:10] == old[1:]  # retained lines byte-identical
    answers = [json.loads(l) for l in (root / 'data/v5/release/project2_math_train_answers.jsonl').read_text().splitlines()]
    assert {a['sample_id'] for a in answers} == {r['sample_id'] for r in math}
    with pytest.raises(FileExistsError):
        release_v5.approve_candidate(root, 'owner', paths=paths)


def test_approval_rejects_tampered_candidate(tmp_path, monkeypatch):
    root, paths, official, original_pool = fixture_root(tmp_path)
    monkeypatch.setattr(release_v5, 'OFFICIAL_CODE', official)
    release_v5.build_candidate(root, paths, length_fn=lambda t, d: 5, rejected_fn=lambda _: original_pool)
    added = root / 'data/v5/candidate/added_math_train.jsonl'
    added.write_text(added.read_text().replace('"training_approved": false', '"training_approved": true'))
    with pytest.raises(ValueError, match='changed after audit'):
        release_v5.approve_candidate(root, 'owner', paths=paths)


def test_budget_failures_are_removed(tmp_path, monkeypatch):
    root, paths, official, original_pool = fixture_root(tmp_path)
    monkeypatch.setattr(release_v5, 'OFFICIAL_CODE', official)
    length = lambda text, domain: 99999 if text.startswith('c4z') else 5
    report = release_v5.build_candidate(root, paths, length_fn=length, rejected_fn=lambda _: original_pool)
    changes = json.loads((root / 'data/v5/candidate/changes.json').read_text())
    assert changes['removed']['code/vfc_4']['reason'] == 'prompt_budget'
    assert report['budget']['code_train']['over_budget'] == ['code/vfc_4']
