"""Evaluation protocol v3: one fixed greedy protocol for students, teachers and baselines.

* Prompts: math and internal code dev use exactly the training prompt contract.
  HumanEval+/MBPP+ use EvalPlus's own chat prompt construction (instruction
  prefix plus an assistant response prefix that opens a ```python block), so
  scores are comparable with the EvalPlus leaderboard protocol.
* Decoding: greedy, a fixed per-domain budget, generation restricted to the
  audited common support (as in training).
* Math grading: the repository's conservative verifier (strict, no implicit
  multiplication / intervals / text answers) AND HF Math-Verify, if installed.
  ``grader_self_check`` grades the references against themselves, which gives
  each grader's coverage ceiling. On MATH-500 the conservative verifier
  accepts only 389/500 references.
* Code: this module only writes candidate files. Execution stays in the Docker
  sandbox (scripts/evaluate_code_v2.py, scripts/evaluate_evalplus_v2.py),
  never on the GPU host.
"""
import hashlib
import json
from pathlib import Path

from src.evaluation.math_verifier import check as conservative_check

EVALPLUS_INSTRUCTION = ('Please provide a self-contained Python script that solves the following problem '
                        'in a markdown code block:')
EVALPLUS_RESPONSE = ('Below is a Python script with a self-contained function that solves the problem and '
                     'passes corresponding tests:')
MAGIC = '<<<__EVALPLUS_MAGIC_SPLITTER__>>>'
SUITES = {
    'math_dev': ('math', 'release:project2_math_dev.jsonl'),
    'math500': ('math', 'data/math/math500_test.jsonl'),
    'gsm8k_test': ('math', 'data/math/gsm8k_test.jsonl'),
    'code_dev': ('code', 'release:project2_code_dev.jsonl'),
    'humanevalplus': ('evalplus', 'data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl'),
    'mbppplus': ('evalplus', 'data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl'),
}
DEFAULT_MAX_NEW = {'math': 2048, 'code': 1536, 'evalplus': 1536}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def suite_rows(name, release_manifest):
    kind, location = SUITES[name]
    path = Path(release_manifest).parent / location.split(':', 1)[1] if location.startswith('release:') else Path(location)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r.get('sample_id', r.get('task_id')) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate evaluation IDs in ' + name)
    return kind, path, rows


def render(tokenizer, kind, row, contract):
    """Token IDs of the evaluation prompt; never truncates the problem."""
    if kind == 'math':
        messages = [{'role': 'system', 'content': contract['math_system']}, {'role': 'user', 'content': row['problem']}]
        return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if kind == 'code':
        return tokenizer.apply_chat_template([{'role': 'user', 'content': row.get('prompt', row['problem'])}],
                                             tokenize=True, add_generation_prompt=True)
    user = f"{EVALPLUS_INSTRUCTION}\n```\n{row['prompt'].strip()}\n```\n"
    assistant = f'{EVALPLUS_RESPONSE}\n```python\n{MAGIC}\n```\n'
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': user}, {'role': 'assistant', 'content': assistant}],
                                         tokenize=False)
    prefix = text.split(MAGIC)[0]
    return tokenizer(prefix, add_special_tokens=False)['input_ids']


def evalplus_text(generated):
    """Re-attach the opened ```python fence so extract_code sees exactly one block."""
    body = generated.split('```')[0]
    return '```python\n' + body.rstrip() + '\n```'


def math_verify_check(text, reference):
    try:
        from math_verify import parse, verify
    except ImportError:
        return {'status': 'unavailable', 'correct': None}
    try:
        gold, answer = parse('\\boxed{' + reference + '}'), parse(text)
        return {'status': 'ok' if gold else 'unparsed_reference', 'correct': bool(gold and answer and verify(gold, answer))}
    except Exception as error:  # Math-Verify raises on pathological inputs; count as wrong, keep the reason.
        return {'status': 'error:' + type(error).__name__, 'correct': False}


def grade_math(text, reference, truncated):
    conservative = conservative_check(text, reference)
    standard = math_verify_check(text, reference)
    return {'conservative': conservative, 'math_verify': standard, 'truncated': truncated}


def grader_self_check(rows):
    out = {'rows': len(rows), 'conservative': 0, 'math_verify': 0, 'math_verify_available': True}
    for r in rows:
        boxed = 'The final answer is \\boxed{' + r['reference_answer'] + '}.'
        out['conservative'] += bool(conservative_check(boxed, r['reference_answer'])['correct'])
        verdict = math_verify_check(boxed, r['reference_answer'])
        if verdict['status'] == 'unavailable':
            out['math_verify_available'] = False
        out['math_verify'] += bool(verdict['correct'])
    return out


def summarize(kind, records):
    n = len(records)
    out = {'count': n, 'truncated': sum(r['finish_reason'] == 'length' for r in records),
           'mean_generated_tokens': sum(r['generated_tokens'] for r in records) / max(n, 1)}
    if kind == 'math':
        out['accuracy_conservative'] = sum(bool(r['grade']['conservative']['correct']) for r in records) / max(n, 1)
        available = [r for r in records if r['grade']['math_verify']['status'] != 'unavailable']
        out['accuracy_math_verify'] = (sum(bool(r['grade']['math_verify']['correct']) for r in available) / n
                                       if available and len(available) == n else None)
    return out
