"""Evaluation protocol v3: prompt rendering, candidate formats and grader coverage (CPU)."""
import json
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from src.evaluation.code_v2 import extract_code
from src.evaluation.eval_v3 import (EVALPLUS_RESPONSE, evalplus_text, grade_math, grader_self_check, render,
                                    suite_rows)
from src.evaluation.evalplus_v2 import convert_predictions
from src.teachers.opd_contract import load_contract, render_student_prompt

TOKENIZER = 'data/tokenizers/Qwen2.5-1.5B-Instruct'


@pytest.fixture(scope='module')
def tok():
    return AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True)


def test_math_and_code_dev_prompts_equal_the_training_contract(tok):
    contract = load_contract()
    row = {'problem': 'What is $1+1$?', 'prompt': 'Write f.\nRequired Python interface:\ndef f():'}
    assert render(tok, 'math', row, contract) == render_student_prompt(tok, row['problem'], 'math', contract)
    assert render(tok, 'code', row, contract) == render_student_prompt(tok, row['prompt'], 'code', contract)


def test_evalplus_prompt_ends_inside_assistant_python_block(tok):
    ids = render(tok, 'evalplus', {'prompt': 'def add(a, b):\n    """Add."""\n'}, load_contract())
    text = tok.decode(ids)
    assert text.rstrip().endswith('```python') and EVALPLUS_RESPONSE in text
    assert text.count('<|im_start|>assistant') == 1 and '<|im_end|>' not in text.split('<|im_start|>assistant')[1]


def test_candidate_texts_are_accepted_by_existing_scorers():
    text = evalplus_text('def add(a, b):\n    return a + b\n```\nThis adds.')
    assert extract_code(text).strip() == 'def add(a, b):\n    return a + b'
    tasks = [{'task_id': 'HumanEval/0'}]
    assert convert_predictions([{'task_id': 'HumanEval/0', 'text': text}], tasks)[0]['solution'].startswith('def add')
    unterminated = evalplus_text('def add(a, b):\n    return a + b')
    assert extract_code(unterminated).strip().endswith('return a + b')


def test_math_grading_reports_both_graders_and_truncation():
    result = grade_math('so the answer is \\boxed{3\\sqrt{13}}', '3\\sqrt{13}', truncated=False)
    assert result['conservative']['correct'] is False  # implicit multiplication unsupported by design
    assert result['math_verify']['status'] in {'ok', 'unavailable'}
    assert grade_math('\\boxed{72}', '72', True)['truncated'] is True


def test_grader_self_check_measures_conservative_ceiling():
    rows = [json.loads(l) for l in Path('data/math/math500_test.jsonl').read_text(encoding='utf-8').splitlines()[:60]]
    result = grader_self_check(rows)
    assert result['rows'] == 60 and result['conservative'] < 60  # strict verifier cannot parse every reference


def test_suites_resolve_release_files_and_reject_unknown():
    kind, path, rows = suite_rows('math_dev', 'data/code_training_repair_20261025_v4/release/manifest.json')
    assert kind == 'math' and len(rows) == 256 and path.name == 'project2_math_dev.jsonl'
    with pytest.raises(KeyError):
        suite_rows('unknown', 'x')
