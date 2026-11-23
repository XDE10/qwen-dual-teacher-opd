import ast
from pathlib import Path

import pytest

from src.evaluation.code_v2 import compare, evaluate, extract_code, make_job, provisional_registry


def fixture_row():
    return {'sample_id': 'fixture', 'interface': 'function', 'entry_point': 'add',
        'function_signature': 'def add(x, y):', 'reference_code': 'def add(x, y):\n    return x+y',
        'test_cases': [{'args': [2, 3], 'kwargs': {}, 'expected_repr': '5'}]}


def test_comparators_have_distinct_float_and_stdout_semantics():
    assert not compare('1.00001\n', '1\n', {'type': 'stdout_tokens_exact'})
    assert compare('1.00001\n', '1\n', {'type': 'stdout_float_tokens', 'atol': .0001, 'rtol': 0})
    assert not compare(float('nan'), 1, {'type': 'float', 'rtol': 1., 'atol': 1.})
    assert not compare(True, 1, {'type': 'exact'})
    assert compare(5.0, 5, {'type': 'exact'})  # exact Python numeric equality, no tolerance
    assert not compare(5.0000001, 5, {'type': 'exact'})
    assert compare([2, 1, 1], [1, 2, 1], {'type': 'unordered'})
    assert not compare([1, 2], [1, 1, 2], {'type': 'unordered'})
    with pytest.raises(ValueError, match='dedicated'):
        compare('x', 'y', {'type': 'constructive'})


def test_worker_and_host_comparators_match_without_executing_worker():
    # Importing Linux worker on Windows is forbidden. Compile ONLY pure comparator AST.
    tree = ast.parse(Path('scripts/code_eval_worker_v2.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compare')
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pure comparator>', 'exec'), scope)
    for a, b, spec in [('1.00001', '1', {'type': 'stdout_float_tokens', 'atol': .0001, 'rtol': 0}),
                       ([2, 1], [1, 2], {'type': 'unordered'}), (True, 1, {'type': 'exact'})]:
        assert scope['compare'](a, b, spec) == compare(a, b, spec)


def test_extraction_signature_and_fail_closed_registry():
    r = fixture_row()
    spec = {'type': 'exact', 'status': 'reviewed'}
    assert make_job(r, '```python\n' + r['reference_code'] + '\n```', spec)['candidate_code']
    with pytest.raises(ValueError, match='signature'):
        make_job(r, 'def add(z, w): return z+w', spec)
    with pytest.raises(ValueError, match='reviewed'):
        make_job(r, r['reference_code'], provisional_registry([r])['fixture'])
    with pytest.raises(ValueError, match='Ambiguous'):
        extract_code('```python\nx=1\n```\n```python\nx=2\n```')


def test_reference_candidate_pipeline_transport_with_fake_executor():
    r = fixture_row()
    captured = []
    def fake(jobs):
        captured.extend(jobs)
        return {'results': [{'sample_id': 'fixture', 'status': 'passed', 'checked': 1}]}
    result = evaluate([r], {'fixture': r['reference_code']},
                      {'fixture': {'type': 'exact', 'status': 'reviewed'}}, fake)
    assert captured[0]['candidate_code'] == r['reference_code']
    assert captured[0]['tests'][0]['args'] == [2, 3]
    assert result['results'][0]['status'] == 'passed'
