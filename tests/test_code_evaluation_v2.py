import ast
import json
from pathlib import Path

import pytest

from src.evaluation.code_v2 import compare, extract_code, make_job, evaluate, provisional_registry
from src.evaluation.evalplus_v2 import convert_predictions, docker_command
from src.data.code_sandbox_v2 import environment, ROOT
from src.data.decontamination_v2 import Index, calibrate


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
    # Fake transport is deliberately NOT a Docker execution acceptance test.


def test_sandbox_relocates_current_worker_and_never_reuses_historical_mount():
    history = Path('reports/code_v2_sandbox_environment.json').read_bytes()
    result = environment()
    command = result['command']
    assert any((ROOT / 'scripts/code_sandbox_worker.py').resolve().as_posix() in x for x in command)
    assert not any('modern_genai' in x for x in command)
    assert '--network' in command and '--read-only' in command and '--cap-drop' in command
    assert Path('reports/code_v2_sandbox_environment.json').read_bytes() == history


def test_official_adapter_enforces_task_set_and_official_protocol(tmp_path):
    tasks = [{'task_id': 'HumanEval/0'}]
    samples = convert_predictions([{'task_id': 'HumanEval/0', 'text': 'def f(): return 1'}], tasks)
    assert samples == [{'task_id': 'HumanEval/0', 'solution': 'def f(): return 1'}]
    with pytest.raises(ValueError):
        convert_predictions([], tasks)
    command = docker_command('sha256:pinned', 'humaneval', tmp_path, tmp_path / 'data.jsonl')
    assert 'HUMANEVAL_OVERRIDE_PATH=/dataset.jsonl' in command
    assert any("m.version('evalplus')=='0.3.1'" in x for x in command)
    assert '--network' in command


def test_readonly_duplicate_detection_original_paraphrase_numeric_replacement():
    original = 'Find the total number of marbles when Alice has 12 red marbles and Bob has 15 blue marbles in the box.'
    paraphrase = 'Please solve this problem: ' + original
    numeric = original.replace('12', '27').replace('15', '31')
    index = Index([{'id': 'original', 'text': original}])
    assert index.query(original, .8)[0]['exact']
    assert index.query(paraphrase, .8)
    assert index.query(numeric, .8)[0]['numeric_template_5gram'] == 1.
    calibration = calibrate([(original, original), (original, numeric)], [('unrelated small task', original)])
    assert calibration['positive_recall'] == 1.


def test_heavy_paraphrase_limit_is_explicit():
    index = Index([{'id': 'x', 'text': 'Find the sum of all integers from one to ten.'}])
    assert not index.query('Compute the aggregate value for numbers 1 through 10.', .8)


def test_evalplus_memory_policy_is_versioned_and_core_is_disabled(tmp_path):
    from src.evaluation.evalplus_v2 import container_entry
    command = docker_command('sha256:pinned', 'mbpp', tmp_path, tmp_path / 'data.jsonl')
    assert command[command.index('--ulimit') + 1] == 'core=0:0'
    assert command[command.index('--memory') + 1] == '6g'
    assert 'OPD_EVALPLUS_EXTRA_MEMORY_BYTES=1073741824' in command
    assert 'memory_with_headroom' in command[command.index('-c') + 1]
    legacy = docker_command('sha256:pinned', 'mbpp', tmp_path, tmp_path / 'data.jsonl',
                            memory_policy='legacy-fixed-1g')
    assert 'memory_with_headroom' not in legacy[legacy.index('-c') + 1]
    assert 'prctl(4, 0' in legacy[legacy.index('-c') + 1]
    assert 'cli.main()' in legacy[legacy.index('-c') + 1]
    assert 'EVALPLUS_MAX_MEMORY_BYTES=1073741824' in legacy
    with pytest.raises(ValueError, match='policy'):
        container_entry('unknown')


def test_evalplus_headroom_is_calculated_per_child_and_not_fixed_4g():
    import io
    from types import SimpleNamespace
    from src.evaluation.evalplus_v2 import container_entry
    tree = ast.parse(container_entry())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    pages = [1200000]  # inherited VMS already exceeds 4 GiB
    fake_os = SimpleNamespace(environ={'OPD_EVALPLUS_EXTRA_MEMORY_BYTES': str(1 << 30)},
                              sysconf=lambda _: 4096)
    scope = {'os': fake_os, 'open': lambda *a, **kw: io.StringIO(str(pages[0]) + ' 0')}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<resource-policy>', 'exec'), scope)
    query = scope['memory_with_headroom']
    assert query() == pages[0] * 4096 + (1 << 30)
    pages[0] += 1000
    assert query() == pages[0] * 4096 + (1 << 30)
    fake_os.environ['OPD_EVALPLUS_EXTRA_MEMORY_BYTES'] = '0'
    with pytest.raises(ValueError, match='positive'):
        query()
