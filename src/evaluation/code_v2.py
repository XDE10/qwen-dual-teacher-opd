"""Internal code evaluation v2. Candidate code executes only in the Docker worker."""
import ast
import collections
import json
import math
from pathlib import Path
import re


def extract_code(text):
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, flags=re.S)
    if len(blocks) > 1:
        raise ValueError('Ambiguous multiple code blocks')
    code = blocks[0] if blocks else text
    ast.parse(code)
    return code


def compare(actual, expected, spec):
    kind = spec['type']
    if kind == 'exact':
        # Preserve Python source-test equality (5 == 5.0), with no epsilon.
        # Booleans are kept distinct from numeric values unless both are bool.
        if isinstance(actual, bool) != isinstance(expected, bool):
            return False
        return actual == expected
    if kind == 'stdout_tokens_exact':
        # Explicit legacy whitespace policy; NEVER presented as float tolerance.
        return actual.split() == expected.split()
    if kind == 'float':
        if isinstance(actual, bool) or isinstance(expected, bool):
            return False
        if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
            return len(actual) == len(expected) and all(compare(a, b, spec) for a, b in zip(actual, expected))
        try:
            a, b = float(actual), float(expected)
        except (TypeError, ValueError):
            return False
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=spec['rtol'], abs_tol=spec['atol'])
    if kind == 'stdout_float_tokens':
        a, b = actual.split(), expected.split()
        return len(a) == len(b) and all(compare(x, y, {**spec, 'type': 'float'}) for x, y in zip(a, b))
    if kind == 'unordered':
        # Multiset, preserves duplicates. Canonical representations retain types.
        return collections.Counter(repr(x) for x in actual) == collections.Counter(repr(x) for x in expected)
    if kind == 'constructive':
        raise ValueError('Constructive task requires a registered dedicated checker')
    raise ValueError('Unknown comparator: ' + kind)


def make_job(row, candidate, spec):
    if spec.get('protocol_version') == 'review-v3':
        from src.evaluation.reviewed_registry_v3 import make_reviewed_job
        return make_reviewed_job(row, candidate, spec)
    code = extract_code(candidate)
    if spec.get('status') not in {'reviewed', 'reference_selfcheck_only'}:
        raise ValueError('Comparator must be explicitly reviewed before model scoring')
    if spec['type'] == 'constructive':
        raise ValueError('No dedicated constructive checker registered')
    tests = row.get('test_cases') or (row.get('verification_info') or {}).get('test_cases')
    if not tests:
        raise ValueError('Missing sealed source tests')
    tests = [dict(t) for t in tests]
    if row['interface'] == 'function':
        tree = ast.parse(code)
        functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == row['entry_point']]
        if len(functions) != 1:
            raise ValueError('Expected one module-level entry point')
        signature = ast.parse(row['function_signature'] + '\n    pass').body[0]
        required_names = [a.arg for a in signature.args.posonlyargs + signature.args.args]
        actual_names = [a.arg for a in functions[0].args.posonlyargs + functions[0].args.args]
        if actual_names != required_names:
            raise ValueError('Candidate entry signature differs from sealed interface')
        if any('args' not in t or 'kwargs' not in t for t in tests):
            raise ValueError('Missing structured function test arguments')
        for test in tests:
            if test.get('assertion'):
                calls = [n for n in ast.walk(ast.parse(test['assertion'])) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Name) and n.func.id == row['entry_point']]
                if len(calls) != 1:
                    raise ValueError('Ambiguous source test call')
                # Keep tuple/set argument syntax lost by JSON serialization.
                test['sealed_call_expression'] = ast.unparse(calls[0])
    return {'sample_id': row['sample_id'], 'kind': 'candidate_v2', 'interface': row['interface'],
            'candidate_code': code, 'entry_point': row.get('entry_point'),
            'tests': tests, 'comparator': spec}


def evaluate(rows, candidates, registry, executor=None):
    from src.data.code_sandbox_v2 import run_jobs, ROOT
    reviewed = any(s.get('protocol_version') == 'review-v3' for s in registry.values())
    if reviewed:
        from src.evaluation.reviewed_registry_v3 import validate_registry
        validate_registry(rows, registry)
    worker = 'code_eval_worker_v3.py' if reviewed else 'code_eval_worker_v2.py'
    executor = executor or (lambda jobs: run_jobs(jobs, worker=ROOT / 'scripts' / worker))
    jobs, failed = [], []
    for row in rows:
        sid = row['sample_id']
        if registry.get(sid, {}).get('status') == 'blocked_after_review':
            failed.append({'sample_id': sid, 'status': 'blocked_after_review',
                           'error': registry[sid]['review_reason']})
            continue
        try:
            jobs.append(make_job(row, candidates[sid], registry[sid]))
        except (ValueError, KeyError, SyntaxError) as error:
            failed.append({'sample_id': sid, 'status': 'unsupported_or_extraction_error', 'error': str(error)})
    response = executor(jobs) if jobs else {'results': []}
    results = response['results'] + failed
    if len(results) != len(rows) or {r['sample_id'] for r in results} != {r['sample_id'] for r in rows}:
        raise ValueError('Executor returned wrong sample identities')
    return {'protocol': 'internal-dev-review-v3' if reviewed else 'internal-dev-v2', 'results': results,
            'coverage': {'total_tasks': len(rows), 'blocked_tasks': sum(r['status']=='blocked_after_review' for r in results),
                         'passed_tasks': sum(r['status']=='passed' for r in results)},
            'warning': 'Source-test compatibility only; not official EvalPlus or proof of semantic completeness'}


def provisional_registry(rows):
    # This sidecar does not change published rows. No silent comparator inference.
    return {r['sample_id']: {'type': 'exact' if r['interface'] == 'function' else 'stdout_tokens_exact',
            'status': 'unreviewed', 'source': 'legacy source-test comparator; 未验证 semantic suitability',
            'requires_review': ['float tolerance', 'unordered outputs', 'multiple valid constructions']}
            for r in rows}
