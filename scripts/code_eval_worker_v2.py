"""Docker-only candidate worker v2. Never run dataset code on the host."""
import ast
import json
import os
import resource
import subprocess
import sys
import tempfile

# Shared pure comparator is embedded in the mounted worker to avoid repo mounts.
def compare(a, b, spec):
    import math
    import collections
    kind = spec['type']
    if kind == 'exact':
        if isinstance(a, bool) != isinstance(b, bool):
            return False
        return a == b
    if kind == 'stdout_tokens_exact':
        return a.split() == b.split()
    if kind in {'float', 'stdout_float_tokens'}:
        if kind == 'stdout_float_tokens':
            a, b = a.split(), b.split()
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(compare(x, y, {**spec, 'type': 'float'}) for x, y in zip(a, b))
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            return False
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=spec['rtol'], abs_tol=spec['atol'])
    if kind == 'unordered':
        return collections.Counter(repr(x) for x in a) == collections.Counter(repr(x) for x in b)
    raise ValueError('Dedicated checker not registered')


def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_NPROC, (8, 8))


def execute(program, stdin=''):
    with tempfile.TemporaryDirectory(dir='/tmp') as d:
        path = os.path.join(d, 'candidate.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(program)
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                result = subprocess.run([sys.executable, '-I', path], input=stdin.encode(),
                    stdout=stdout, stderr=stderr, timeout=5, cwd=d,
                    env={'PATH': '/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE': '1'}, preexec_fn=limits)
            except subprocess.TimeoutExpired:
                return {'ok': False, 'error': 'timeout'}
            stdout.seek(0)
            stderr.seek(0)
            return {'ok': result.returncode == 0, 'stdout': stdout.read(32768).decode('utf-8', 'replace'),
                    'error': stderr.read(1500).decode('utf-8', 'replace')}


def evaluate(job):
    checked = 0
    for test in job['tests']:
        if job['interface'] == 'function':
            # Call structured args/kwargs, preserving Python repr return protocol.
            expression = test.get('sealed_call_expression') or job['entry_point'] + '(*' + repr(test['args']) + ', **' + repr(test['kwargs']) + ')'
            call = '\n_opd_result = ' + expression + '\nprint("__OPD_RETURN__" + repr(_opd_result))\n'
            execution = execute('from __future__ import annotations\n' + job['candidate_code'] + call)
            if not execution['ok']:
                return {'sample_id': job['sample_id'], 'status': 'execution_error', 'detail': execution, 'checked': checked}
            lines = [l for l in execution['stdout'].splitlines() if l.startswith('__OPD_RETURN__')]
            if len(lines) != 1:
                return {'sample_id': job['sample_id'], 'status': 'return_protocol_error', 'checked': checked}
            try:
                actual = ast.literal_eval(lines[0][len('__OPD_RETURN__'):])
                expected = ast.literal_eval(test['expected_repr'])
            except (ValueError, SyntaxError):
                return {'sample_id': job['sample_id'], 'status': 'unsupported_return_repr', 'checked': checked}
        else:
            execution = execute(job['candidate_code'], test['input'])
            if not execution['ok']:
                return {'sample_id': job['sample_id'], 'status': 'execution_error', 'detail': execution, 'checked': checked}
            actual, expected = execution['stdout'], test['output']
        if not compare(actual, expected, job['comparator']):
            return {'sample_id': job['sample_id'], 'status': 'mismatch', 'checked': checked}
        checked += 1
    return {'sample_id': job['sample_id'], 'status': 'passed', 'checked': checked}


if __name__ == '__main__':
    jobs = json.load(sys.stdin)
    json.dump({'python': sys.version, 'results': [evaluate(j) for j in jobs]}, sys.stdout)
