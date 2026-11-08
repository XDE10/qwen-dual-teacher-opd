"""Reviewed finite-test comparators, not a proof of complete specification.

Only standard-library code: embedded verbatim into the Docker worker by the
review-asset builder. No candidate code is executed by this module.
"""
import math
from collections import Counter


def exact(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return a == b


def constructive(output, stdin, checker):
    try:
        values = output.split()
        data = stdin.split()
        if checker == 'cycle_walk':
            n, m = map(int, data[:2])
            grid = data[2:]
            if len(grid) != n or any(len(r) != m for r in grid) or len(values) != 1:
                return False
            cells = {(i, j) for i in range(n) for j in range(m) if grid[i][j] in '*S'}
            starts = [(i, j) for i, j in cells if grid[i][j] == 'S']
            if len(starts) != 1 or len(values[0]) != len(cells):
                return False
            start = starts[0]
            position = start
            visited = {start}
            directions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
            for i, letter in enumerate(values[0]):
                if letter not in directions:
                    return False
                dr, dc = directions[letter]
                position = position[0] + dr, position[1] + dc
                if position not in cells:
                    return False
                if i == len(values[0]) - 1:
                    return position == start and visited == cells
                if position in visited:
                    return False
                visited.add(position)
            return False
        numbers = list(map(int, values))
        source = list(map(int, data))
        if checker == 'zero_sum_subset':
            p, q = 1, 0
            for _ in range(source[0]):
                n = source[p]
                p += 1
                array = source[p:p+n]
                p += n
                size = numbers[q]
                q += 1
                selected = numbers[q:q+size]
                q += size
                if not 1 <= size <= n or len(selected) != size or len(set(selected)) != size:
                    return False
                if any(i < 1 or i > n for i in selected) or sum(array[i-1] for i in selected) != 0:
                    return False
            return p == len(source) and q == len(numbers)
        if checker == 'equal_candy_groups':
            n = source[0]
            if len(source) != 1 or len(numbers) != n*n or set(numbers) != set(range(1, n*n+1)):
                return False
            return len({sum(numbers[i:i+n]) for i in range(0, n*n, n)}) == 1
        if checker == 'equal_leg_pairs':
            n = source[0]
            return (len(numbers) == 2*n and Counter(numbers) == Counter(source[1:])
                    and len({numbers[i] + numbers[i+1] for i in range(0, 2*n, 2)}) == 1)
    except (ValueError, IndexError, TypeError, OverflowError):
        return False
    return False


def compare(a, b, spec, test=None):
    kind = spec['type']
    if kind == 'exact':
        return exact(a, b)
    if kind == 'float':
        if isinstance(a, bool) or isinstance(b, bool):
            return False
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return False
        return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=spec['rtol'], abs_tol=spec['atol'])
    if kind == 'unordered':
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False
        # Match semantic values rather than repr(dict insertion order).
        remaining = list(b)
        for item in a:
            found = next((i for i, other in enumerate(remaining) if exact(item, other)), None)
            if found is None:
                return False
            remaining.pop(found)
        return not remaining
    if kind == 'tree':
        if 'fields' in spec:
            return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                    and all(compare(a[k], b[k], spec['fields'].get(k, {'type': 'exact'}), test) for k in b))
        children = spec['items']
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) == len(children)
                and all(compare(x, y, rule, test) for x, y, rule in zip(a, b, children)))
    if kind == 'maximum_geometry':
        rates, coordinates = test['args']
        if not rates:
            return exact(a, [])
        maximum = max(rates)
        return any(exact(a, coordinate) for rate, coordinate in zip(rates, coordinates) if rate == maximum)
    if kind == 'stdout_tokens_exact':
        return a.split() == b.split()
    if kind == 'stdout_tokens_casefold':
        return a.casefold().split() == b.casefold().split()
    if kind == 'stdout_lines_exact':
        return a.replace('\r\n', '\n').strip('\n') == b.replace('\r\n', '\n').strip('\n')
    if kind == 'constructive':
        return constructive(a, test['input'], spec['checker'])
    raise ValueError('Unsupported reviewed comparator: ' + kind)
