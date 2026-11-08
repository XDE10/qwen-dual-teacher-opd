"""Data-bound review sidecars; blocked decisions remain visible in denominator."""
import hashlib
import json


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def validate_registry(rows, registry, data_sha=None):
    if set(registry) != {r['sample_id'] for r in rows}:
        raise ValueError('Reviewed registry task set differs from data')
    for row in rows:
        spec = registry[row['sample_id']]
        if spec.get('protocol_version') != 'review-v3':
            raise ValueError('Missing review-v3 schema')
        if spec.get('status') not in {'reviewed', 'blocked_after_review'}:
            raise ValueError('Review decision incomplete')
        if spec.get('row_sha256') != row_sha(row) or (data_sha is not None and spec.get('data_sha256') != data_sha):
            raise ValueError('Reviewed comparator data identity changed')
        if not spec.get('review_reason') or not spec.get('evidence_quote'):
            raise ValueError('Missing per-task review evidence')


def make_reviewed_job(row, candidate, spec):
    if spec.get('row_sha256') != row_sha(row):
        raise ValueError('Reviewed comparator row identity changed')
    if spec.get('status') == 'blocked_after_review':
        raise ValueError('Blocked after review: ' + spec['review_reason'])
    if spec.get('status') != 'reviewed':
        raise ValueError('Comparator review is incomplete')
    allowed = {'exact', 'float', 'unordered', 'tree', 'maximum_geometry',
               'stdout_tokens_exact', 'stdout_tokens_casefold', 'stdout_lines_exact', 'constructive'}
    if spec['type'] not in allowed:
        raise ValueError('Unsupported comparator')
    if spec['type'] == 'constructive' and spec.get('checker') not in {'cycle_walk', 'zero_sum_subset', 'equal_candy_groups', 'equal_leg_pairs'}:
        raise ValueError('Dedicated checker is not registered')
    # Reuse sealed extraction/signature checks without recursive v3 dispatch.
    from src.evaluation.code_v2 import make_job
    temporary = {**spec, 'type': 'exact'}
    temporary.pop('protocol_version')
    job = make_job(row, candidate, temporary)
    job['comparator'] = spec
    return job
