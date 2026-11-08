"""Subtractive sampling policy only. Never grants approval or modifies a release."""
import json
from pathlib import Path
from src.training.runtime_v2 import sha


def filter_training_pools(pools, manifest_path, policy_path):
    policy = json.loads(Path(policy_path).read_text(encoding='utf-8'))
    if policy.get('schema') != 'training-exclusions-v3' or policy.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('Training exclusion policy identity differs')
    if policy.get('review_overlay') and sha(policy['review_overlay']) != policy.get('review_overlay_sha256'):
        raise ValueError('Training exclusion evidence identity differs')
    # A single-domain configuration may contain only part of the release.
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    root = Path(manifest_path).resolve().parent
    release_ids = set()
    for name in manifest['approved_training_files']:
        release_ids.update(json.loads(l)['sample_id'] for l in (root/name).read_text(encoding='utf-8').splitlines() if l.strip())
    exclusions = set(policy['excluded_sample_ids'])
    if len(exclusions) != len(policy['excluded_sample_ids']) or not exclusions <= release_ids:
        raise ValueError('Unknown/duplicate excluded IDs')
    filtered = {d:[r for r in rows if r['sample_id'] not in exclusions] for d,rows in pools.items()}
    if any(not rows for rows in filtered.values()):
        raise ValueError('Exclusion policy empties a training domain')
    return filtered
