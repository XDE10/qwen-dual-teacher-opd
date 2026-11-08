"""Compact, shareable summary of an opd-v3 / sft-v3 run directory (read-only).

  python -m scripts.summarize_run_v3 --run-dir "$OPD_OUTPUT_ROOT/opd_v3_smoke"

Prints one JSON object: environment, per-step timing/memory/quality, and totals.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

GIB = 2 ** 30


def phase_totals(phases):
    seconds, peak = defaultdict(float), 0
    for record in phases:
        seconds[record['phase']] += record['seconds']
        peak = max(peak, record.get('peak_reserved') or 0, record.get('peak_allocated') or 0)
    return dict(seconds), peak


def summarize(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    identity_path = run_dir / 'run_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.exists() else {}
    soft = identity.get('soft', {})
    pool_path = run_dir / 'teacher_pool.json'
    steps, overall_peak, totals = [], 0, defaultdict(float)
    for r in rows:
        seconds, peak = phase_totals(r.get('phases', []))
        overall_peak = max(overall_peak, peak)
        for k, v in seconds.items():
            totals[k] += v
        step = {'step': r['step'], 'wall_s': round(r.get('total_wall_seconds', 0), 1), 'peak_gib': round(peak / GIB, 2),
                'loss': r.get('loss', r.get('loss_token_mean_nll')), 'grad_norm': r.get('gradient_norm'),
                'cosine_math_code': r.get('gradient_cosine_math_code'), 'lr': r.get('learning_rate'),
                'phase_s': {k: round(v, 1) for k, v in sorted(seconds.items())}}
        for d, s in (r.get('domains') or {}).items():
            step[d] = {'kl': round(s.get('kl_exact_token_mean', float('nan')), 4),
                       'student_entropy': round(s.get('student_entropy_token_mean', float('nan')), 3),
                       'teacher_entropy': round(s.get('teacher_entropy_token_mean', float('nan')), 3),
                       'eos_rate': s.get('eos_rate'), 'mean_response_tokens': round(s.get('mean_response_tokens', 0), 1),
                       'removed_mass_max': (s.get('removed_mass_teacher') or {}).get('max')}
        steps.append(step)
    return {'run_dir': run_dir.as_posix(), 'steps': len(rows),
            'gpu': (soft.get('runtime') or {}).get('gpu'), 'gpu_total_gib': round(((soft.get('runtime') or {}).get('gpu_total_bytes') or 0) / GIB, 1),
            'torch_cuda': (soft.get('runtime') or {}).get('torch_cuda'),
            'teacher_pool': json.loads(pool_path.read_text(encoding='utf-8')) if pool_path.exists() else None,
            'overall_peak_gib': round(overall_peak / GIB, 2), 'phase_seconds_total': {k: round(v, 1) for k, v in sorted(totals.items())},
            'per_step': steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    print(json.dumps(summarize(parser.parse_args().run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
