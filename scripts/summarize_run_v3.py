"""Compact, shareable summary of an opd-v3 / sft-v3 run directory (read-only).

  python -m scripts.summarize_run_v3 --run-dir "$OPD_OUTPUT_ROOT/opd_v3_smoke"

Prints one JSON object: environment, per-step timing/memory/quality, totals, and for opd-v3
runs with at least 6 steps a stability gate that compares the first and the last window of
steps (window = min(10, steps // 3)) per domain:
  * KL to the teacher must go down;
  * student entropy must stay within 0.7x-1.5x;
  * mean response length must stay within 0.7x-1.3x, and the EOS rate may drop by at most 0.10;
  * math rollout accuracy (Math-Verify) may drop by at most 0.08;
  * at most one step whose gradient norm exceeds 10x the median of the steps before it.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

GIB = 2 ** 30
GATE = {'kl_ratio_below': 1.0, 'entropy_ratio_range': [0.7, 1.5], 'length_ratio_range': [0.7, 1.3],
        'eos_rate_drop_max': 0.10, 'math_accuracy_drop_max': 0.08, 'grad_spike_factor': 10.0, 'grad_spikes_max': 1,
        'window_max': 10, 'min_steps': 6}


def phase_totals(phases):
    seconds, peak = defaultdict(float), 0
    for record in phases:
        seconds[record['phase']] += record['seconds']
        peak = max(peak, record.get('peak_reserved') or 0, record.get('peak_allocated') or 0)
    return dict(seconds), peak


def rounded(value, digits):
    return None if value is None else round(value, digits)


def domain_series(rows, domain, key):
    return [r['domains'][domain].get(key) for r in rows if domain in (r.get('domains') or {})]


def window_check(values, window, test):
    values = [v for v in values if v is not None]
    if len(values) < 2 * window:
        return {'pass': None, 'reason': 'not enough values'}
    first, last = statistics.fmean(values[:window]), statistics.fmean(values[-window:])
    ok, detail = test(first, last)
    return {'first': round(first, 4), 'last': round(last, 4), **detail, 'pass': ok}


def ratio(first, last):
    return last / first if first else None


def gate(rows):
    """Stability gate over the first and last window of steps; see the module docstring."""
    steps = len(rows)
    if steps < GATE['min_steps'] or not any('kl_exact_token_mean' in d for r in rows for d in r.get('domains', {}).values()):
        return {'pass': None, 'reason': f'needs an opd-v3 run with at least {GATE["min_steps"]} steps'}
    low, high = GATE['entropy_ratio_range']
    length_low, length_high = GATE['length_ratio_range']
    checks, window = {}, None
    for domain in sorted({d for r in rows for d in (r.get('domains') or {})}):
        count = sum(1 for r in rows if domain in r['domains'])
        window = max(1, min(GATE['window_max'], count // 3))
        series = lambda key: domain_series(rows, domain, key)

        def within(lo, hi):
            def test(first, last):
                value = ratio(first, last)
                return (value is not None and lo <= value <= hi), {'ratio': rounded(value, 3)}
            return test

        cell = {'window': window,
                'kl': window_check(series('kl_exact_token_mean'), window,
                                   lambda f, l: ((ratio(f, l) or 0) < GATE['kl_ratio_below'], {'ratio': rounded(ratio(f, l), 3)})),
                'student_entropy': window_check(series('student_entropy_token_mean'), window, within(low, high)),
                'mean_response_tokens': window_check(series('mean_response_tokens'), window, within(length_low, length_high)),
                'eos_rate': window_check(series('eos_rate'), window,
                                         lambda f, l: (f - l <= GATE['eos_rate_drop_max'], {'drop': round(f - l, 4)}))}
        accuracy = series('rollout_accuracy_math_verify')
        if any(v is not None for v in accuracy):
            cell['rollout_accuracy_math_verify'] = window_check(
                accuracy, window, lambda f, l: (f - l <= GATE['math_accuracy_drop_max'], {'drop': round(f - l, 4)}))
        checks[domain] = cell
    norms = [r.get('gradient_norm') for r in rows]
    spikes = [r['step'] for i, r in enumerate(rows) if i >= 3 and norms[i] is not None and
              norms[i] > GATE['grad_spike_factor'] * statistics.median(n for n in norms[:i] if n is not None)]
    outcome = {f'{domain}.{key}': value['pass'] for domain, cell in checks.items()
               for key, value in cell.items() if isinstance(value, dict)}
    outcome['gradient_norm'] = len(spikes) <= GATE['grad_spikes_max']
    checks['gradient_norm'] = {'spike_steps': spikes, 'pass': outcome['gradient_norm']}
    failed = [name for name, ok in outcome.items() if ok is False]
    undecided = [name for name, ok in outcome.items() if ok is None]
    return {'steps': steps, 'thresholds': GATE, 'checks': checks, 'failed': failed, 'undecided': undecided,
            'pass': False if failed else None if undecided else True}


def summarize(run_dir):
    run_dir = Path(run_dir)
    rows = [json.loads(line) for line in (run_dir / 'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    identity_path = run_dir / 'run_identity.json'
    identity = json.loads(identity_path.read_text(encoding='utf-8')) if identity_path.exists() else {}
    soft = identity.get('soft', {})
    config = (identity.get('hard') or {}).get('config') or {}
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
            if s.get('rollouts_graded'):
                step[d]['rollout_accuracy_math_verify'] = rounded(s.get('rollout_accuracy_math_verify'), 4)
                step[d]['rollout_accuracy_conservative'] = rounded(s.get('rollout_accuracy_conservative'), 4)
        steps.append(step)
    walls = [r.get('total_wall_seconds') for r in rows if r.get('total_wall_seconds')]
    return {'run_dir': run_dir.as_posix(), 'steps': len(rows), 'max_steps': config.get('max_steps'),
            'learning_rate': config.get('learning_rate'), 'prompts_per_step': config.get('prompts_per_step'),
            'gpu': (soft.get('runtime') or {}).get('gpu'), 'gpu_total_gib': round(((soft.get('runtime') or {}).get('gpu_total_bytes') or 0) / GIB, 1),
            'torch_cuda': (soft.get('runtime') or {}).get('torch_cuda'),
            'teacher_pool': json.loads(pool_path.read_text(encoding='utf-8')) if pool_path.exists() else None,
            'overall_peak_gib': round(overall_peak / GIB, 2), 'phase_seconds_total': {k: round(v, 1) for k, v in sorted(totals.items())},
            'median_step_seconds': round(statistics.median(walls), 1) if walls else None,
            'gate': gate(rows), 'per_step': steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    print(json.dumps(summarize(parser.parse_args().run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
