"""Compact, shareable summary of an opd-v3 / sft-v3 run directory (read-only).

  python -m scripts.summarize_run_v3 --run-dir "$OPD_OUTPUT_ROOT/opd_v3_smoke"

Prints one JSON object: environment, per-step timing/memory/quality, totals, and for opd-v3 runs with
at least 6 steps a stability gate per domain. The baseline is the first step in which the domain appears
(the untrained student, or the warm-start adapter; runs with the same seed and data share it exactly),
and the window is the last min(10, steps // 3) steps:
  * KL to the teacher must be below the baseline;
  * entropy_vs_teacher: student entropy divided by the teacher's entropy on the same rollouts must stay
    within 0.7x-1.5x of its baseline value. This is the mode-seeking check. When the student's rollouts
    move to text that both models predict confidently, both entropies fall together and the check holds;
    it fails when the student sharpens or flattens relative to the teacher;
  * mean response length must stay within 0.7x-1.3x of the baseline, and the EOS rate may drop by at most 0.10;
  * math rollout accuracy (Math-Verify) may drop by at most 0.08, first window vs last window (per-step
    accuracy is too noisy for a single-step baseline);
  * at most one step whose gradient norm exceeds 10x the median of the steps before it.
A student entropy below 0.5x or above 2x of the baseline is reported as a warning, not a failure.

`gate_v1_first_window` keeps the first gate (2026-11-09, used for Session 2's summary.json files):
everything compared first window vs last window, with absolute student entropy as a hard check. For a
fast learning rate the first window already contains most of the change, so its ratios understate drift;
see reports/g4_session2_lr_sweep_20261109/README.md.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

GIB = 2 ** 30
GATE = {'baseline': 'first step of the domain', 'kl_ratio_below': 1.0, 'entropy_vs_teacher_range': [0.7, 1.5],
        'length_ratio_range': [0.7, 1.3], 'eos_rate_drop_max': 0.10, 'math_accuracy_drop_max': 0.08,
        'student_entropy_warning_range': [0.5, 2.0], 'grad_spike_factor': 10.0, 'grad_spikes_max': 1,
        'window_max': 10, 'min_steps': 6}
GATE_V1 = {'kl_ratio_below': 1.0, 'entropy_ratio_range': [0.7, 1.5], 'length_ratio_range': [0.7, 1.3],
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


def ratio(first, last):
    return last / first if first else None


def domain_rows(rows, domain):
    return [r for r in rows if domain in (r.get('domains') or {})]


def is_opd(rows):
    return any('kl_exact_token_mean' in d for r in rows for d in (r.get('domains') or {}).values())


def grad_spikes(rows, factor):
    norms = [r.get('gradient_norm') for r in rows]
    return [r['step'] for i, r in enumerate(rows) if i >= 3 and norms[i] is not None and
            norms[i] > factor * statistics.median(n for n in norms[:i] if n is not None)]


def entropy_ratio(stats):
    teacher = stats.get('teacher_entropy_token_mean')
    return stats['student_entropy_token_mean'] / teacher if teacher else None


def last_mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def gate(rows):
    """Stability gate against the domain's first step; see the module docstring."""
    steps = len(rows)
    if steps < GATE['min_steps'] or not is_opd(rows):
        return {'pass': None, 'reason': f'needs an opd-v3 run with at least {GATE["min_steps"]} steps'}
    checks, outcome, warnings = {}, {}, []
    for domain in sorted({d for r in rows for d in (r.get('domains') or {})}):
        own = domain_rows(rows, domain)
        window = max(1, min(GATE['window_max'], len(own) // 3))
        base, late = own[0]['domains'][domain], [r['domains'][domain] for r in own[-window:]]
        early = [r['domains'][domain] for r in own[:window]]
        cell = {'window': window, 'baseline_step': own[0]['step']}

        def against_base(name, key, test):
            first, last = base.get(key), last_mean(s.get(key) for s in late)
            if first is None or last is None:
                cell[name] = {'pass': None, 'reason': 'not recorded'}
                return
            ok, detail = test(first, last)
            cell[name] = {'baseline': round(first, 4), 'last': round(last, 4), **detail, 'pass': ok}

        against_base('kl', 'kl_exact_token_mean',
                     lambda f, l: (l / f < GATE['kl_ratio_below'], {'ratio': round(l / f, 3)}))
        first_ratio, last_ratio = entropy_ratio(base), last_mean(entropy_ratio(s) for s in late)
        if first_ratio and last_ratio is not None:
            low, high = GATE['entropy_vs_teacher_range']
            relative = last_ratio / first_ratio
            cell['entropy_vs_teacher'] = {'baseline': round(first_ratio, 4), 'last': round(last_ratio, 4),
                                          'ratio': round(relative, 3), 'pass': low <= relative <= high}
        else:
            cell['entropy_vs_teacher'] = {'pass': None, 'reason': 'teacher entropy not recorded'}
        low, high = GATE['length_ratio_range']
        against_base('mean_response_tokens', 'mean_response_tokens',
                     lambda f, l: (low <= l / f <= high, {'ratio': round(l / f, 3)}))
        against_base('eos_rate', 'eos_rate',
                     lambda f, l: (f - l <= GATE['eos_rate_drop_max'], {'drop': round(f - l, 4)}))
        if any(s.get('rollout_accuracy_math_verify') is not None for s in early + late) and len(own) >= 2 * window:
            first, last = (last_mean(s.get('rollout_accuracy_math_verify') for s in part) for part in (early, late))
            cell['rollout_accuracy_math_verify'] = {
                'first_window': round(first, 4), 'last': round(last, 4), 'drop': round(first - last, 4),
                'pass': first - last <= GATE['math_accuracy_drop_max']}
        entropy_first = base['student_entropy_token_mean']
        entropy_last = last_mean(s['student_entropy_token_mean'] for s in late)
        entropy = {'baseline': round(entropy_first, 4), 'last': round(entropy_last, 4),
                   'ratio': round(entropy_last / entropy_first, 3)}
        low, high = GATE['student_entropy_warning_range']
        if not low <= entropy['ratio'] <= high:
            warnings.append(f'{domain}.student_entropy ratio {entropy["ratio"]} outside {low}-{high} (information only)')
        cell['student_entropy'] = entropy
        checks[domain] = cell
        outcome.update({f'{domain}.{k}': v['pass'] for k, v in cell.items() if isinstance(v, dict) and 'pass' in v})
    spikes = grad_spikes(rows, GATE['grad_spike_factor'])
    outcome['gradient_norm'] = len(spikes) <= GATE['grad_spikes_max']
    checks['gradient_norm'] = {'spike_steps': spikes, 'pass': outcome['gradient_norm']}
    failed = [name for name, ok in outcome.items() if ok is False]
    undecided = [name for name, ok in outcome.items() if ok is None]
    return {'version': 2, 'steps': steps, 'thresholds': GATE, 'checks': checks, 'failed': failed,
            'undecided': undecided, 'warnings': warnings, 'pass': False if failed else None if undecided else True}


def window_check(values, window, test):
    values = [v for v in values if v is not None]
    if len(values) < 2 * window:
        return {'pass': None, 'reason': 'not enough values'}
    first, last = statistics.fmean(values[:window]), statistics.fmean(values[-window:])
    ok, detail = test(first, last)
    return {'first': round(first, 4), 'last': round(last, 4), **detail, 'pass': ok}


def gate_v1_first_window(rows):
    """The first gate: first window vs last window throughout (kept to reproduce Session 2's summaries)."""
    steps = len(rows)
    if steps < GATE_V1['min_steps'] or not is_opd(rows):
        return {'pass': None, 'reason': f'needs an opd-v3 run with at least {GATE_V1["min_steps"]} steps'}
    low, high = GATE_V1['entropy_ratio_range']
    length_low, length_high = GATE_V1['length_ratio_range']
    checks = {}
    for domain in sorted({d for r in rows for d in (r.get('domains') or {})}):
        own = domain_rows(rows, domain)
        window = max(1, min(GATE_V1['window_max'], len(own) // 3))
        series = lambda key: [r['domains'][domain].get(key) for r in own]

        def within(lo, hi):
            def test(first, last):
                value = ratio(first, last)
                return (value is not None and lo <= value <= hi), {'ratio': rounded(value, 3)}
            return test

        cell = {'window': window,
                'kl': window_check(series('kl_exact_token_mean'), window,
                                   lambda f, l: ((ratio(f, l) or 0) < GATE_V1['kl_ratio_below'], {'ratio': rounded(ratio(f, l), 3)})),
                'student_entropy': window_check(series('student_entropy_token_mean'), window, within(low, high)),
                'mean_response_tokens': window_check(series('mean_response_tokens'), window, within(length_low, length_high)),
                'eos_rate': window_check(series('eos_rate'), window,
                                         lambda f, l: (f - l <= GATE_V1['eos_rate_drop_max'], {'drop': round(f - l, 4)}))}
        accuracy = series('rollout_accuracy_math_verify')
        if any(v is not None for v in accuracy):
            cell['rollout_accuracy_math_verify'] = window_check(
                accuracy, window, lambda f, l: (f - l <= GATE_V1['math_accuracy_drop_max'], {'drop': round(f - l, 4)}))
        checks[domain] = cell
    spikes = grad_spikes(rows, GATE_V1['grad_spike_factor'])
    outcome = {f'{domain}.{key}': value['pass'] for domain, cell in checks.items()
               for key, value in cell.items() if isinstance(value, dict)}
    outcome['gradient_norm'] = len(spikes) <= GATE_V1['grad_spikes_max']
    checks['gradient_norm'] = {'spike_steps': spikes, 'pass': outcome['gradient_norm']}
    failed = [name for name, ok in outcome.items() if ok is False]
    undecided = [name for name, ok in outcome.items() if ok is None]
    return {'steps': steps, 'thresholds': GATE_V1, 'checks': checks, 'failed': failed, 'undecided': undecided,
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
            'gate': gate(rows), 'gate_v1_first_window': gate_v1_first_window(rows), 'per_step': steps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    print(json.dumps(summarize(parser.parse_args().run_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
