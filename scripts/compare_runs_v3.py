"""Compare opd-v3 runs from their metrics.jsonl (read-only, stdlib only).

  python3 -m scripts.compare_runs_v3 --run 3e-5=runs/opd_v3_d4_dual --run 1e-4=runs/opd_v3_d4_dual_lr1e-4 \
      [--steps 21-30] [--json comparison.json]

For each run and domain it prints the baseline (the domain's first step) and the mean of the last window
of steps: KL to the teacher, student and teacher entropy on the same rollouts and their ratio, mean
response length, EOS rate, and math rollout accuracy (Math-Verify and the conservative verifier).

Runs with the same seed and data see the same prompts in the same order, so math accuracy can be compared
pair by pair: a rollout of run X and a rollout of run Y are paired when they answer the same prompt for
the same time (key: sample ID and how often the run has seen it). The step range selects the pairs by the
first run's step numbers; the default is the last window of the steps all runs have. Each pair of runs
gets the discordant counts and an exact two-sided McNemar p-value. Accuracy here is sampled at T=1 on
training prompts; it is training telemetry, not an evaluation result.
"""
import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

DOMAIN_KEYS = (('kl', 'kl_exact_token_mean'), ('student_entropy', 'student_entropy_token_mean'),
               ('teacher_entropy', 'teacher_entropy_token_mean'), ('mean_response_tokens', 'mean_response_tokens'),
               ('eos_rate', 'eos_rate'), ('accuracy_math_verify', 'rollout_accuracy_math_verify'),
               ('accuracy_conservative', 'rollout_accuracy_conservative'))


def load(run_dir):
    path = Path(run_dir) / 'metrics.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'{path} has no steps')
    return rows


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def domain_summary(rows, domain, window):
    own = [r for r in rows if domain in (r.get('domains') or {})]
    if not own:
        return None
    window = max(1, min(window, len(own)))
    base, late = own[0]['domains'][domain], [r['domains'][domain] for r in own[-window:]]
    out = {'baseline_step': own[0]['step'], 'window_steps': [own[-window]['step'], own[-1]['step']]}
    for name, key in DOMAIN_KEYS:
        first, last = base.get(key), mean(s.get(key) for s in late)
        if first is None and last is None:
            continue
        out[name] = {'baseline': first, 'last': last}
    ratios = [s['student_entropy_token_mean'] / s['teacher_entropy_token_mean'] for s in late
              if s.get('teacher_entropy_token_mean')]
    if base.get('teacher_entropy_token_mean') and ratios:
        out['entropy_student_over_teacher'] = {
            'baseline': base['student_entropy_token_mean'] / base['teacher_entropy_token_mean'], 'last': mean(ratios)}
    return out


def keyed_math(rows):
    """{(sample_id, occurrence): (step, math_verify, conservative)} for graded math rollouts."""
    seen, out = Counter(), {}
    for r in rows:
        for s in r.get('samples', []):
            if s.get('domain') != 'math':
                continue
            seen[s['sample_id']] += 1
            if 'correct' in s:
                out[(s['sample_id'], seen[s['sample_id']])] = (r['step'], s['correct'].get('math_verify'),
                                                               s['correct'].get('conservative'))
    return out


def mcnemar(only_x, only_y):
    n, k = only_x + only_y, min(only_x, only_y)
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(runs, low, high):
    keyed = {name: keyed_math(rows) for name, rows in runs.items()}
    names = list(runs)
    reference = keyed[names[0]]
    selected = [k for k, v in reference.items() if low <= v[0] <= high]
    out = []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            for index, grader in ((1, 'math_verify'), (2, 'conservative')):
                pairs = [(keyed[x][k][index], keyed[y][k][index]) for k in selected
                         if k in keyed[x] and k in keyed[y] and keyed[x][k][index] is not None
                         and keyed[y][k][index] is not None]
                if not pairs:
                    continue
                only_x = sum(1 for a, b in pairs if a and not b)
                only_y = sum(1 for a, b in pairs if b and not a)
                out.append({'runs': [x, y], 'grader': grader, 'pairs': len(pairs),
                            'accuracy': [sum(a for a, _ in pairs) / len(pairs), sum(b for _, b in pairs) / len(pairs)],
                            'only_first_correct': only_x, 'only_second_correct': only_y,
                            'difference': (only_x - only_y) / len(pairs), 'mcnemar_p': mcnemar(only_x, only_y)})
    return out


def parse_steps(text):
    low, _, high = text.partition('-')
    return int(low), int(high or low)


def fmt(value, digits=4):
    return '-' if value is None else f'{value:.{digits}f}'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', action='append', required=True, metavar='NAME=RUN_DIR')
    p.add_argument('--window', type=int, default=10, help='last-window length in steps (default 10)')
    p.add_argument('--steps', help='step range LO-HI of the first run for the paired comparison')
    p.add_argument('--json', help='also write the comparison to this file')
    a = p.parse_args(argv)
    runs = {}
    for spec in a.run:
        name, sep, path = spec.partition('=')
        if not sep or not name or name in runs:
            p.error(f'--run needs a unique NAME=RUN_DIR, got {spec!r}')
        runs[name] = load(path)
    common = min(len(rows) for rows in runs.values())
    window = max(1, min(a.window, common))
    low, high = parse_steps(a.steps) if a.steps else (runs[next(iter(runs))][common - window]['step'],
                                                      runs[next(iter(runs))][common - 1]['step'])
    domains = sorted({d for rows in runs.values() for r in rows for d in (r.get('domains') or {})})
    report = {'runs': {}, 'paired_math_accuracy': {'steps': [low, high], 'results': paired(runs, low, high)},
              'shared_baseline': {}}
    for name, rows in runs.items():
        report['runs'][name] = {'steps': len(rows), 'last_learning_rate': rows[-1].get('learning_rate'),
                                'late_gradient_cosine': mean(r.get('gradient_cosine_math_code') for r in rows[-window:]),
                                'domains': {d: domain_summary(rows, d, a.window) for d in domains}}
    for d in domains:
        firsts = [json.dumps(next((r['domains'][d] for r in rows if d in (r.get('domains') or {})), None), sort_keys=True)
                  for rows in runs.values()]
        report['shared_baseline'][d] = len(set(firsts)) == 1
    for d in domains:
        print(f'== {d} (baseline = first step; last = mean of the last {a.window} steps of each run)')
        print(f'   {"run":12s} {"steps":>5s} {"KL":>15s} {"H_student":>15s} {"H_teacher":>15s} {"H_s/H_t":>11s} '
              f'{"length":>13s} {"EOS":>11s} {"acc_MV":>11s}')
        for name, info in report['runs'].items():
            s = info['domains'][d]
            if not s:
                continue
            cell = lambda key, digits=3: (f'{fmt(s[key]["baseline"], digits)}>{fmt(s[key]["last"], digits)}'
                                          if key in s else '-')
            print(f'   {name:12s} {info["steps"]:5d} {cell("kl"):>15s} {cell("student_entropy"):>15s} '
                  f'{cell("teacher_entropy"):>15s} {cell("entropy_student_over_teacher", 2):>11s} '
                  f'{cell("mean_response_tokens", 0):>13s} {cell("eos_rate", 2):>11s} {cell("accuracy_math_verify", 2):>11s}')
        print(f'   identical baseline across runs: {report["shared_baseline"][d]}')
    print(f'== paired math accuracy on the first run\'s steps {low}-{high} (T=1 training rollouts)')
    for item in report['paired_math_accuracy']['results']:
        x, y = item['runs']
        print(f'   {x} vs {y} [{item["grader"]}]: n={item["pairs"]}, {item["accuracy"][0]:.3f} vs {item["accuracy"][1]:.3f}, '
              f'only {x} {item["only_first_correct"]} / only {y} {item["only_second_correct"]}, '
              f'diff {item["difference"]:+.4f}, McNemar p={item["mcnemar_p"]:.3f}')
    if a.json:
        Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    main()
