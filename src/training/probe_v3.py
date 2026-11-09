"""Teacher-fit probe: how far is each candidate teacher from the student, before training?

MOPD (arXiv 2606.30406, Sec. 4.4.2) reports that a stronger but non-same-origin
teacher raised the initial per-token reverse KL from about 0.04 to about 0.19
and destabilised distillation. This probe measures the same quantity for our
off-the-shelf teachers on the untrained student's own rollouts.

For every (teacher, domain) cell it reports the token-mean exact reverse KL
on the support, student and teacher entropy, sampled-token log-probabilities,
the share of student tokens the teacher considers very unlikely, KL by
relative response position, the probability mass outside the support, and the
sampled tokens that carry the most KL. With ``dump_rollouts=K`` it also keeps
per-token values for the first K rollouts of each domain. No parameters change.
"""
import gc
import math
from collections import Counter

from src.losses.mopd_v3 import STAT, microbatch_loss
from src.training.opd_v3 import DomainAccumulator, pack_by_tokens, right_pad

THRESHOLDS = {'teacher_prob_below_1e-2': math.log(1e-2), 'teacher_prob_below_1e-4': math.log(1e-4),
              'teacher_prob_below_1e-6': math.log(1e-6)}
TOKEN_FIELDS = ('kl_exact', 'student_logp', 'teacher_logp', 'student_entropy', 'teacher_entropy')
TOP_TOKENS = 40


def top_kl_tokens(sums, total_kl, limit=TOP_TOKENS):
    """Sampled tokens with the largest summed KL: [token_id, count, share of all KL, mean KL,
    mean student entropy, mean teacher entropy]. The KL of a row belongs to the distribution
    that produced the sampled token, so this shows where the student and teacher disagree."""
    count, kl, student_entropy, teacher_entropy = sums
    order = kl.argsort(descending=True)[:limit]
    return [[int(i), int(count[i]), round(float(kl[i] / total_kl), 5) if total_kl > 0 else None,
             round(float(kl[i] / count[i]), 4), round(float(student_entropy[i] / count[i]), 4),
             round(float(teacher_entropy[i] / count[i]), 4)]
            for i in order.tolist() if count[i] > 0]


def score_teacher(student, teacher, rollouts, support, pad_id, device, microbatch_tokens, dump_rollouts=0):
    import torch
    out = {}
    for domain, items in rollouts.items():
        accumulator, low, rows = DomainAccumulator(), Counter(), 0
        sums = torch.zeros(4, support, dtype=torch.float64)  # count, KL, student entropy, teacher entropy
        dumped = {}
        for group in pack_by_tokens([len(x['prompt_ids']) + len(x['response_ids']) for x in items], microbatch_tokens):
            chosen = [items[i] for i in group]
            ids, attention = right_pad([x['prompt_ids'] + x['response_ids'] for x in chosen], pad_id, device)
            weights = torch.ones(len(chosen), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, detail = microbatch_loss(student, teacher, ids, attention, [len(x['prompt_ids']) for x in chosen],
                                            [x['prompt_ids'] for x in chosen], support, weights, {'mode': 'exact_rkl'})
            accumulator.add(detail, [len(x['prompt_ids']) for x in chosen], [len(x['response_ids']) for x in chosen])
            stats = detail['stats'].double().cpu()
            b, positions = detail['batch_index'].cpu(), detail['positions'].cpu()
            targets = ids[detail['batch_index'], detail['positions'] + 1].cpu()
            teacher_logp = stats[:, STAT['teacher_logp']]
            rows += teacher_logp.numel()
            for name, threshold in THRESHOLDS.items():
                low[name] += int((teacher_logp < threshold).sum())
            sums[0].index_add_(0, targets, torch.ones_like(teacher_logp))
            for k, name in enumerate(('kl_exact', 'student_entropy', 'teacher_entropy'), start=1):
                sums[k].index_add_(0, targets, stats[:, STAT[name]])
            if dump_rollouts:
                original = torch.tensor(group)[b]
                prompt = torch.tensor([len(x['prompt_ids']) for x in chosen])[b]
                for r in (original < dump_rollouts).nonzero().flatten().tolist():
                    dumped.setdefault(int(original[r]), []).append(
                        [int(positions[r] + 1 - prompt[r]), int(targets[r]),
                         *[round(float(stats[r, STAT[f]]), 4) for f in TOKEN_FIELDS]])
        cell = {**accumulator.summary(), **{k: v / max(rows, 1) for k, v in low.items()},
                'rollouts': len(items), 'eos_rate': sum(x['ended_with_eos'] for x in items) / len(items),
                'mean_response_tokens': sum(len(x['response_ids']) for x in items) / len(items),
                'top_kl_tokens': top_kl_tokens(sums, float(sums[1].sum()))}
        if dump_rollouts:
            cell['token_dump'] = [{'rollout': i, 'fields': ['response_position', 'token_id', *TOKEN_FIELDS],
                                   'tokens': sorted(dumped[i])} for i in sorted(dumped)]
        out[domain] = cell
    return out


def probe(student, teacher_loaders, rollouts, support, pad_id, device, microbatch_tokens=16384, dump_rollouts=0):
    """``teacher_loaders``: {name: zero-arg callable returning a frozen teacher}. Loads one at a time."""
    results = {}
    for name, loader in teacher_loaders.items():
        teacher = loader()
        teacher.eval().requires_grad_(False)
        results[name] = score_teacher(student, teacher, rollouts, support, pad_id, device, microbatch_tokens,
                                      dump_rollouts)
        del teacher
        gc.collect()
        if str(device).startswith('cuda'):
            import torch
            torch.cuda.empty_cache()
    return results


def verdict(results, same_origin_kl=0.04, mopd_external_kl=0.19):
    """Plain-language reading of the KL cells against MOPD's two reference points."""
    lines = []
    for name, cells in results.items():
        for domain, cell in cells.items():
            kl = cell['kl_exact_token_mean']
            level = ('close to MOPD same-origin' if kl <= 2 * same_origin_kl else
                     'similar to MOPD external-teacher case' if kl <= 2 * mopd_external_kl else
                     'larger than MOPD external-teacher case')
            lines.append(f'{name} on {domain}: KL {kl:.3f} nats/token ({level})')
    return lines
