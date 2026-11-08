"""Teacher-fit probe: how far is each candidate teacher from the student, before training?

MOPD (arXiv 2606.30406, Sec. 4.4.2) reports that a stronger but non-same-origin
teacher raised the initial per-token reverse KL from about 0.04 to about 0.19
and destabilised distillation. This probe measures the same quantity for our
off-the-shelf teachers on the untrained student's own rollouts.

For every (teacher, domain) cell it reports the token-mean exact reverse KL
on the support, student and teacher entropy, sampled-token log-probabilities,
the share of student tokens the teacher considers very unlikely, KL by
relative response position, and the probability mass outside the support.
No parameters change.
"""
import gc
import math
from collections import Counter

from src.losses.mopd_v3 import STAT, microbatch_loss
from src.training.opd_v3 import DomainAccumulator, pack_by_tokens, right_pad

THRESHOLDS = {'teacher_prob_below_1e-2': math.log(1e-2), 'teacher_prob_below_1e-4': math.log(1e-4),
              'teacher_prob_below_1e-6': math.log(1e-6)}


def score_teacher(student, teacher, rollouts, support, pad_id, device, microbatch_tokens):
    import torch
    out = {}
    for domain, items in rollouts.items():
        accumulator, low, rows = DomainAccumulator(), Counter(), 0
        for group in pack_by_tokens([len(x['prompt_ids']) + len(x['response_ids']) for x in items], microbatch_tokens):
            chosen = [items[i] for i in group]
            ids, attention = right_pad([x['prompt_ids'] + x['response_ids'] for x in chosen], pad_id, device)
            weights = torch.ones(len(chosen), dtype=torch.float32, device=device)
            with torch.no_grad():
                _, detail = microbatch_loss(student, teacher, ids, attention, [len(x['prompt_ids']) for x in chosen],
                                            [x['prompt_ids'] for x in chosen], support, weights, {'mode': 'exact_rkl'})
            accumulator.add(detail, [len(x['prompt_ids']) for x in chosen], [len(x['response_ids']) for x in chosen])
            teacher_logp = detail['stats'][:, STAT['teacher_logp']].float().cpu()
            rows += teacher_logp.numel()
            for name, threshold in THRESHOLDS.items():
                low[name] += int((teacher_logp < threshold).sum())
        out[domain] = {**accumulator.summary(), **{k: v / max(rows, 1) for k, v in low.items()},
                       'rollouts': len(items), 'eos_rate': sum(x['ended_with_eos'] for x in items) / len(items),
                       'mean_response_tokens': sum(len(x['response_ids']) for x in items) / len(items)}
    return out


def probe(student, teacher_loaders, rollouts, support, pad_id, device, microbatch_tokens=16384):
    """``teacher_loaders``: {name: zero-arg callable returning a frozen teacher}. Loads one at a time."""
    results = {}
    for name, loader in teacher_loaders.items():
        teacher = loader()
        teacher.eval().requires_grad_(False)
        results[name] = score_teacher(student, teacher, rollouts, support, pad_id, device, microbatch_tokens)
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
