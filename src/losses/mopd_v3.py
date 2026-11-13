"""MOPD-style per-token reverse-KL losses on response rows only (opd-v3).

Logits are computed only for rows that predict a response token, and only for
the audited common support: the lm_head weight is sliced to its first
``support`` rows, so ``log_softmax`` of these logits is already the
renormalised common-support distribution of the prompt contract.

Three instantiations of the same objective (MOPD, arXiv 2606.30406, Sec. 3.2):

* ``exact_rkl``: full-support reverse KL, sum_v p(v) [log p(v) - log q(v)].
  This is the K=V case, with no sampling or truncation variance.
* ``topk_rkl``: MOPD Eq. 5 on the teacher's top-k tokens, including the
  ``- p + q`` correction that keeps the optimum at p = q on the kept tokens.
* ``pg``: MOPD Eq. 4, policy gradient with the per-token advantage
  sg[log q(y) - log p(y)], clipped to [-c, c].

``proximal_teacher_alpha`` < 1 replaces q with alpha*q + (1-alpha)*sg(p).
This is the trust-region teacher of NeMo RL (TROPD); alpha = 1 is plain MOPD.

All three forms are myopic token-level objectives. The prefix (state)
distribution is not differentiated, as in GKD, MiniLLM single-step and MOPD.
"""
import math

import torch
import torch.nn.functional as F

from src.teachers.opd_contract import teacher_batch_from_student

MODES = ('exact_rkl', 'topk_rkl', 'pg')
STAT_FIELDS = ('kl_exact', 'student_logp', 'teacher_logp', 'student_entropy', 'teacher_entropy',
               'student_lse', 'teacher_lse')
STAT = {name: i for i, name in enumerate(STAT_FIELDS)}


def unwrap(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def decoder_and_head(model):
    base = unwrap(model)
    decoder = base.get_decoder() if hasattr(base, 'get_decoder') else getattr(base, 'model', None)
    head = base.get_output_embeddings()
    if decoder is None or head is None:
        raise ValueError('Model lacks decoder / output-embedding accessors')
    return decoder, head


def prediction_rows(response_mask):
    """Logit rows (batch, position) whose next token is a response target."""
    if response_mask.dim() != 2 or response_mask.shape[1] < 2 or response_mask[:, 0].any():
        raise ValueError('response_mask must be [batch, seq>=2] with no target on the first token')
    rows = response_mask[:, 1:].nonzero(as_tuple=False)
    if rows.shape[0] == 0:
        raise ValueError('Empty response rows in microbatch')
    return rows[:, 0], rows[:, 1]


def hidden_rows(model, ids, attention, batch_index, positions):
    decoder, head = decoder_and_head(model)
    out = decoder(input_ids=ids, attention_mask=attention, use_cache=False)
    hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
    return hidden[batch_index, positions], head


def support_logits(hidden, head, support):
    if not 0 < support <= head.weight.shape[0]:
        raise ValueError('Invalid common support')
    bias = head.bias[:support] if getattr(head, 'bias', None) is not None else None
    return F.linear(hidden, head.weight[:support], bias)


@torch.no_grad()
def removed_mass(hidden, head, support, support_lse):
    """Probability that the full head assigns outside the audited support."""
    if head.weight.shape[0] == support:
        return torch.zeros_like(support_lse)
    bias = head.bias[support:] if getattr(head, 'bias', None) is not None else None
    extra = F.linear(hidden, head.weight[support:], bias).float().logsumexp(-1)
    total = torch.logaddexp(support_lse.float(), extra)
    return (extra - total).exp()


def _logprobs(student, teacher, alpha, dtype):
    s, t = student.to(dtype), teacher.to(dtype)
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite logits on common support')
    log_p, log_q = s.log_softmax(-1), t.log_softmax(-1)
    if alpha < 1:
        target = torch.logaddexp(log_q + math.log(alpha), log_p.detach() + math.log1p(-alpha))
    else:
        target = log_q
    return s, t, log_p, log_q, target


def _safe_mul(prob, log_term):
    return torch.where(prob > 0, prob * log_term, torch.zeros_like(prob))


def _forward_chunk(student, teacher, targets, mode, topk, clip, alpha, dtype):
    s, t, log_p, log_q, target = _logprobs(student, teacher, alpha, dtype)
    p, q = log_p.exp(), log_q.exp()
    y = targets[:, None]
    lp_y, lq_y = log_p.gather(-1, y).squeeze(-1), log_q.gather(-1, y).squeeze(-1)
    index = None
    if mode == 'exact_rkl':
        value = _safe_mul(p, log_p - target).sum(-1)
    elif mode == 'topk_rkl':
        index = target.topk(topk, dim=-1).indices
        lp, lt = log_p.gather(-1, index), target.gather(-1, index)
        value = (_safe_mul(lp.exp(), lp - lt) - lp.exp() + lt.exp()).sum(-1)
    else:
        advantage = (target.gather(-1, y).squeeze(-1) - lp_y).clamp(-clip, clip)
        value = -advantage * lp_y
    stats = torch.stack([_safe_mul(p, log_p - log_q).sum(-1), lp_y, lq_y, -_safe_mul(p, log_p).sum(-1),
                         -_safe_mul(q, log_q).sum(-1), s.logsumexp(-1), t.logsumexp(-1)], -1)
    return value, stats.float(), index


def _grad_chunk(student, teacher, targets, index, mode, clip, alpha, dtype):
    _, _, log_p, _, target = _logprobs(student, teacher, alpha, dtype)
    p = log_p.exp()
    if mode == 'exact_rkl':
        r = log_p - target
        pr = _safe_mul(p, r)
        return _safe_mul(p, r - pr.sum(-1, keepdim=True))
    if mode == 'topk_rkl':
        lp, lt = log_p.gather(-1, index), target.gather(-1, index)
        weighted = _safe_mul(lp.exp(), lp - lt)
        grad = -p * weighted.sum(-1, keepdim=True)
        return grad.scatter_add(-1, index, weighted)
    y = targets[:, None]
    advantage = (target.gather(-1, y) - log_p.gather(-1, y)).clamp(-clip, clip)
    return (advantage * p).scatter_add(-1, y, -advantage)


class RowDivergence(torch.autograd.Function):
    """Weighted sum of per-row MOPD losses with chunked recomputation.

    Inputs are [N, S] student/teacher logits on the support, target token ids
    [N] and fixed row weights [N]. Nothing of size [N, S] in FP32 is kept for
    backward; each chunk is recomputed. Returns (loss, stats[N, 7]).
    """

    @staticmethod
    def forward(ctx, student, teacher, targets, weights, mode, topk, clip, alpha, chunk):
        if student.dim() != 2 or student.shape != teacher.shape:
            raise ValueError('Expected matching [rows, support] logits')
        rows = student.shape[0]
        if rows == 0 or targets.shape != (rows,) or weights.shape != (rows,):
            raise ValueError('Empty rows or misaligned targets/weights')
        if mode not in MODES or not 0 < alpha <= 1 or chunk < 1 or clip <= 0:
            raise ValueError('Invalid loss options')
        if mode == 'topk_rkl' and not 0 < topk <= student.shape[1]:
            raise ValueError('Invalid top-k')
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        total = torch.zeros((), dtype=dtype, device=student.device)
        stats = torch.empty((rows, len(STAT_FIELDS)), dtype=torch.float32, device=student.device)
        indices = []
        for start in range(0, rows, chunk):
            part = slice(start, min(start + chunk, rows))
            value, chunk_stats, index = _forward_chunk(student[part], teacher[part], targets[part],
                                                       mode, topk, clip, alpha, dtype)
            total += (value * weights[part].to(dtype)).sum()
            stats[part] = chunk_stats
            if index is not None:
                indices.append(index)
        saved = [student, teacher.detach(), targets, weights]
        if indices:
            saved.append(torch.cat(indices))
        ctx.save_for_backward(*saved)
        ctx.options = mode, clip, alpha, chunk, dtype
        ctx.mark_non_differentiable(stats)
        return total, stats

    @staticmethod
    def backward(ctx, grad_total, grad_stats):
        student, teacher, targets, weights, *rest = ctx.saved_tensors
        index = rest[0] if rest else None
        mode, clip, alpha, chunk, dtype = ctx.options
        grad = torch.zeros_like(student)
        scale = grad_total.to(dtype)
        for start in range(0, student.shape[0], chunk):
            part = slice(start, min(start + chunk, student.shape[0]))
            g = _grad_chunk(student[part], teacher[part], targets[part],
                            index[part] if index is not None else None, mode, clip, alpha, dtype)
            grad[part] = (g * (weights[part].to(dtype) * scale)[:, None]).to(student.dtype)
        return grad, None, None, None, None, None, None, None, None


def row_divergence(student, teacher, targets, weights, mode='exact_rkl', topk=64, clip=10.,
                   alpha=1., chunk=64):
    return RowDivergence.apply(student, teacher.detach(), targets, weights.detach(), mode,
                               topk, clip, alpha, chunk)


def microbatch_loss(student, teacher, ids, attention, prompt_lengths, prefixes, support,
                    sequence_weights, loss_cfg):
    """One right-padded, single-domain micro-batch. Teacher is frozen (no_grad)."""
    _, response = teacher_batch_from_student(ids, attention, prompt_lengths, prefixes, support)
    batch_index, positions = prediction_rows(response)
    targets = ids[batch_index, positions + 1]
    with torch.no_grad():
        teacher_hidden, teacher_head = hidden_rows(teacher, ids, attention, batch_index, positions)
        teacher_logits = support_logits(teacher_hidden, teacher_head, support)
    student_hidden, student_head = hidden_rows(student, ids, attention, batch_index, positions)
    student_logits = support_logits(student_hidden, student_head, support)
    weights = sequence_weights.to(student_logits.device)[batch_index].float()
    loss, stats = row_divergence(student_logits, teacher_logits, targets, weights,
                                 loss_cfg.get('mode', 'exact_rkl'), loss_cfg.get('topk', 64),
                                 loss_cfg.get('pg_advantage_clip', 10.), loss_cfg.get('proximal_teacher_alpha', 1.),
                                 loss_cfg.get('row_chunk', 64))
    detail = {'stats': stats, 'batch_index': batch_index, 'positions': positions,
              'removed_student': removed_mass(student_hidden.detach(), student_head, support,
                                              stats[:, STAT['student_lse']]),
              'removed_teacher': removed_mass(teacher_hidden, teacher_head, support,
                                              stats[:, STAT['teacher_lse']])}
    return loss, detail
