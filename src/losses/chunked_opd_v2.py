"""Exact full-common-support reverse KL with bounded recomputation workspaces.

The full model logits and their student gradient still exist. This does NOT
chunk the model's LM head. Only response-row KL workspaces are bounded.
"""
import torch


def row_blocks(mask, sequence_chunk, row_chunk):
    for b in range(mask.shape[0]):
        for start in range(0, mask.shape[1] - 1, sequence_chunk):
            stop = min(start + sequence_chunk, mask.shape[1] - 1)
            positions = mask[b, start + 1:stop + 1].nonzero().flatten() + start
            for block in positions.split(row_chunk):
                if block.numel():
                    yield b, block


def distributions(student, teacher, b, rows, support):
    # Select only a bounded group of response prediction rows FIRST, then cols.
    s = student[b].index_select(0, rows)[:, :support]
    t = teacher[b].index_select(0, rows)[:, :support]
    dtype = torch.float64 if s.dtype == torch.float64 else torch.float32
    s, t = s.to(dtype), t.to(dtype)
    if not torch.isfinite(s).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite logits on common support')
    return s.log_softmax(-1), t.log_softmax(-1)


class _ExactKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, mask, support, sequence_chunk, row_chunk):
        if student.shape[:2] != teacher.shape[:2] or mask.shape != student.shape[:2]:
            raise ValueError('Sequence shapes differ')
        if not 0 < support <= min(student.shape[-1], teacher.shape[-1]):
            raise ValueError('Invalid common support')
        if sequence_chunk < 1 or row_chunk < 1:
            raise ValueError('Chunks must be positive')
        count = int(mask[:, 1:].sum())
        if count == 0:
            raise ValueError('Empty response')
        ctx.save_for_backward(student, teacher.detach(), mask)
        ctx.options = support, sequence_chunk, row_chunk, count
        dtype = torch.float64 if student.dtype == torch.float64 else torch.float32
        total = torch.zeros((), dtype=dtype, device=student.device)
        for b, rows in row_blocks(mask, sequence_chunk, row_chunk):
            s, t = distributions(student, teacher, b, rows, support)
            total += (s.exp() * (s - t)).sum()
        return total / count

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher, mask = ctx.saved_tensors
        support, sequence_chunk, row_chunk, count = ctx.options
        grad = torch.zeros_like(student)
        # Recompute chunks rather than retain all FP32 probabilities for backward.
        for b, rows in row_blocks(mask, sequence_chunk, row_chunk):
            s, t = distributions(student, teacher, b, rows, support)
            p = s.exp()
            difference = s - t
            kl = (p * difference).sum(-1, keepdim=True)
            g = p * (difference - kl) * (grad_output / count)
            grad[b, rows, :support] = g.to(student.dtype)
        return grad, None, None, None, None, None


def chunked_reverse_kl(student, teacher, response_mask, support, sequence_chunk=128, row_chunk=16):
    if response_mask.dtype != torch.bool:
        response_mask = response_mask.bool()
    return _ExactKL.apply(student, teacher, response_mask, support, sequence_chunk, row_chunk)


@torch.no_grad()
def removed_mass_values(logits, mask, support, sequence_chunk=128, row_chunk=16):
    values = []
    for b, rows in row_blocks(mask, sequence_chunk, row_chunk):
        x = logits[b].index_select(0, rows).float()
        # Keep only a response-length vector on CPU for exact full-update quantiles.
        values.append((1 - (x[:, :support].logsumexp(-1) - x.logsumexp(-1)).exp()).cpu())
    return torch.cat(values) if values else torch.empty(0)


def summarize_mass(values):
    x = torch.cat(values).double()
    return {'tokens': x.numel(), 'mean': x.mean().item(), 'max': x.max().item(),
            'p50': x.quantile(.5).item(), 'p95': x.quantile(.95).item(),
            'p99': x.quantile(.99).item(), 'min': x.min().item()}


def gradient_normalizer(mode, response_tokens, microbatches):
    if mode == 'response_tokens_v2':
        return response_tokens
    if mode == 'microbatch_mean_v1':
        return microbatches
    raise ValueError('Unknown normalization: ' + mode)
