"""Chunked cross-entropy on response rows over the audited support (sft-v3).

Same memory discipline as mopd_v3.RowDivergence: logits arrive as [N, S] in
the model dtype; FP32 softmax workspaces exist one chunk at a time and are
recomputed in backward instead of being stored.
"""
import torch


class RowCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, weights, chunk):
        if logits.dim() != 2 or targets.shape != (logits.shape[0],) or weights.shape != (logits.shape[0],):
            raise ValueError('Expected [rows, support] logits with aligned targets/weights')
        if logits.shape[0] == 0 or chunk < 1:
            raise ValueError('Empty rows or invalid chunk')
        if (targets < 0).any() or (targets >= logits.shape[1]).any():
            raise ValueError('Target token outside the audited support')
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        total = torch.zeros((), dtype=dtype, device=logits.device)
        nll = torch.empty(logits.shape[0], dtype=torch.float32, device=logits.device)
        for start in range(0, logits.shape[0], chunk):
            part = slice(start, min(start + chunk, logits.shape[0]))
            x = logits[part].to(dtype)
            if not torch.isfinite(x).all():
                raise ValueError('Nonfinite logits on common support')
            value = x.logsumexp(-1) - x.gather(-1, targets[part, None]).squeeze(-1)
            nll[part] = value.float()
            total += (value * weights[part].to(dtype)).sum()
        ctx.save_for_backward(logits, targets, weights)
        ctx.options = chunk, dtype
        ctx.mark_non_differentiable(nll)
        return total, nll

    @staticmethod
    def backward(ctx, grad_total, grad_nll):
        logits, targets, weights = ctx.saved_tensors
        chunk, dtype = ctx.options
        grad = torch.zeros_like(logits)
        for start in range(0, logits.shape[0], chunk):
            part = slice(start, min(start + chunk, logits.shape[0]))
            p = logits[part].to(dtype).softmax(-1)
            g = p.scatter_add(-1, targets[part, None], -torch.ones_like(p[:, :1]))
            grad[part] = (g * (weights[part].to(dtype) * grad_total.to(dtype))[:, None]).to(logits.dtype)
        return grad, None, None, None


def row_cross_entropy(logits, targets, weights, chunk=64):
    """Returns (weighted sum of per-row NLL, per-row NLL)."""
    return RowCrossEntropy.apply(logits, targets, weights.detach(), chunk)
