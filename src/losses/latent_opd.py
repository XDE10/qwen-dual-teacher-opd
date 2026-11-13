"""Routed same-lineage Latent-MOPD losses (arXiv:2610.02381, Appendix C)."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.losses.mopd_v3 import decoder_and_head, prediction_rows, support_logits
from src.teachers.opd_contract import teacher_batch_from_student

TOP_K = 16
LOGIT_ROWS = 256


def loss_weights(variant, domain, completed_updates):
    if variant == 'token':
        return 1.0, 0.0
    latent_steps = 20 if variant == 'math20' and domain == 'math' else 10
    return min(1.0, completed_updates / 10), max(0.0, 1 - completed_updates / latent_steps)


def hidden_alignment(student, teacher):
    """Per-position normalized squared error, averaged over the selected layers."""
    student = F.normalize(student.float(), dim=-1)
    teacher = F.normalize(teacher.detach().float(), dim=-1)
    return (student - teacher).square().mean(dim=(0, 2))


def response_batch(ids, attention, prefixes, support):
    batch, mask = teacher_batch_from_student(ids, attention, [len(p) for p in prefixes], prefixes, support)
    return batch, *prediction_rows(mask)


def _layer_rows(model, response, last_three=False):
    batch, rows, positions = response
    decoder, head = decoder_and_head(model)
    result = decoder(**batch, use_cache=False, output_hidden_states=last_three)
    states = result.hidden_states[-3:] if last_three else (result.last_hidden_state,)
    return tuple(state[rows, positions] for state in states), head


def _candidate_log_probs(hidden, head, support, candidates):
    logits = support_logits(hidden, head, support).float()
    return logits.gather(-1, candidates) - logits.logsumexp(-1, keepdim=True)


@torch.no_grad()
def score_candidates(student, teacher, response, support):
    (student_hidden,), student_head = _layer_rows(student, response)
    (teacher_hidden,), teacher_head = _layer_rows(teacher, response)
    candidates, rewards = [], []
    for student_rows, teacher_rows in zip(student_hidden.split(LOGIT_ROWS), teacher_hidden.split(LOGIT_ROWS)):
        logits = support_logits(student_rows, student_head, support).float()
        selected, ids = logits.topk(TOP_K, dim=-1)
        student_log_probs = selected - logits.logsumexp(-1, keepdim=True)
        del logits
        teacher_log_probs = _candidate_log_probs(teacher_rows, teacher_head, support, ids)
        candidates.append(ids)
        rewards.append(selected.softmax(-1) * (teacher_log_probs - student_log_probs))
    return torch.cat(candidates), torch.cat(rewards)


def reward_scale(scored):
    """One sample standard deviation over the entire domain batch, without centering."""
    rewards = torch.cat([rewards.flatten() for _, rewards in scored])
    return rewards.std(correction=1).clamp_min(1e-6) if rewards.numel() > 1 else 1.0


def token_objective(hidden, head, support, candidates, advantages):
    log_probs = []
    for rows, ids in zip(hidden.split(LOGIT_ROWS), candidates.split(LOGIT_ROWS)):
        log_probs.append(checkpoint(_candidate_log_probs, rows, head, support, ids, use_reentrant=False))
    return -(advantages.detach() * torch.cat(log_probs)).sum(-1).mean()


def routed_loss(student, teacher, response, support, token_weight, latent_weight, candidates, advantages):
    student_layers, head = _layer_rows(student, response, last_three=bool(latent_weight))
    loss = 0.0
    if latent_weight:
        with torch.no_grad():
            teacher_layers, _ = _layer_rows(teacher, response, last_three=True)
        loss = latent_weight * hidden_alignment(torch.stack(student_layers), torch.stack(teacher_layers)).mean()
    if token_weight:
        loss = loss + token_weight * token_objective(student_layers[-1], head, support, candidates, advantages)
    return loss
