"""Batched on-policy rollouts for opd-v3.

Generation uses a frozen copy of the student. Each step the LoRA delta is
merged into that copy out of place, in FP32. The training model is never
merged/unmerged: in bf16, W + dW - dW does not round-trip, so repeated
merge/unmerge would drift the frozen base weights over many steps.
"""
import copy

import torch

from src.training.sampling_v2 import generate_exact


def left_pad(prefixes, pad_id, device):
    width = max(len(p) for p in prefixes)
    ids = torch.full((len(prefixes), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(prefixes), width), dtype=torch.long)
    for i, prefix in enumerate(prefixes):
        ids[i, width - len(prefix):] = torch.tensor(prefix, dtype=torch.long)
        attention[i, width - len(prefix):] = 1
    return ids.to(device), attention.to(device), width


def split_generated(sequences, width, eos_id):
    """Generated tokens per row, cut after the first EOS (the EOS is kept).

    Rows that finish early are filled with pad after their EOS, so everything
    before the first EOS is real output. Rows without EOS hit max_new_tokens.
    """
    out = []
    for row in sequences[:, width:].tolist():
        if eos_id in row:
            out.append((row[:row.index(eos_id) + 1], True))
        else:
            out.append((row, False))
    return out


class GenerationCopy:
    """Frozen bf16 copy of the base model that receives W + s*B@A each step."""

    def __init__(self, base_model):
        self.model = copy.deepcopy(base_model).eval().requires_grad_(False)

    @torch.no_grad()
    def sync_from(self, peft_model, adapter='default'):
        from peft.tuners.lora import LoraLayer
        merged = 0
        for name, module in peft_model.get_base_model().named_modules():
            if isinstance(module, LoraLayer) and adapter in getattr(module, 'lora_A', {}):
                target = self.model.get_submodule(name)
                base = module.get_base_layer().weight
                delta = module.get_delta_weight(adapter)
                target.weight.copy_((base.detach().float() + delta.float()).to(target.weight.dtype))
                merged += 1
        if not merged:
            raise ValueError('No LoRA layers found to merge')
        return merged


class HFRolloutEngine:
    """Rollouts with transformers.generate under the audited sampling-v2 chain."""

    def __init__(self, model, configs, contracts, support, pad_id, eos_id, batch_size, sync=None):
        self.model, self.configs, self.contracts = model, configs, contracts
        self.support, self.pad_id, self.eos_id = support, pad_id, eos_id
        self.batch_size, self._sync = batch_size, sync

    def sync(self, student):
        return self._sync(student) if self._sync else 0

    @torch.no_grad()
    def generate(self, prefixes, domain):
        if not prefixes:
            return []
        device = next(self.model.parameters()).device
        # Length-sorted batches waste less left padding; results keep input order.
        order = sorted(range(len(prefixes)), key=lambda i: (len(prefixes[i]), i))
        results = [None] * len(prefixes)
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, len(order), self.batch_size):
                chunk = order[start:start + self.batch_size]
                ids, attention, width = left_pad([prefixes[i] for i in chunk], self.pad_id, device)
                sequences = generate_exact(self.model, ids, attention, self.configs[domain], self.support,
                                           self.contracts[domain]['logits_processors'])
                for i, (tokens, ended) in zip(chunk, split_generated(sequences, width, self.eos_id)):
                    results[i] = {'prompt_ids': list(prefixes[i]), 'response_ids': tokens,
                                  'ended_with_eos': ended, 'domain': domain}
        finally:
            self.model.train(was_training)
        return results
