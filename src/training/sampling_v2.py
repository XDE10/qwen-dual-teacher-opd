"""OPD sampling contract v2: fresh config, complete support, audited real chain."""
import copy
from contextlib import contextmanager
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList


class CommonSupportMask(LogitsProcessor):
    def __init__(self, support):
        self.support = support

    def __call__(self, input_ids, scores):
        scores[:, self.support:] = -float('inf')
        return scores


def generation_config(eos, pad, max_new_tokens):
    # Do not merge model.generation_config: its sampling defaults are untrusted.
    return GenerationConfig(
        do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
        repetition_penalty=1.0, encoder_repetition_penalty=1.0,
        typical_p=1.0, min_p=None, epsilon_cutoff=0.0, eta_cutoff=0.0,
        no_repeat_ngram_size=0, encoder_no_repeat_ngram_size=0,
        bad_words_ids=None, force_words_ids=None, forced_bos_token_id=None,
        forced_eos_token_id=None, suppress_tokens=None, begin_suppress_tokens=None,
        sequence_bias=None, renormalize_logits=False, remove_invalid_values=False,
        min_length=0, min_new_tokens=None, num_beams=1, num_beam_groups=1,
        diversity_penalty=0.0, penalty_alpha=None, guidance_scale=None,
        watermarking_config=None, token_healing=False,
        eos_token_id=eos, pad_token_id=pad, max_new_tokens=max_new_tokens,
        use_cache=True,
    )


def chain_description(chain):
    names = [type(p).__module__ + '.' + type(p).__name__ for p in chain]
    if len(chain) != 1 or type(chain[0]) is not CommonSupportMask:
        raise ValueError('Unexpected distribution-changing logits processor chain: ' + repr(names))
    return [{'class': names[0], 'support': chain[0].support}]


def resolved_contract(model, config, support, device):
    # This private API is pinned/tested against transformers 4.57.6; fail closed
    # if a new version changes its signature or injects another processor.
    resolved, _ = model._prepare_generation_config(config, use_model_defaults=False)
    prepared = copy.deepcopy(resolved)
    model._prepare_special_tokens(prepared, kwargs_has_attention_mask=True, device=device)
    chain = model._get_logits_processor(
        generation_config=prepared, input_ids_seq_length=1,
        encoder_input_ids=None, prefix_allowed_tokens_fn=None,
        logits_processor=LogitsProcessorList([CommonSupportMask(support)]),
        device=device, model_kwargs={'use_cache': True})
    return {'version': 'sampling-v2', 'use_model_defaults': False, 'generation_config': resolved.to_dict(),
            'logits_processors': chain_description(chain)}


@contextmanager
def audit_actual_chain(model, expected):
    original = model._get_logits_processor

    def audited(*args, **kwargs):
        chain = original(*args, **kwargs)
        if chain_description(chain) != expected:
            raise ValueError('Actual processor chain differs from run identity')
        return chain

    model._get_logits_processor = audited
    try:
        yield
    finally:
        del model._get_logits_processor


def generate_exact(model, ids, attention, config, support, expected, diagnostics=False):
    target = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with audit_actual_chain(target, expected):
        return model.generate(
            ids, attention_mask=attention, generation_config=config,
            use_model_defaults=False,
            do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
            repetition_penalty=1.0,
            return_dict_in_generate=diagnostics, output_scores=diagnostics,
            logits_processor=LogitsProcessorList([CommonSupportMask(support)]))
