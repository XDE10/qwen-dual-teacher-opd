"""Batched greedy generation on the audited common vocabulary."""


def greedy_config(eos, pad, max_new_tokens):
    """Plain greedy decoding. The checkpoints' generation_config.json files set repetition_penalty 1.05-1.1 and
    sampling defaults; generate(..., use_model_defaults=False) with this config keeps them out, the same as
    vLLM's temperature-0 path and the training rollouts (sampling-v2)."""
    from transformers import GenerationConfig
    return GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                            eos_token_id=eos, pad_token_id=pad, use_cache=True)


def generate_greedy(model, ids, attention, config, support, expected):
    """model.generate with only the audited support mask in the processor chain (checked on every call)."""
    from transformers import LogitsProcessorList
    from src.training.sampling_v2 import CommonSupportMask, audit_actual_chain
    target = model.get_base_model() if hasattr(model, 'get_base_model') else model
    with audit_actual_chain(target, expected):
        return model.generate(ids, attention_mask=attention, generation_config=config, use_model_defaults=False,
                              logits_processor=LogitsProcessorList([CommonSupportMask(support)]))


def hf_generate(args, prompts, max_new, support, eos, tok):
    import torch
    from transformers import AutoModelForCausalLM
    from src.training.rollout_v3 import left_pad, split_generated
    from src.training.sampling_v2 import resolved_contract
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').to('cuda').eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    target = model.get_base_model() if hasattr(model, 'get_base_model') else model
    results, contracts = {}, {}
    for name, items in prompts.items():
        config = greedy_config(eos, tok.pad_token_id, max_new[name])
        contracts[name] = resolved_contract(target, config, support, 'cuda')
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        out = [None] * len(items)
        for start in range(0, len(order), args.batch_size):
            chunk = order[start:start + args.batch_size]
            ids, attention, width = left_pad([items[i] for i in chunk], tok.pad_token_id, 'cuda')
            with torch.no_grad():
                seq = generate_greedy(model, ids, attention, config, support, contracts[name]['logits_processors'])
            for i, (tokens, ended) in zip(chunk, split_generated(seq, width, eos)):
                out[i] = (tok.decode(tokens, skip_special_tokens=True), len(tokens), 'stop' if ended else 'length', tokens)
        results[name] = out
    return results, {'engine': 'hf', 'version': __import__('transformers').__version__,
                     'generation': {name: c['generation_config'] for name, c in contracts.items()},
                     'logits_processors': next(iter(contracts.values()))['logits_processors'] if contracts else None}
