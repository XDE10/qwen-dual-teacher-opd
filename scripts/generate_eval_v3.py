"""Greedy evaluation generation (protocol v3) with vLLM, falling back to batched HF.

Examples (run from the repository root on the GPU machine):

  python -m scripts.generate_eval_v3 --model Qwen/Qwen2.5-1.5B-Instruct \
      --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 --suites math500 gsm8k_test humanevalplus mbppplus \
      --output-dir /content/drive/MyDrive/project2/eval/d0
  python -m scripts.generate_eval_v3 ... --adapter runs/opd_v3_d4/checkpoint-80/student --output-dir .../d4
  python -m scripts.generate_eval_v3 --self-check --suites math500 math_dev     # grader coverage only

Math is graded here. Code suites only write *.candidates.jsonl, which are then
executed in the local Docker sandbox:
  python -m scripts.evaluate_code_v2 --mode student --candidates <code_dev.candidates.jsonl> \
      --registry reports/handover_20261024_v3/reviewed_comparators.json --output-dir <dir>
  python -m scripts.evaluate_evalplus_v2 --dataset humaneval --predictions <humanevalplus.candidates.jsonl> ...
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from src.evaluation.eval_v3 import (DEFAULT_MAX_NEW, SUITES, evalplus_text, grade_math, grader_self_check, render,
                                    suite_rows, summarize)
from src.teachers.opd_contract import load_contract

RELEASE = 'data/project2_release_v5_20261108/release/manifest.json'


def adapter_identity(path):
    if not path:
        return None
    path = Path(path)
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()}


def vllm_generate(args, prompts, max_new, support, eos):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, revision=args.revision, dtype='bfloat16', max_model_len=args.max_model_len,
              generation_config='vllm', seed=0, enable_lora=bool(args.adapter), max_lora_rank=args.max_lora_rank,
              gpu_memory_utilization=args.gpu_memory_utilization)
    lora = LoRARequest('adapter', 1, str(Path(args.adapter).resolve())) if args.adapter else None
    results = {}
    for name, items in prompts.items():
        params = SamplingParams(temperature=0.0, max_tokens=max_new[name], stop_token_ids=[eos],
                                allowed_token_ids=list(range(support)), skip_special_tokens=True)
        outputs = llm.generate([{'prompt_token_ids': ids} for ids in items], params, lora_request=lora)
        results[name] = [(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason) for o in outputs]
    return results, {'engine': 'vllm', 'version': __import__('vllm').__version__}


def hf_generate(args, prompts, max_new, support, eos, tok):
    import torch
    from transformers import AutoModelForCausalLM, LogitsProcessorList
    from src.training.rollout_v3 import left_pad, split_generated
    from src.training.sampling_v2 import CommonSupportMask
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').to('cuda').eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()
    results = {}
    for name, items in prompts.items():
        order = sorted(range(len(items)), key=lambda i: len(items[i]))
        out = [None] * len(items)
        for start in range(0, len(order), args.batch_size):
            chunk = order[start:start + args.batch_size]
            ids, attention, width = left_pad([items[i] for i in chunk], tok.pad_token_id, 'cuda')
            with torch.no_grad():
                seq = model.generate(ids, attention_mask=attention, do_sample=False, max_new_tokens=max_new[name],
                                     eos_token_id=eos, pad_token_id=tok.pad_token_id,
                                     logits_processor=LogitsProcessorList([CommonSupportMask(support)]))
            for i, (tokens, ended) in zip(chunk, split_generated(seq, width, eos)):
                out[i] = (tok.decode(tokens, skip_special_tokens=True), len(tokens), 'stop' if ended else 'length')
        results[name] = out
    return results, {'engine': 'hf', 'version': __import__('transformers').__version__}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model')
    p.add_argument('--revision')
    p.add_argument('--adapter', help='LoRA adapter directory (e.g. checkpoint-N/student)')
    p.add_argument('--suites', nargs='+', required=True, choices=sorted(SUITES))
    p.add_argument('--output-dir')
    p.add_argument('--release-manifest', default=RELEASE)
    p.add_argument('--tokenizer-path', default='data/tokenizers/Qwen2.5-1.5B-Instruct')
    p.add_argument('--tokenizer-audit', default='reports/tokenizer_alignment_report.json')
    p.add_argument('--engine', choices=['vllm', 'hf'], default='vllm')
    p.add_argument('--max-new-math', type=int, default=DEFAULT_MAX_NEW['math'])
    p.add_argument('--max-new-code', type=int, default=DEFAULT_MAX_NEW['code'])
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--max-lora-rank', type=int, default=64)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--self-check', action='store_true', help='grade math references against themselves; no model')
    a = p.parse_args()
    if a.self_check:
        for name in a.suites:
            kind, path, rows = suite_rows(name, a.release_manifest)
            if kind == 'math':
                print(name, json.dumps(grader_self_check(rows)))
        return
    if not a.model or not a.revision or a.revision == 'main' or not a.output_dir:
        raise ValueError('Pin --model, --revision (not main) and --output-dir')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_path, local_files_only=True)
    contract = load_contract()
    audit = json.loads(Path(a.tokenizer_audit).read_text(encoding='utf-8'))
    pinned = {r['model']: r['revision'] for r in audit['models']}
    if pinned.get(a.model) not in (None, a.revision):
        raise ValueError('Revision differs from tokenizer audit for ' + a.model)
    support = len(audit['common_support']['ids'])
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    loaded = {name: suite_rows(name, a.release_manifest) for name in a.suites}
    prompts = {name: [render(tok, kind, r, contract) for r in rows] for name, (kind, _, rows) in loaded.items()}
    budget = {'math': a.max_new_math, 'code': a.max_new_code, 'evalplus': a.max_new_code}
    max_new = {name: budget[kind] for name, (kind, _, _) in loaded.items()}
    for name, items in prompts.items():
        if max(len(x) for x in items) + max_new[name] > a.max_model_len:
            raise ValueError(f'{name}: prompt + budget exceeds max_model_len; never truncate problems')
    started = time.time()
    if a.engine == 'vllm':
        results, engine = vllm_generate(a, prompts, max_new, support, tok.eos_token_id)
    else:
        results, engine = hf_generate(a, prompts, max_new, support, tok.eos_token_id, tok)
    identity = {'protocol': 'eval-v3', 'model': a.model, 'revision': a.revision, 'adapter': a.adapter,
                # Prompts are rendered with the audited student tokenizer; reference models outside the
                # audit (e.g. Qwen2.5-Math-1.5B-Instruct) share the Qwen2.5 vocabulary but are not audited.
                'tokenizer_audited_for_model': a.model in pinned,
                'adapter_files_sha256': adapter_identity(a.adapter), 'engine': engine, 'decoding': 'greedy',
                'max_new_tokens': max_new, 'support': support, 'seconds': time.time() - started,
                'prompt_contract_sha256': hashlib.sha256(Path('configs/project2_prompt_contract.json').read_bytes()).hexdigest()}
    for name, (kind, path, rows) in loaded.items():
        records = []
        for row, ids, (text, n, finish) in zip(rows, prompts[name], results[name]):
            sid = row.get('sample_id', row.get('task_id'))
            record = {'id': sid, 'prompt_tokens': len(ids), 'generated_tokens': n, 'finish_reason': finish, 'text': text}
            if kind == 'math':
                record['grade'] = grade_math(text, row['reference_answer'], finish == 'length')
            records.append(record)
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as f:
            f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
        if kind != 'math':
            key = 'task_id' if kind == 'evalplus' else 'sample_id'
            with (out / f'{name}.candidates.jsonl').open('w', encoding='utf-8') as f:
                for r in records:
                    text = evalplus_text(r['text']) if kind == 'evalplus' else r['text']
                    f.write(json.dumps({key: r['id'], 'text': text}, ensure_ascii=False) + '\n')
        summary = {**identity, 'suite': name, 'data_path': path.as_posix(),
                   'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **summarize(kind, records)}
        (out / f'{name}.summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, json.dumps({k: summary[k] for k in summary if k.startswith(('accuracy', 'count', 'truncated'))}))


if __name__ == '__main__':
    main()
