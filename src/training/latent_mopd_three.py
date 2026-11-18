"""Three-teacher reconstruction of Latent-MOPD Table 1, using one CUDA GPU."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from src.data.approved_training_inputs import load_approved
from src.losses.latent_opd import LOGIT_ROWS, TOP_K, hidden_alignment, loss_weights, response_batch, reward_scale
from src.losses.mopd_v3 import decoder_and_head, support_logits
from src.teachers.tokenizer_audit import compare_vocabularies
from src.training import opd_v3 as opd
from src.training.latent_opd import STUDENT, TEACHERS as DUAL_TEACHERS
from src.training.latent_progress import RunProgress
from src.training.model_drive_cache import copy_model, verify
from src.training.rollout_v3 import HFRolloutEngine
from src.training.runtime_v2 import read, sha, write
from src.training.runtime_v3 import SOURCES, build_identity, check_identity, prepare_output
from src.training.sampling_v2 import generation_config, resolved_contract

DOMAINS = ('math', 'code', 'logic')
TEACHERS = {
    **DUAL_TEACHERS,
    'logic': {'model': 'nvidia/Nemotron-Research-Reasoning-Qwen-1.5B',
              'revision': 'a8e647c7a46f04e84ce732d6fa3fa627ba343b8d'},
}
SUPPORT = 151665
PROMPTS = 32
RESPONSES = 4
WORKER_RESPONSES = 16
PACK_TOKENS = 24576
STEPS = 62
MAX_RESPONSE = 16384
MAX_PROMPT = 2048
SOURCE_FILES = [*SOURCES, 'src/training/latent_mopd_three.py', 'src/training/latent_opd.py',
                'src/training/model_drive_cache.py', 'src/training/latent_progress.py',
                'src/losses/latent_opd.py', 'src/teachers/tokenizer_audit.py']


def render_prefix(tokenizer, prompt):
    ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False,
    )
    if not ids or any(not 0 <= token < SUPPORT for token in ids):
        raise ValueError('Rendered prefix is empty or contains tokens outside the audited vocabulary')
    return ids


def tokenizer_audit(root):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    specs = [STUDENT, *TEACHERS.values()]
    models = [Path(snapshot_download(
        spec['model'], revision=spec['revision'], local_dir=root / spec['model'].split('/')[-1],
        allow_patterns=['config.json', 'tokenizer*.json', 'special_tokens_map.json', 'chat_template.jinja'],
    )) for spec in specs]
    tokenizers = [AutoTokenizer.from_pretrained(folder, local_files_only=True) for folder in models]
    support, differences = compare_vocabularies([tok.get_vocab() for tok in tokenizers])
    if differences or support['ids'] != list(range(SUPPORT)):
        raise ValueError('Student and all three teachers must share 151665 ordered token IDs')
    config = [read(folder / 'config.json') for folder in models]
    required = {('qwen2', 28, 1536)}
    if {(c['model_type'], c['num_hidden_layers'], c['hidden_size']) for c in config} != required:
        raise ValueError('Hidden size, depth or model family differs across the four models')
    if min(c['vocab_size'] for c in config) < SUPPORT:
        raise ValueError('A model head does not cover the audited vocabulary')
    audit = root / 'tokenizer_audit.json'
    write(audit, {'models': specs, 'common_support': support, 'mapping_difference_count': 0})
    return tokenizers[0], models[0], audit


def stage_weights(spec, root, drive=None):
    """Verify the index and every file belonging to the pinned safetensors snapshot."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(spec['model'], revision=spec['revision'], files_metadata=True)
    if info.sha != spec['revision']:
        raise ValueError('Resolved model revision differs from the training recipe')
    files = [f for f in info.siblings if '/' not in f.rfilename
             and (f.rfilename.endswith('.safetensors') or f.rfilename.endswith('.json'))]
    names = {f.rfilename for f in files}
    index = 'model.safetensors.index.json'
    if 'config.json' not in names or not ('model.safetensors' in names or index in names):
        raise ValueError('Incomplete safetensors snapshot')
    cache_name = f"{spec['model'].split('/')[-1]}--{spec['revision']}"
    folder = root / cache_name
    manifest = {'files': [{'path': f.rfilename, 'size': f.size,
                           'lfs_sha256': f.lfs.sha256 if f.lfs else None, 'git_blob_id': f.blob_id}
                          for f in files]}
    if not verify(folder, manifest):
        if drive and shutil.which('rclone'):
            copy_model(f"{drive.rstrip('/')}/{cache_name}", folder, sorted(names))
        if not verify(folder, manifest):
            snapshot_download(spec['model'], revision=spec['revision'], allow_patterns=sorted(names),
                              local_dir=folder)
            if not verify(folder, manifest):
                raise ValueError('Hash mismatch in model snapshot ' + spec['model'])
    if index in names:
        shards = set(read(folder / index)['weight_map'].values())
        if not shards or shards != {name for name in names if name.endswith('.safetensors')}:
            raise ValueError('Safetensors index does not match pinned shard names')
    return folder


def build_schedule(pools, steps=STEPS, prompts=PROMPTS):
    """File order is the training order. Each domain has its own cursor."""
    schedule = []
    for step in range(steps):
        domain = DOMAINS[step % len(DOMAINS)]
        offset = step // len(DOMAINS) * prompts
        rows = pools[domain][offset:offset + prompts]
        if len(rows) != prompts:
            raise ValueError(f'{domain} has too few approved prompts for step {step + 1}')
        schedule.append((domain, rows))
    return schedule


def read_schedule(manifest, tokenizer, steps, prompts):
    doc = read(manifest)
    if doc.get('schema') != 'latent-mopd-three-release-1' or not doc.get('approval', {}).get('approver'):
        raise ValueError('Training requires an owner-approved three-teacher release')
    pools = {domain: load_approved(manifest, f'{domain}_train.jsonl') for domain in DOMAINS}
    protected_path = manifest.parent / doc['protected_dev_ids_file']
    if sha(protected_path) != doc['files'][protected_path.name]['sha256']:
        raise ValueError('Protected ID digest differs from release')
    protected = set(read(protected_path)['sample_ids'])
    seen = set()
    for domain, rows in pools.items():
        for row in rows:
            sample = row['sample_id']
            if row['domain'] != domain or sample in seen or sample in protected:
                raise ValueError(f'Wrong domain, duplicate or protected prompt: {sample}')
            seen.add(sample)
    schedule = build_schedule(pools, steps, prompts)
    rendered = []
    for domain, rows in schedule:
        prefixes = [render_prefix(tokenizer, row['prompt']) for row in rows]
        if any(len(ids) != row['prompt_tokens'] or len(ids) > MAX_PROMPT
               or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != row['prefix_sha256']
               for ids, row in zip(prefixes, rows, strict=True)):
            raise ValueError(f'Prompt rendering differs from the sealed {domain} release')
        rendered.append((domain, prefixes, [row['sample_id'] for row in rows]))
    return rendered


def microbatches(lengths):
    """Keep the paper's worker-local packing boundaries on a single device."""
    for start in range(0, len(lengths), WORKER_RESPONSES):
        local = lengths[start:start + WORKER_RESPONSES]
        for group in opd.pack_by_tokens(local, PACK_TOKENS):
            yield [i + start for i in group]


def state_rows(model, response, last_three):
    batch, rows, positions = response
    decoder, head = decoder_and_head(model)
    out = decoder(**batch, output_hidden_states=last_three, use_cache=False)
    if last_three:
        # Qwen2 returns the final RMSNorm output as its last hidden state.
        return tuple(h[rows, positions] for h in out.hidden_states[-3:]), head
    return (out.last_hidden_state[rows, positions],), head


def candidate_log_probs(hidden, head, support, ids):
    logits = support_logits(hidden, head, support).float()
    return logits.gather(-1, ids) - logits.logsumexp(-1, keepdim=True)


@torch.no_grad()
def score(student, teacher, batch, support):
    (student_rows,), student_head = state_rows(student, batch, False)
    (teacher_rows,), teacher_head = state_rows(teacher, batch, False)
    ids, advantages = [], []
    for s, t in zip(student_rows.split(LOGIT_ROWS), teacher_rows.split(LOGIT_ROWS), strict=True):
        logits = support_logits(s, student_head, support).float()
        selected, indices = logits.topk(TOP_K, dim=-1)
        student_logp = selected - logits.logsumexp(-1, keepdim=True)
        del logits
        teacher_logp = candidate_log_probs(t, teacher_head, support, indices)
        ids.append(indices)
        advantages.append(selected.softmax(-1) * (teacher_logp - student_logp))
    return torch.cat(ids), torch.cat(advantages)


def objective(student, teacher, batch, ids, advantage, token_weight, rep_weight, support):
    states, head = state_rows(student, batch, bool(rep_weight))
    token_loss = rep_loss = states[-1].new_zeros((), dtype=torch.float32)
    if rep_weight:
        with torch.no_grad():
            target, _ = state_rows(teacher, batch, True)
        rep_loss = hidden_alignment(torch.stack(states), torch.stack(target)).mean()
    if token_weight:
        logp = torch.cat([
            checkpoint(candidate_log_probs, h, head, support, i, use_reentrant=False)
            for h, i in zip(states[-1].split(LOGIT_ROWS), ids.split(LOGIT_ROWS), strict=True)
        ])
        token_loss = -(advantage * logp).sum(-1).mean()
    return token_weight * token_loss + rep_weight * rep_loss, token_loss.detach(), rep_loss.detach()


def update(student, teacher, optimizer, prefixes, rollouts, token_weight, rep_weight, pad_id, device,
           support=SUPPORT, on_phase=lambda *args, **kwargs: None):
    if len(prefixes) != len(rollouts) or any(not row['response_ids'] or row['prompt_ids'] != prefix
                                           for prefix, row in zip(prefixes, rollouts)):
        raise ValueError('Missing, empty or mismatched rollout')
    sequences = [ids + row['response_ids'] for ids, row in zip(prefixes, rollouts, strict=True)]
    groups = list(microbatches([len(ids) for ids in sequences]))

    def make_batch(group):
        ids, mask = opd.right_pad([sequences[i] for i in group], pad_id, device)
        return response_batch(ids, mask, [prefixes[i] for i in group], support)

    autocast = (lambda: torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)
                if torch.device(device).type == 'cuda' else nullcontext())
    student.eval()
    scored = []
    if token_weight:
        for index, group in enumerate(groups, 1):
            on_phase('score', microbatch=index, total_microbatches=len(groups))
            with autocast():
                ids, rewards = score(student, teacher, make_batch(group), support)
            scored.append((ids.cpu(), rewards.cpu()))
        scale = reward_scale(scored)
    else:
        scale = 1.0
        scored = [(None, None)] * len(groups)

    student.train()
    optimizer.zero_grad(set_to_none=True)
    loss_value = token_value = rep_value = 0.0
    for index, (group, (ids, rewards)) in enumerate(zip(groups, scored, strict=True), 1):
        on_phase('backward', microbatch=index, total_microbatches=len(groups))
        weight = len(group) / len(rollouts)
        with autocast():
            loss, token, rep = objective(
                student, teacher, make_batch(group), ids.to(device) if token_weight else None,
                (rewards / scale).to(device) if token_weight else None, token_weight, rep_weight, support,
            )
            loss = weight * loss
        loss.backward()
        loss_value += float(loss.detach())
        token_value += weight * float(token)
        rep_value += weight * float(rep)

    # Detect nonfinite gradients without clipping the paper's update.
    norm = torch.nn.utils.get_total_norm([p.grad for p in student.parameters() if p.grad is not None],
                                        error_if_nonfinite=True)
    on_phase('optimizer_start')
    optimizer.step()
    on_phase('optimizer_step_returned')
    return {'loss': loss_value, 'token_loss': token_value, 'raw_rep_loss': rep_value,
            'reward_std': float(scale), 'gradient_norm': float(norm), 'optimizer_updates': 1}


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_checkpoint(output, student, optimizer, identity, step, keep):
    target = output / f'checkpoint-{step}'
    with tempfile.TemporaryDirectory(prefix='.checkpoint-incomplete-', dir=output) as workspace:
        tmp = Path(workspace) / target.name
        tmp.mkdir()
        student.save_pretrained(tmp / 'student', safe_serialization=True)
        torch.save({'optimizer': optimizer.state_dict(), 'step': step, **opd.rng_state()}, tmp / 'engine.pt')
        write(tmp / 'run_identity.json', identity)
        files = {p.relative_to(tmp).as_posix(): file_digest(p) for p in tmp.rglob('*') if p.is_file()}
        write(tmp / 'checkpoint_manifest.json', files)
        tmp.rename(target)
    checkpoints = sorted(
        (p for p in output.glob('checkpoint-*') if p.is_dir() and p.name.removeprefix('checkpoint-').isdigit()),
        key=lambda p: int(p.name.removeprefix('checkpoint-')),
    )
    for obsolete in checkpoints[:-keep]:
        shutil.rmtree(obsolete)


def verify_checkpoint_files(path, *, verify_optimizer=True):
    """Verify the sealed model and optionally the training-only optimizer payload."""
    manifest = path / 'checkpoint_manifest.json'
    if path.is_symlink() or manifest.is_symlink():
        raise ValueError('Checkpoint directory and manifest must not be symbolic links')
    manifest_bytes = manifest.read_bytes()
    files = json.loads(manifest_bytes)
    entries = list(path.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Checkpoint contains a symbolic link')
    actual = {p.relative_to(path).as_posix() for p in entries if p.is_file() and p != manifest}
    required = {'engine.pt', 'run_identity.json', 'student/config.json'}
    if not isinstance(files, dict) or not required <= actual or actual != files.keys():
        raise ValueError('Checkpoint file set differs from manifest')
    for name, digest in sorted(files.items()):
        if (verify_optimizer or name != 'engine.pt') and file_digest(path / name) != digest:
            raise ValueError(f'Checkpoint digest mismatch: {name}')

    weights = {name for name in actual if name.startswith('student/') and name.endswith('.safetensors')}
    index = 'student/model.safetensors.index.json'
    if index in actual:
        index_document = read(path / index)
        weight_map = index_document.get('weight_map') if isinstance(index_document, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Invalid student safetensors index')
        if (any(not isinstance(name, str) or Path(name).name != name or '\\' in name
                for name in weight_map.values())
                or weights != {f'student/{name}' for name in weight_map.values()}
                or 'student/model.safetensors' in weights):
            raise ValueError('Student safetensors index does not match the sealed shards')
    elif weights != {'student/model.safetensors'}:
        raise ValueError('Missing or unexpected student safetensors weights')
    return hashlib.sha256(manifest_bytes).hexdigest()


def restore_checkpoint(path, identity):
    verify_checkpoint_files(path)
    check_identity(read(path / 'run_identity.json'), identity)
    saved = torch.load(path / 'engine.pt', map_location='cpu', weights_only=False, mmap=True)
    if saved['step'] != int(path.name.removeprefix('checkpoint-')):
        raise ValueError('Checkpoint step differs from directory')
    for key in ('optimizer', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'):
        if key not in saved:
            raise ValueError(f'Checkpoint is missing {key}')
    return saved


def latest_resumable_checkpoint(output, identity):
    checkpoints = sorted(
        (p for p in output.glob('checkpoint-*') if p.is_dir() and p.name.removeprefix('checkpoint-').isdigit()),
        key=lambda p: int(p.name.removeprefix('checkpoint-')), reverse=True,
    )
    for path in checkpoints:
        try:
            return path, restore_checkpoint(path, identity)
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
            print(f'[latent-mopd-three] skipping {path.name}: {error}', flush=True)
    raise ValueError(f'No valid checkpoint matches the run identity in {output}')


def run(args, tokenizer, tokenizer_path, audit, schedule, assets):
    from transformers import AutoModelForCausalLM, set_seed

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Training requires one visible CUDA GPU with BF16 support')
    set_seed(args.seed)
    device = torch.device('cuda')
    output = args.output_root / f'latent_mopd_three_{args.variant}'
    student_path = stage_weights(STUDENT, assets / 'weights', args.model_drive)
    teacher_paths = {domain: stage_weights(spec, assets / 'weights', args.model_drive)
                     for domain, spec in TEACHERS.items()}

    def load(path, dtype):
        return AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, attn_implementation='sdpa', local_files_only=True,
            device_map={'': device},
        )

    generation = {d: generation_config(tokenizer.eos_token_id, tokenizer.pad_token_id, args.max_new_tokens)
                  for d in DOMAINS}
    # Resolve sampling from the pinned base before opening any checkpoint.
    student = load(student_path, torch.float32)
    contracts = {d: resolved_contract(student, cfg, SUPPORT, device) for d, cfg in generation.items()}
    config = {'student': STUDENT, 'teachers': TEACHERS, 'implementation': 'three-teacher-full-parameter',
              'data_manifest': str(args.manifest.resolve()), 'tokenizer_path': str(tokenizer_path.resolve()),
              'tokenizer_audit': str(audit.resolve()), 'output_dir': str(output.resolve()), 'seed': args.seed,
              'variant': args.variant, 'steps': args.steps, 'prompts': args.prompts, 'responses': args.responses,
              'rollout_batch': args.rollout_batch, 'max_new_tokens': args.max_new_tokens,
              'packing_tokens': PACK_TOKENS, 'worker_responses': WORKER_RESPONSES}
    identity = build_identity(config, contracts, opd.ROOT, sources=SOURCE_FILES)
    initial = 0
    saved = None
    if args.resume:
        saved_path, saved = latest_resumable_checkpoint(output, identity)
        initial = saved['step']
        del student
        torch.cuda.empty_cache()
        student = load(saved_path / 'student', torch.float32)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    student.config.use_cache = False
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-5, betas=(0.9, 0.999),
                                  weight_decay=0.01, eps=1e-8, fused=True)
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer'])
    teachers = {d: load(path, torch.bfloat16).eval().requires_grad_(False) for d, path in teacher_paths.items()}
    engine = HFRolloutEngine(student, generation, contracts, SUPPORT, tokenizer.pad_token_id,
                             tokenizer.eos_token_id, args.rollout_batch)
    prepare_output(output, identity, initial if args.resume else None, 'archive_future_v2')
    if saved is not None:
        # Restore after initialization, which can consume RNG during a cold start.
        opd.restore_rng(saved)
        del saved
    progress = RunProgress(output, identity, device)
    last = args.stop_at_step or args.steps
    with (output / 'metrics.jsonl').open('a', encoding='utf-8') as log:
        for step in range(initial + 1, last + 1):
            domain, prompt_ids, sample_ids = schedule[step - 1]
            prefixes = [ids for ids in prompt_ids for _ in range(args.responses)]
            token, rep = loss_weights(args.variant, domain, (step - 1) // len(DOMAINS))
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            initial_stats = torch.cuda.memory_stats(device)
            progress.event(step, domain, 'rollout_start', responses=len(prefixes), token_weight=token,
                           latent_weight=rep, teacher=TEACHERS[domain])
            rollout = []
            for start in range(0, len(prefixes), args.rollout_batch):
                with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=True):
                    rollout.extend(engine.generate(prefixes[start:start + args.rollout_batch], domain))
                progress.event(step, domain, 'rollout_progress', responses=len(rollout), total=len(prefixes))
            # Return unused rollout allocations before scoring or accumulating gradients.
            torch.cuda.synchronize(device)
            before_free, _ = torch.cuda.mem_get_info(device)
            progress.event(step, domain, 'rollout_cache_release_start', driver_free_bytes=before_free)
            cache_started = time.monotonic()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            after_free, _ = torch.cuda.mem_get_info(device)
            progress.event(step, domain, 'rollout_cache_release_done', driver_free_bytes=after_free,
                           released_driver_bytes=after_free - before_free,
                           seconds=time.monotonic() - cache_started)
            record = update(student, teachers[domain], optimizer, prefixes, rollout, token, rep,
                            tokenizer.pad_token_id, device,
                            on_phase=lambda phase, **detail: progress.event(step, domain, phase, **detail))
            final_stats = torch.cuda.memory_stats(device)
            record.update(step=step, domain=domain, sample_ids=sample_ids, responses=len(rollout),
                          allocator_retries=final_stats['num_alloc_retries'] - initial_stats['num_alloc_retries'],
                          allocator_ooms=final_stats['num_ooms'] - initial_stats['num_ooms'],
                          response_tokens=sum(len(r['response_ids']) for r in rollout),
                          truncated_responses=sum(not r['ended_with_eos'] for r in rollout),
                          token_weight=token, latent_weight=rep, seconds=time.monotonic() - started,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            print(json.dumps(record), file=log, flush=True)
            progress.event(step, domain, 'optimizer_done', **{k: v for k, v in record.items()
                                                               if k not in ('step', 'domain')})
            if step == 1 or step % args.checkpoint_every == 0 or step == last:
                progress.event(step, domain, 'checkpoint_start')
                save_checkpoint(output, student, optimizer, identity, step, args.keep_checkpoints)
                progress.event(step, domain, 'checkpoint_saved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=('audit', 'token', 'latent'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get('OPD_OUTPUT_ROOT', 'runs')))
    parser.add_argument('--model-drive', default=os.environ.get('OPD_MODEL_DRIVE', 'mydrive:models'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prompts', type=int, default=PROMPTS)
    parser.add_argument('--responses', type=int, default=RESPONSES)
    parser.add_argument('--steps', type=int, default=STEPS)
    parser.add_argument('--max-new-tokens', type=int, default=MAX_RESPONSE)
    parser.add_argument('--rollout-batch', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--keep-checkpoints', type=int, default=2)
    parser.add_argument('--stop-at-step', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.prompts, args.responses, args.steps, args.max_new_tokens, args.rollout_batch,
           args.checkpoint_every, args.keep_checkpoints) < 1:
        parser.error('Training budgets must be positive')
    if args.steps > STEPS or (args.stop_at_step is not None and not 1 <= args.stop_at_step <= args.steps):
        parser.error('Step budget exceeds the Table 1 schedule')
    if args.variant != 'audit' and args.manifest is None:
        parser.error('Training requires a released manifest')
    assets = opd.ROOT / 'runs/latent_mopd_three_assets'
    tok, tokenizer_path, audit = tokenizer_audit(assets)
    schedule = read_schedule(args.manifest, tok, args.steps, args.prompts) if args.manifest else []
    if args.variant == 'audit':
        print(json.dumps({'tokenizer_audit': str(audit), 'common_token_ids': SUPPORT,
                          'teachers': TEACHERS, 'approved_steps': len(schedule)}, ensure_ascii=False))
    else:
        run(args, tok, tokenizer_path, audit, schedule, assets)


if __name__ == '__main__':
    main()
