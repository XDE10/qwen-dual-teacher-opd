"""Fork a sealed three-teacher checkpoint when the completed data prefix agrees."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer

from src.training.latent_mopd_three import PROMPTS, SOURCE_FILES, STEPS, read_schedule, restore_checkpoint
from src.training.opd_v3 import ROOT
from src.training.runtime_v2 import read, sha, write
from src.training.runtime_v3 import build_identity, check_identity


TRAINER = 'src/training/latent_mopd_three.py'
ORIGINAL_TRAINER_SHA = 'e7f774918b98b550b2220f579d667af0a14babcaeb3064c4ee9c71aab2cb8e5a'
VOCAB_FIX_TRAINER_SHA = '5993d35e69d372f609ad123cd0e64fbeeb5a128b32c127cf13849f58924882d7'
ORIGINAL_SEAL = '10f3be926c88e5aadcb351ce1017ab9e6b9067278438c0d0050ffca486701b51'
ORIGINAL_METRICS_SHA = '10ebdec7828524a0c4949a235a0571e4590ce135b94062c79e8d56085d19d431'
ORIGINAL_RELEASE_SHA = '071898e06aa54ad0471f3c949628d36d92f735e929be26c859f3869ff38c0499'
POOL_FIX_RELEASE_SHA = 'e5f4c579f8a4514795e6ba80e1e4c2df034e724346bbfce7a2ea150d7d969252'
ARCHIVE_ROOT = Path('/content/project2')


def expected_vocab_fix(identity, source, old_manifest, new_manifest, original_trainer, local_cpu):
    hard = identity['hard']
    if (source.name != 'checkpoint-1' or sha(source / 'checkpoint_manifest.json') != ORIGINAL_SEAL
            or sha(source.parent / 'metrics.jsonl') != ORIGINAL_METRICS_SHA
            or sha(old_manifest) != ORIGINAL_RELEASE_SHA or sha(new_manifest) != POOL_FIX_RELEASE_SHA
            or hard['source_files'].get(TRAINER) != ORIGINAL_TRAINER_SHA
            or sha(original_trainer) != ORIGINAL_TRAINER_SHA or sha(ROOT / TRAINER) != VOCAB_FIX_TRAINER_SHA):
        raise ValueError('Vocabulary migration requires the pinned checkpoint-1, releases and trainer hashes')
    expected = {**hard, 'source_files': {**hard['source_files'], TRAINER: VOCAB_FIX_TRAINER_SHA}}
    if local_cpu:
        if hard['pinned_packages']['torch'] != '2.8.0+cu128':
            raise ValueError('Offline replay requires the archived CUDA 2.8.0 identity')
        config = dict(hard['config'])
        for field in ('tokenizer_path', 'tokenizer_audit'):
            config[field] = str(ROOT / Path(config[field]).relative_to(ARCHIVE_ROOT))
        config.update(data_manifest=str(old_manifest), output_dir=str(source.parent))
        expected.update(config=config, pinned_packages={**hard['pinned_packages'], 'torch': '2.8.0+cpu'})
    return {**identity, 'hard': expected}


def migrate(source, old_manifest, new_manifest, destination, *, original_trainer=None, local_cpu=False):
    source, old_manifest, new_manifest, destination = (
        path.resolve() for path in (source, old_manifest, new_manifest, destination)
    )
    step = int(source.name.removeprefix('checkpoint-'))
    if source.name != f'checkpoint-{step}' or not 1 <= step < STEPS:
        raise ValueError('Source must be a numbered checkpoint before the final update')
    if destination.exists():
        raise FileExistsError(destination)
    if local_cpu and original_trainer is None:
        raise ValueError('Local CPU replay is restricted to the pinned vocabulary migration')
    identity = read(source / 'run_identity.json')
    check_identity(read(source.parent / 'run_identity.json'), identity)
    hard = identity['hard']
    expected = (expected_vocab_fix(identity, source, old_manifest, new_manifest, original_trainer, local_cpu)
                if original_trainer is not None else identity)
    config = expected['hard']['config']
    if (config.get('data_manifest') != str(old_manifest) or config.get('output_dir') != str(source.parent)
            or config.get('steps') != STEPS or config.get('prompts') != PROMPTS
            or config.get('variant') != 'latent' or config.get('responses') != 4
            or hard['manifest_sha'] != sha(old_manifest)):
        raise ValueError('Checkpoint does not identify the expected original 62-step run')
    if hard['manifest_sha'] == sha(new_manifest):
        raise ValueError('Expected a distinct approved release')

    current = build_identity(config, hard['generation'], ROOT, sources=SOURCE_FILES)
    check_identity(expected, current)
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'], local_files_only=True)
    old_schedule = read_schedule(old_manifest, tokenizer, STEPS, PROMPTS)
    new_schedule = read_schedule(new_manifest, tokenizer, STEPS, PROMPTS)
    if old_schedule[:step] != new_schedule[:step]:
        raise ValueError('The two releases differ in an update already applied to the checkpoint')
    completed = []
    with (source.parent / 'metrics.jsonl').open(encoding='utf-8') as log:
        for number in range(1, step + 1):
            row = json.loads(log.readline())
            domain, _, sample_ids = old_schedule[number - 1]
            if (row['step'] != number or row['domain'] != domain or row['sample_ids'] != sample_ids
                    or row['responses'] != 128 or row['optimizer_updates'] != 1):
                raise ValueError(f'Training log does not match the agreed data prefix at step {number}')
            if original_trainer is not None and (row['token_weight'] != 0 or row['latent_weight'] != 1):
                raise ValueError('The completed update was not purely Latent')
            completed.append(row)

    # Validate and load the sealed source under its archived identity.
    restore_checkpoint(source, identity)
    new_config = {**config, 'data_manifest': str(new_manifest), 'output_dir': str(destination)}
    migrated = build_identity(new_config, hard['generation'], ROOT, sources=SOURCE_FILES)
    expected_target = {**expected, 'hard': {**expected['hard'], 'config': new_config,
                                          'manifest_sha': sha(new_manifest)}}
    check_identity(migrated, expected_target)
    original_files = read(source / 'checkpoint_manifest.json')

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.latent-three-fork-', dir=destination.parent) as workspace:
        prepared = Path(workspace) / destination.name
        checkpoint = prepared / source.name
        checkpoint.mkdir(parents=True)
        for name in original_files:
            if name == 'run_identity.json':
                continue
            copy = checkpoint / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            os.link(source / name, copy)
        write(checkpoint / 'run_identity.json', migrated)
        write(checkpoint / 'checkpoint_manifest.json',
              {**original_files, 'run_identity.json': sha(checkpoint / 'run_identity.json')})
        write(prepared / 'run_identity.json', migrated)
        (prepared / 'metrics.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in completed), encoding='utf-8',
        )
        receipt = {
            'step': step,
            'next_step': step + 1,
            'source_checkpoint': str(source),
            'source_checkpoint_manifest_sha256': sha(source / 'checkpoint_manifest.json'),
            'source_release_sha256': hard['manifest_sha'],
            'new_release_sha256': migrated['hard']['manifest_sha'],
            'matching_completed_token_prefixes': step * PROMPTS,
            'immutable_state_hardlinked': True,
        }
        if original_trainer is not None:
            receipt.update(
                vocabulary_source_transition={'path': TRAINER, 'original_sha256': ORIGINAL_TRAINER_SHA,
                                              'repaired_sha256': VOCAB_FIX_TRAINER_SHA},
                pure_latent_completed_update=True,
                local_cpu_verification_only=local_cpu,
            )
        write(prepared / 'migration.json', receipt)
        prepared.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'old-manifest', 'new-manifest', 'destination'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--vocab-fix-original-trainer', type=Path,
                        help='Exact archived trainer from the original G4 bundle')
    parser.add_argument('--local-cpu-verification', action='store_true',
                        help='Create a local CPU verification fork, with an identity unsuitable for G4 resume')
    args = parser.parse_args()
    migrate(args.source, args.old_manifest, args.new_manifest, args.destination,
            original_trainer=args.vocab_fix_original_trainer, local_cpu=args.local_cpu_verification)
