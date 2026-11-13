"""Phase telemetry and replayable rollout batches for the Latent-MOPD run."""

import gzip
import hashlib
import json
import os
import tempfile
import time

import torch


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


class RunProgress:
    def __init__(self, output, identity, device):
        self.output = output
        self.run_id = digest(identity['hard'])
        self.seed = identity['hard']['config']['seed']
        self.device = torch.device(device)

    def event(self, step, domain, phase, **detail):
        row = {'step': step, 'domain': domain, 'phase': phase, 'timestamp': time.time(), **detail}
        if self.device.type == 'cuda':
            row.update(allocated_bytes=torch.cuda.memory_allocated(self.device),
                       reserved_bytes=torch.cuda.memory_reserved(self.device))
        line = json.dumps(row)
        with (self.output / 'phase_metrics.jsonl').open('a', encoding='utf-8') as log:
            print(line, file=log)
        print(line, flush=True)

    def rollouts(self, engine, prefixes, step, domain):
        folder = self.output / 'rollout_spool' / f'step-{step:06d}'
        folder.mkdir(parents=True, exist_ok=True)
        order = sorted(range(len(prefixes)), key=lambda i: (len(prefixes[i]), i))
        results = [None] * len(prefixes)
        for number, start in enumerate(range(0, len(order), engine.batch_size), 1):
            indices = order[start:start + engine.batch_size]
            batch = [prefixes[i] for i in indices]
            path = folder / f'batch-{number:04d}.json.gz'
            key = digest([self.run_id, step, domain, indices, batch])
            self.event(step, domain, 'rollout_batch_start', batch=number, done_responses=start,
                       total_responses=len(prefixes))
            began = time.monotonic()
            rollouts = self._read_batch(path, key, batch, domain)
            reused = rollouts is not None
            if not reused:
                # Each batch has an independent RNG stream so missing batches can be replayed.
                seed = int(digest([self.seed, step, domain, number])[:16], 16)
                devices = [self.device] if self.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    rollouts = engine.generate(batch, domain)
                if len(rollouts) != len(batch):
                    raise ValueError('Rollout count differs from the sampled batch')
                self._write_batch(path, key, rollouts)
            for i, rollout in zip(indices, rollouts):
                results[i] = rollout
            lengths = sorted(len(row['response_ids']) for row in rollouts)
            self.event(step, domain, 'rollout_batch_done', batch=number, reused=reused,
                       done_responses=start + len(batch), total_responses=len(prefixes),
                       response_tokens=sum(lengths), max_response_tokens=lengths[-1],
                       truncated_responses=sum(not row['ended_with_eos'] for row in rollouts),
                       seconds=time.monotonic() - began)
        return results

    @staticmethod
    def _read_batch(path, key, prefixes, domain):
        if not path.is_file():
            return None
        try:
            with gzip.open(path, 'rb') as file:
                record = json.load(file)
            rows = record['rollouts']
            if record['key'] == key and record['sha256'] == digest(rows) and len(rows) == len(prefixes):
                if all(row['prompt_ids'] == prefix and row['domain'] == domain and row['response_ids']
                       for row, prefix in zip(rows, prefixes)):
                    return rows
        except (OSError, EOFError, ValueError, KeyError, TypeError):
            pass
        return None

    @staticmethod
    def _write_batch(path, key, rollouts):
        data = encode({'key': key, 'sha256': digest(rollouts), 'rollouts': rollouts})
        blob = gzip.compress(data, compresslevel=3, mtime=0)
        file = tempfile.NamedTemporaryFile(dir=path.parent, prefix='.batch-', delete=False)
        try:
            with file:
                file.write(blob)
            os.replace(file.name, path)
        finally:
            if os.path.exists(file.name):
                os.unlink(file.name)
