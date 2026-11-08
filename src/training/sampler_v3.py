"""Seeded, stratified, resumable per-domain prompt sampler (opd-v3).

v2 walked each approved file in file order. The v4 math file holds 1419 MATH rows
before any GSM8K row and the code file holds 908 function rows before any stdin
row, so short runs never saw GSM8K or stdin tasks. This sampler fixes that.
"""
import hashlib
import json
import random
from collections import defaultdict


def order_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False).encode()).hexdigest()


class StratifiedSampler:
    """Interleave strata so every prefix of an epoch keeps the pool's stratum mix.

    Each stratum is shuffled with a seed derived from (seed, domain, epoch), its
    k-th member gets rank (k + u) / n with one random offset u per stratum, and
    all members are merged by rank. The order is a pure function of the rows,
    so a resume only needs (epoch, cursor) plus a digest check.
    """

    def __init__(self, rows, domain, seed, key):
        self.rows = list(rows)
        if not self.rows:
            raise ValueError('Empty sampler pool for ' + domain)
        if len({r['sample_id'] for r in self.rows}) != len(self.rows):
            raise ValueError('Duplicate sample IDs in sampler pool')
        self.domain, self.seed, self.key = domain, seed, key
        self.epoch, self.cursor = 0, 0
        self._order = self._build(0)

    def _build(self, epoch):
        rng = random.Random(f'opd-v3/{self.seed}/{self.domain}/{epoch}')
        strata = defaultdict(list)
        for index, row in enumerate(self.rows):
            strata[str(row.get(self.key))].append(index)
        ranked = []
        for name in sorted(strata):
            members = strata[name]
            rng.shuffle(members)
            offset = rng.random()
            ranked.extend(((k + offset) / len(members), name, index) for k, index in enumerate(members))
        ranked.sort()
        return [index for _, _, index in ranked]

    def epoch_ids(self):
        return [self.rows[i]['sample_id'] for i in self._order]

    def next(self, count):
        out = []
        while len(out) < count:
            if self.cursor >= len(self._order):
                self.epoch += 1
                self.cursor = 0
                self._order = self._build(self.epoch)
            out.append(self.rows[self._order[self.cursor]])
            self.cursor += 1
        return out

    def state(self):
        return {'epoch': self.epoch, 'cursor': self.cursor, 'pool_size': len(self.rows),
                'stratify_key': self.key, 'order_sha256': order_digest(self.epoch_ids())}

    def load_state(self, state):
        if state['pool_size'] != len(self.rows) or state['stratify_key'] != self.key:
            raise ValueError('Sampler pool changed since checkpoint')
        self.epoch, self.cursor = state['epoch'], state['cursor']
        self._order = self._build(self.epoch)
        if order_digest(self.epoch_ids()) != state['order_sha256']:
            raise ValueError('Sampler order digest mismatch on resume')
