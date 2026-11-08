"""Domain-routed frozen teachers for opd-v3.

``resident``: every distinct teacher stays on the GPU. Two 7B bf16 teachers take
about 28.4 GiB, so this fits on 80/96 GB cards and removes all switching.
``cpu_swap``: for smaller cards; teachers wait in host memory and the routed
one is moved in at the domain boundary inside a step, never from disk.
Teachers sharing (model, revision) are loaded once, e.g. the D3 general teacher.
"""
import gc
import time

from src.training.runtime_v2 import teacher_key

PLACEMENTS = ('resident', 'cpu_swap')


class TeacherPool:
    def __init__(self, specs, loader, device, placement='resident'):
        if placement not in PLACEMENTS:
            raise ValueError('Unknown teacher placement ' + placement)
        self.device, self.placement = device, placement
        self.route = {domain: teacher_key(spec) for domain, spec in specs.items()}
        self.models, self.active, self.swaps, self.load_seconds = {}, None, 0, {}
        for domain, spec in specs.items():
            key = self.route[domain]
            if key in self.models:
                continue
            start = time.monotonic()
            model = loader(spec)
            model.eval().requires_grad_(False)
            self.models[key] = model.to(device if placement == 'resident' else 'cpu')
            self.load_seconds['/'.join(key)] = time.monotonic() - start
        if placement == 'resident':
            self.active = set(self.models)

    def keys(self):
        return sorted(self.models)

    def get(self, domain):
        key = self.route[domain]
        if self.placement == 'resident':
            return self.models[key]
        if self.active != key:
            if self.active is not None:
                self.models[self.active].to('cpu')
                gc.collect()
                if str(self.device).startswith('cuda'):
                    import torch
                    torch.cuda.empty_cache()
            self.models[key].to(self.device)
            self.active = key
            self.swaps += 1
        return self.models[key]
