"""Read real pinned generation/config metadata ONLY. Never downloads weights.

Not executed in local handover. Optional --allow-config-fetch permits tiny JSON
metadata downloads on Colab; without it only cached metadata may be read.
"""
import argparse
import json
from pathlib import Path
from transformers import AutoConfig, GenerationConfig
from src.training.sampling_v2 import generation_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/project2_opd_smoke.json')
    p.add_argument('--allow-config-fetch', action='store_true')
    p.add_argument('--output', default='reports/handover_20261024_v2/colab_real_generation_configs.json')
    args = p.parse_args()
    c = json.loads(Path(args.config).read_text(encoding='utf-8'))
    result = []
    for spec in [c['student'], *c['teachers'].values()]:
        kwargs = {'revision': spec['revision'], 'local_files_only': not args.allow_config_fetch}
        try:
            gc = GenerationConfig.from_pretrained(spec['model'], **kwargs)
            source = 'generation_config.json'
        except OSError:
            gc = GenerationConfig.from_model_config(AutoConfig.from_pretrained(spec['model'], **kwargs))
            source = 'config.json fallback; no generation_config.json located'
        item = {'model': spec, 'source': source, 'raw_generation_config': gc.to_dict(),
                'v2_requested_sampling': generation_config(gc.eos_token_id, gc.pad_token_id, 32).to_dict(),
                'use_model_defaults': False, 'weights_loaded': False,
                'actual_model_processor_chain': '未验证: printed by trainer after real model load'}
        result.append(item)
        print(json.dumps(item, ensure_ascii=False, indent=2))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
