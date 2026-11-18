"""Offline repository-shape acceptance: layout, relocation map, provenance bytes.

Stdlib only; no network, no training. Run from the project root:
    D:/anaconda3/envs/genai-test/python.exe tools/verify_repository.py
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

TOP_LEVEL_ALLOWED = {'.gitignore', 'AGENTS.md', 'README.md', 'pytest.ini', 'requirements',
                     'configs', 'data', 'docs', 'provenance', 'reports', 'scripts', 'src',
                     'tests', 'tools', 'runs'}
TOP_LEVEL_REQUIRED = TOP_LEVEL_ALLOWED - {'runs'}
SRC_PACKAGES = {'data', 'evaluation', 'losses', 'teachers', 'training'}
DATA_DIRS = {'portable_release', 'code_training_repair_20261025_v4', 'tokenizers', 'math', 'project2_release_v5_20261108'}
# Restored on demand by scripts/restore_official_benchmarks.py (gitignored, sha256-verified).
DATA_OPTIONAL = {'code_benchmarks'}
# v5 is built as a candidate; release/ appears only after the owner approval command.
PENDING_MANIFESTS = {'data/project2_release_v5_20261108/release/manifest.json':
                     'data/project2_release_v5_20261108/candidate/candidate_manifest.json'}
REPORT_DIRS = {'code_training_repair_20261025_v4', 'handover_20261024_v3', 'project2_release_v5_20261108',
               'g4_first_session_20261109', 'g4_session2_lr_sweep_20261109', 'g4_opd_v3_experiments_20261112',
               'wsl_evalplus_crash_fix_20261113', 'argon_latent_mopd_audit_20261115',
               'latent_mopd_three_status_20261118'}
REPORT_LOCAL_ONLY = {'argon_latent_mopd_papers_20261115.zip'}
REPORT_ROOT_FILES = {'README.md', 'tokenizer_alignment_report.json', 'tokenizer_alignment_report_14b.json',
                     'tokenizer_alignment_revalidation_20261023.json',
                     'tokenizer_alignment_summary.md', 'project2_prompt_contract_audit.json',
                     'code_v2_sandbox_environment.json', 'code_v2_sandbox_selftest.json',
                     'code_interface_screening_report.json', 'official_code_overlap_audit.json',
                     'decontamination_20261024.json', 'data_human_review_pack.md', 'data_human_review_pack_v4.md',
                     'cpu_demo.json', 'cpu_environment.json', 'review_of_ultrathink_opd_report_20261111.md',
                     'latent_mopd_three_teacher_reproduction_plan_20261114.md'}
CONFIGS = {'project2_opd_smoke.json', 'project2_opd_v4_smoke.json', 'project2_opd_v4_math_short.json',
           'project2_opd_v4_code_short.json', 'project2_prompt_contract.json',
           *(f'project2_opd_v3_{n}.json' for n in ['smoke', 'd1_math', 'd2_code', 'd3_general', 'd4_dual',
                                                     'd4_blocked', 'd4_pcgrad', 'd4_pg', 'd4_topk', 'd4_warmstart',
                                                     'd4_dual_lr1e-5', 'd4_dual_lr1e-4', 'd4_prox07',
                                                     'd1_math_lr1e-4', 'd2_code_lr1e-4', 'd3_general_lr1e-4',
                                                     'd4_blocked_lr1e-4', 'd4_warmstart_lr1e-4',
                                                     'd4_prox07_lr1e-4', 'd4_aligned_lr1e-4', 'd4_math14b_lr1e-4']),
           'project2_sft_v3_seqkd_mix.json'}
TEST_FILES = 24
DOCS_REQUIRED = ['README.md', 'plan/README.md', 'handover/v2/README.md',
                 'migration/reorg_20261108.md', 'migration/path_mapping.json',
                 'migration/inventory_before_reorg.json', 'migration/deletion_ledger.json']
LEGACY_TOP_LEVEL = {'agentic_rl', 'math', 'video_mapping', 'tiny_posttrain', 'project_blueprints',
                    '.pilot_deps', '迁移记录', 'migration_manifest.json',
                    'ARGON_深度审查提示词.md', 'OVERNIGHT_CPU_README.md'}
LEGACY_PATTERNS = [r'D:/Users/', r'D:\\\\Users', r'D:/anaconda3']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    errors = []
    notes = []

    for name in sorted(TOP_LEVEL_REQUIRED):
        if not (ROOT / name).exists():
            errors.append(f'missing top-level entry: {name}')
    present = {p.name for p in ROOT.iterdir() if p.name not in {'__pycache__', '.pytest_cache', '.git', '.agents', 'runs'}}
    for name in sorted(present - TOP_LEVEL_ALLOWED):
        errors.append(f'unexpected top-level entry: {name}')
    for name in sorted(LEGACY_TOP_LEVEL & present):
        errors.append(f'legacy top-level entry still present: {name}')

    packages = {p.name for p in (ROOT / 'src').iterdir() if p.is_dir() and p.name != '__pycache__'}
    for name in sorted(packages - SRC_PACKAGES):
        errors.append(f'unexpected src package: {name}')
    for name in sorted(SRC_PACKAGES - packages):
        errors.append(f'missing src package: {name}')

    data_present = {p.name for p in (ROOT / 'data').iterdir() if p.name != '__pycache__'}
    for name in sorted(data_present - DATA_DIRS - DATA_OPTIONAL - {'README.md', 'math_manifest.json'}):
        errors.append(f'unexpected data entry: {name}')
    for name in sorted(DATA_DIRS):
        if name not in data_present:
            errors.append(f'missing data entry: {name}')

    reports_present = {p.name for p in (ROOT / 'reports').iterdir()
                       if p.name != '__pycache__' and p.name not in REPORT_LOCAL_ONLY}
    for name in sorted(reports_present - REPORT_DIRS - REPORT_ROOT_FILES):
        errors.append(f'unexpected reports entry: {name}')
    for name in sorted(REPORT_DIRS | REPORT_ROOT_FILES):
        if name not in reports_present:
            errors.append(f'missing reports entry: {name}')

    configs_present = {p.name for p in (ROOT / 'configs').glob('*.json')}
    for name in sorted(CONFIGS ^ configs_present):
        errors.append(f'configs mismatch: {name}')

    tests = sorted(p.name for p in (ROOT / 'tests').glob('test_*.py'))
    if len(tests) != TEST_FILES:
        errors.append(f'expected {TEST_FILES} test files, found {len(tests)}')
    notes.append(f'tests: {len(tests)} files')

    for rel in DOCS_REQUIRED:
        if not (ROOT / 'docs' / rel).exists():
            errors.append(f'missing doc: docs/{rel}')
    if not (ROOT / 'provenance' / 'README.md').exists():
        errors.append('missing provenance/README.md')

    mapping = read('docs/migration/path_mapping.json')
    entries = mapping['entries']
    if len(entries) != 70:
        errors.append(f'path_mapping: expected 70 entries, found {len(entries)}')
    statuses = {}
    for entry in entries:
        status = entry['status']
        statuses[status] = statuses.get(status, 0) + 1
        original, new = entry['original'], entry.get('new_location')
        if status == 'missing_before_reorg':
            if Path(original).exists():
                errors.append(f'{original}: recorded missing_before_reorg but now exists')
            continue
        if not new or not Path(new).exists():
            errors.append(f'{original}: relocation target missing ({new})')
            continue
        actual = sha(new)
        if actual != entry['expected_sha256']:
            errors.append(f'{original}: relocated sha256 mismatch at {new}')
        if status == 'mirrored' and Path(new).as_posix() != f'provenance/{Path(original).as_posix()}':
            errors.append(f'{original}: mirrored copy outside provenance/ ({new})')
    notes.append('path_mapping statuses: ' + json.dumps(statuses, ensure_ascii=False, sort_keys=True))

    if not (ROOT / 'tools' / 'verify_code_repair_v4.py').exists():
        errors.append('missing tools/verify_code_repair_v4.py')
    if not (ROOT / 'tools' / 'verify_repository.py').exists():
        errors.append('missing tools/verify_repository.py')

    for config_path in sorted((ROOT / 'configs').glob('*.json')):
        config = read(config_path)
        for key in ('data_manifest', 'tokenizer_path', 'tokenizer_audit'):
            value = config.get(key)
            if value and not Path(value).exists():
                if key == 'data_manifest' and value in PENDING_MANIFESTS and Path(PENDING_MANIFESTS[value]).exists():
                    note = f'{value}: pending owner approval (candidate present)'
                    if note not in notes:
                        notes.append(note)
                    continue
                errors.append(f'{config_path.name}: {key} does not exist ({value})')
        budget = config.get('math_input_budget')
        if isinstance(budget, dict):
            for value in budget.values():
                if isinstance(value, str) and not Path(value).exists():
                    errors.append(f'{config_path.name}: budget input missing ({value})')

    contract = 'configs/project2_prompt_contract.json'
    if not Path(contract).exists():
        errors.append('missing prompt contract: ' + contract)
    for rel in ['src/teachers/opd_contract.py', 'src/training/runtime_v2.py']:
        if contract not in Path(rel).read_text(encoding='utf-8'):
            errors.append(f'{rel} does not reference {contract}')

    for path in sorted(list(Path('src').rglob('*.py')) + list(Path('scripts').rglob('*.py')) +
                       list(Path('tests').rglob('*.py')) + list(Path('configs').rglob('*.json')) +
                       list(Path('src').rglob('*.sh')) + list(Path('scripts').rglob('*.sh'))):
        text = path.read_text(encoding='utf-8', errors='replace')
        for pattern in LEGACY_PATTERNS:
            if re.search(pattern, text):
                errors.append(f'{path.as_posix()}: legacy absolute path pattern {pattern}')
                break

    tier = next((p for p in Path('data/tokenizers').iterdir() if p.is_dir()), None)
    if tier is None:
        errors.append('data/tokenizers has no tokenizer snapshot')

    result = {'layout_ok': not errors, 'top_level': sorted(present), 'src_packages': sorted(packages),
              'tests': len(tests), 'path_mapping_entries': len(entries), 'path_mapping_statuses': statuses,
              'notes': notes, 'errors': errors}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    sys.exit(main())
