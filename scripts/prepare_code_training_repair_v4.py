"""Read-only reserve inventory and admission checks for ONE replacement prompt."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from src.data.decontamination_v2 import Index
from src.data.screening_io import ScreeningSimilarityIndex
from src.data.code_v2_dedup import Protected
from src.teachers.opd_contract import render_student_prompt
from src.training.runtime_v2 import sha, write

OUT = Path('reports/code_training_repair_20261025_v4')
RESERVE = Path('data/topup_v6_20261024/project2_train_reserve')
SELECTED = 'code/vfc_96076'


def pinned(path):
    """Resolve pinned evidence; missing paths fall back to the 2026-11-08 relocation map."""
    path = Path(path)
    if path.exists():
        return path
    mapping = json.loads(Path('docs/migration/path_mapping.json').read_text(encoding='utf-8'))
    for entry in mapping['entries']:
        if entry['original'] == path.as_posix() and entry.get('new_location') and Path(entry['new_location']).exists():
            return Path(entry['new_location'])
    raise FileNotFoundError(f'pinned evidence not found: {path.as_posix()}')


def rows(path):
    # JSON strings may contain literal U+2028/U+2029. They are not JSONL record separators.
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def heldout_records():
    sources=['data/portable_release/project2_code_dev.jsonl',
        'data/project2_code/dev_verification.jsonl','data/project2_code_v2/dev_verification.jsonl',
        'data/project2_code_v3_dev_verified/dev_verification.jsonl',
        'data/topup_v6_20261024/project2/candidates.jsonl',
        'data/topup_v6_20261024/project2_dev_reserve/candidates.jsonl',
        'data/delegated_ai_releases_20261024/project2_fixed_dev_audit.jsonl']
    heldout=[]
    for filename in sources:
        for r in rows(filename):
            if r.get('domain') not in {None,'code'} or r.get('split')=='train':continue
            if r.get('problem'):heldout.append({'id':r['sample_id'],'text':r['problem']})
    for filename in ['data/code_benchmarks/official/HumanEvalPlus-v0.1.10.jsonl','data/code_benchmarks/official/MbppPlus-v0.2.0.jsonl']:
        heldout.extend({'id':r['task_id'],'text':r['prompt']} for r in rows(filename))
    return list({(r['id'],r['text']):r for r in heldout}.values())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RESERVE/'manifest.json').read_text(encoding='utf-8'))
    assert all(sha(RESERVE/name)==digest for name,digest in manifest['files'].items())
    current = rows('data/portable_release/project2_code_train.jsonl')
    current_ids = {r['sample_id'] for r in current}
    protected = set(json.loads(Path('data/portable_release/reserved_dev_ids_never_train.json').read_text(encoding='utf-8'))['sample_ids'])
    reviews = {r['id']:r for filename in ['reports/topup_v6_20261024/p2_review.jsonl','reports/topup_v6_20261024/p2_dev_reserve_review.jsonl'] for r in rows(filename)}
    candidates = rows(RESERVE/'candidates.jsonl')
    surplus = {r['sample_id']:r['reason'] for r in rows('data/topup_v6_20261024/release/surplus_and_final_exclusions.jsonl')}
    passed = [r for r in candidates if reviews.get('code/train/'+r['sample_id'],{}).get('verdict')=='pass']
    unused = [r for r in passed if r['sample_id'] not in current_ids|protected]
    write(OUT/'reserve_inventory.json',{'source_manifest_sha256':sha(RESERVE/'manifest.json'),
        'reserve_rows':len(candidates),'historical_pass':len(passed),'unused_nondev_pass':len(unused),
        'unused':[{'sample_id':r['sample_id'],'historical_export_reason':surplus.get(r['sample_id']),
                   'historical_request_id':reviews['code/train/'+r['sample_id']].get('request_id')} for r in unused],
        'scope':'Unused reviewed reserve is not automatically training-approved; new admission required.'})
    row = next(r for r in unused if r['sample_id']==SELECTED)
    review = reviews['code/train/'+SELECTED]
    request = next(r for r in rows(RESERVE/'audit_requests.jsonl') if r['record_key']=='code/train/'+SELECTED)
    assert review['valid_schema'] and review['verdict']=='pass' and not review['issues']
    assert review['requires_visual'] is False and review['answer_leak'] is False
    assert row['split']=='train' and row['interface']=='stdin_stdout' and row['training_approved'] is False
    assert surplus[SELECTED]=='train_interface_quota_surplus'
    assert request['reference_code']==row['reference_code']
    assert request['problem']==row['prompt']
    assert len(json.dumps({**request,'id':request['record_key']},ensure_ascii=False))<=60000
    code_tree = ast.parse(row['reference_code'])
    assert hashlib.sha256(ast.dump(code_tree,include_attributes=False).encode()).hexdigest()==row['reference_ast_sha256']
    cases = row['verification_info']['test_cases']
    assert len(cases)>=3 and len({c['input'] for c in cases})>=3
    assert len({tuple(c['output'].split()) for c in cases})>=2
    tokenizer_folder=Path('data/tokenizers/Qwen2.5-1.5B-Instruct')
    tokenizer_audit=json.loads(Path('reports/tokenizer_alignment_report.json').read_text(encoding='utf-8'))
    student_audit=next(m for m in tokenizer_audit['models'] if m['model']=='Qwen/Qwen2.5-1.5B-Instruct')
    assert all(sha(tokenizer_folder/name)==digest for name,digest in student_audit['files'].items())
    tok=AutoTokenizer.from_pretrained(tokenizer_folder,local_files_only=True)
    prompt_ids=render_student_prompt(tok,row['prompt'],'code')
    assert len(prompt_ids)<=2560
    assert all(0<=token<tokenizer_audit['common_support']['size'] for token in prompt_ids)
    # Keep ALL current/old/dev-candidate prompts in the held-out comparison set.
    heldout=heldout_records()
    lexical=Index(heldout).query(row['problem'],.5)
    protected_hit=Protected().match(row)
    old_index=ScreeningSimilarityIndex(.85)
    for r in current:old_index.add(r)
    training_hit=old_index.match(row['problem'])
    same_fields={k:[r['sample_id'] for r in current if row.get(k) and r.get(k)==row[k]]
                 for k in ['family_id','problem_hash','reference_ast_sha256']}
    checks={'schema':'code-repair-admission-v4','sample_id':SELECTED,
        'historical_review_pass':True,'historical_export_reason':surplus[SELECTED],
        'prompt_tokens':len(prompt_ids),'generation_reserved':1536,'total_limit':4096,
        'student_tokenizer_files_match_audit':True,
        'prompt_prefix_sha256':hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest(),
        'distinct_source_test_inputs':len({c['input'] for c in cases}),'source_test_count':len(cases),
        'heldout_prompt_versions':len(heldout),'heldout_5gram_numeric_containment_at_0_5':lexical,
        'legacy_protected_check':protected_hit,'current_training_3gram_at_0_85':training_hit,
        'current_training_same_identity_fields':same_fields,
        'embedding':'未执行 / 未验证; no pinned local model, no weight downloads',
        'source_revision':row['source_revision'],'historical_request_id':review['request_id'],
        'source_hashes':{Path(p).as_posix():sha(pinned(p)) for p in [str(RESERVE/'manifest.json'),str(RESERVE/'candidates.jsonl'),str(RESERVE/'audit_requests.jsonl'),
            'reports/topup_v6_20261024/p2_dev_reserve_review.jsonl','data/portable_release/manifest.json',
            'reports/tokenizer_alignment_report.json','project_blueprints/configs/project2_prompt_contract.json']},
        'candidate_row_sha256':hashlib.sha256(json.dumps(row,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()}
    write(OUT/'admission_static.json',checks)
    write(OUT/'candidate_private.json',row)
    write(OUT/'historical_review.json',review)
    write(OUT/'historical_request.json',request)
    assert not lexical and not protected_hit and not training_hit and not any(same_fields.values()),checks
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
