import copy
from collections import Counter
import json
from pathlib import Path

import pytest

from scripts.prepare_code_training_repair_v4 import rows
from scripts.validate_code_training_repair_v4 import oracle_cases
from src.data.code_training_repair_v4 import admit,replace_one,REPLACEMENT_ID,REMOVED_ID,PRIVATE_FIELDS
from src.data.approved_training_inputs import load_approved,model_view
from src.training.runtime_v2 import read,sha

OUT=Path('reports/code_training_repair_20261025_v4')


def evidence():
    return [read(OUT/p) for p in ['candidate_private.json','admission_static.json','historical_review.json','source_verification.json','execution.json']]


def test_admission_requires_real_evidence_and_does_not_rewrite_candidate():
    values=evidence()
    before=copy.deepcopy(values[0])
    exported=admit(*values)
    assert values[0]==before and values[0]['training_approved'] is False
    assert exported['training_approved'] is True and exported['sample_id']==REPLACEMENT_ID
    assert not PRIVATE_FIELDS & exported.keys()
    assert model_view(exported,'opd')['input']['prompt']==before['prompt']
    assert model_view(exported,'opd')['target'] is None


@pytest.mark.parametrize('failure',['review_isolate','review_incomplete','overlap','missing_overlap_check','source_changed','source_test_failure','oracle_failure','mutant_survives'])
def test_no_flag_or_hash_only_admission(failure):
    row,static,review,source,execution=copy.deepcopy(evidence())
    if failure=='review_isolate':review['verdict']='isolate'
    elif failure=='review_incomplete':review['valid_schema']=False
    elif failure=='overlap':static['heldout_5gram_numeric_containment_at_0_5']=[{'target_id':'protected'}]
    elif failure=='missing_overlap_check':static.pop('legacy_protected_check')
    elif failure=='source_changed':source['ast_sha256']='changed'
    elif failure=='source_test_failure':execution['results'][0]['status']='execution_error'
    elif failure=='oracle_failure':execution['results'][1]['checked']=47
    else:execution['results'][2]['status']='passed'
    with pytest.raises(ValueError):admit(row,static,review,source,execution)


def test_replacement_keeps_2499_approved_rows_and_interface_balance():
    old=load_approved('data/portable_release/manifest.json','project2_code_train.jsonl')
    replacement=admit(*evidence())
    result=replace_one(old,replacement)
    assert len(result)==2500
    assert Counter(r['interface'] for r in result)=={'function':1250,'stdin_stdout':1250}
    assert REMOVED_ID not in {r['sample_id'] for r in result}
    assert sum(r['sample_id']==REPLACEMENT_ID for r in result)==1
    for a,b in zip(old,result):
        if a['sample_id']!=REMOVED_ID:assert a==b


def test_jsonl_unicode_separator_is_not_a_new_record(tmp_path):
    p=tmp_path/'records.jsonl'
    p.write_text(json.dumps({'problem':'a\u2028b'},ensure_ascii=False)+'\n',encoding='utf-8')
    assert rows(p)==[{'problem':'a\u2028b'}]


def test_rectangle_oracle_has_576_independently_enumerated_subcases():
    cases=oracle_cases()
    assert len(cases)==48
    assert all(len(c['output'].split())==12 for c in cases)
    assert cases[0]['output'].split()[:4]==['3','25','3','2000']


def test_new_release_byte_preservation_split_protection_and_config_wiring():
    release=Path('data/code_training_repair_20261025_v4/release')
    manifest=read(release/'manifest.json')
    assert all(sha(release/name)==digest for name,digest in manifest['files'].items())
    code=load_approved(release/'manifest.json','project2_code_train.jsonl')
    assert len(code)==2500 and len({r['sample_id'] for r in code})==2500
    protected=set(read(release/'reserved_dev_ids_never_train.json')['sample_ids'])
    assert not {r['sample_id'] for r in code}&protected
    old=Path('data/portable_release')
    for name in manifest['files']:
        if name!='project2_code_train.jsonl':assert (release/name).read_bytes()==(old/name).read_bytes()
    original=(old/'project2_code_train.jsonl').read_bytes().splitlines(keepends=True)
    changed=(release/'project2_code_train.jsonl').read_bytes().splitlines(keepends=True)
    deltas=[i for i,(a,b) in enumerate(zip(original,changed)) if a!=b]
    assert deltas==[manifest['replacement']['slot_index_zero_based']]
    assert json.loads(original[deltas[0]])['sample_id']==REMOVED_ID
    assert json.loads(changed[deltas[0]])['sample_id']==REPLACEMENT_ID
    c=read('configs/project2_opd_v4_smoke.json')
    assert c['data_manifest']==(release/'manifest.json').as_posix()
    assert 'training_exclusions' not in c
    previous=read('reports/handover_20261024_v3/project2_opd_v3_smoke.json')
    for key in ['student','teachers','tokenizer_audit','tokenizer_path','domain_order','loss_normalization']:
        assert c[key]==previous[key]
