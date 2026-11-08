import copy
import hashlib
import json
from pathlib import Path

import pytest

from src.evaluation.comparators_v3 import compare, constructive
from src.evaluation.reviewed_registry_v3 import validate_registry
from src.evaluation.code_v2 import evaluate, make_job
from src.data.training_exclusions_v3 import filter_training_pools
from src.data.approved_training_inputs import load_approved
from src.training.runtime_v2 import sha

DIR = Path('reports/handover_20261024_v3')


def actual_rows():
    return [json.loads(l) for l in Path('data/portable_release/project2_code_dev.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]


def test_registry_closed_identity_and_blocked_decisions():
    rows = actual_rows()
    registry = json.loads((DIR/'reviewed_comparators.json').read_text(encoding='utf-8'))
    validate_registry(rows, registry, sha('data/portable_release/project2_code_dev.jsonl'))
    assert len(registry)==200
    assert sum(s['status']=='blocked_after_review' for s in registry.values())==8
    altered = copy.deepcopy(rows)
    altered[0]['problem'] += ' changed'
    with pytest.raises(ValueError,match='identity'):
        validate_registry(altered,registry)
    deleted = dict(registry)
    deleted.pop(rows[0]['sample_id'])
    with pytest.raises(ValueError,match='task set'):
        validate_registry(rows,deleted)


def test_finite_tolerance_does_not_relax_integer_count_fields():
    rule={'type':'tree','fields':{'average':{'type':'float','rtol':1e-9,'atol':1e-12}}}
    assert compare({'count':3,'average':1/3+1e-12},{'count':3,'average':1/3},rule)
    assert not compare({'count':3.000001,'average':1/3},{'count':3,'average':1/3},rule)
    assert not compare({'count':True,'average':1/3},{'count':1,'average':1/3},rule)
    assert not compare(float('nan'),1,{'type':'float','rtol':1e-9,'atol':1e-12})
    assert not compare('1.0',1.,{'type':'float','rtol':1e-9,'atol':1e-12})


@pytest.mark.parametrize('checker,stdin,good,bad',[
 ('cycle_walk','3 3\nS**\n*.*\n***\n','DDRRUULL','RRDDLLUUU'),
 ('zero_sum_subset','1\n3\n0 -1 1\n','2\n2 3','2\n2 2'),
 ('equal_candy_groups','2\n','4 1\n2 3','1 2\n3 4'),
 ('equal_leg_pairs','2\n1 2 3 4\n','4 1\n2 3','1 3\n2 4'),
])
def test_dedicated_checker_accepts_alternative_solution_and_rejects_invalid(checker,stdin,good,bad):
    assert constructive(good,stdin,checker)
    assert not constructive(bad,stdin,checker)
    assert not constructive(good+' 999',stdin,checker)


def test_zero_subset_multiple_cases_and_limits():
    data='2\n3\n0 -1 1\n1\n0\n'
    assert constructive('1 1\n1 1',data,'zero_sum_subset')
    assert not constructive('0\n1 1',data,'zero_sum_subset')
    assert not constructive('1 4\n1 1',data,'zero_sum_subset')


def test_tied_maximum_geometry_does_not_force_reference_first_choice():
    test={'args':[[.5,.5],[[1,2],[3,4]]]}
    assert compare([3,4],[1,2],{'type':'maximum_geometry'},test)
    assert not compare([5,6],[1,2],{'type':'maximum_geometry'},test)


def test_permitted_case_and_layout_are_not_globally_relaxed():
    assert compare('yEs\nNO','YES\nno',{'type':'stdout_tokens_casefold'})
    assert not compare('Yes','YES',{'type':'stdout_tokens_exact'})
    assert not compare('*D* DDD *D*','*D*\nDDD\n*D*',{'type':'stdout_lines_exact'})
    assert compare([{'x':1,'y':2}], [{'y':2,'x':1}],{'type':'unordered'})


def test_blocked_tasks_stay_in_full_denominator_and_never_execute():
    rows = actual_rows()
    registry=json.loads((DIR/'reviewed_comparators.json').read_text(encoding='utf-8'))
    captured=[]
    def fake(jobs):
        captured.extend(jobs)
        return {'results':[{'sample_id':j['sample_id'],'status':'passed'} for j in jobs]}
    result=evaluate(rows,{r['sample_id']:r['reference_code'] for r in rows},registry,fake)
    assert len(captured)==192
    assert result['coverage']=={'total_tasks':200,'blocked_tasks':8,'passed_tasks':192}
    assert sum(r['status']=='blocked_after_review' for r in result['results'])==8


def test_confirmed_duplicate_subtracted_without_reordering_or_reapproval():
    manifest='data/portable_release/manifest.json'
    pools={d:load_approved(manifest,f'project2_{d}_train.jsonl') for d in ['math','code']}
    before=copy.deepcopy(pools)
    result=filter_training_pools(pools,manifest,DIR/'training_exclusions.json')
    assert pools==before
    assert result['math']==pools['math']
    assert [r['sample_id'] for r in result['code']]==[r['sample_id'] for r in pools['code'] if r['sample_id']!='code/vfc_121257']
    assert len(result['code'])==2499
    assert sum(r['interface']=='function' for r in result['code'])==1250
    assert sum(r['interface']=='stdin_stdout' for r in result['code'])==1249
    assert all(r['training_approved'] is True for r in result['code'])


def test_exclusion_policy_rejects_unknown_id_and_changed_manifest(tmp_path):
    pools={'code':[{'sample_id':'approved'}]}
    data=tmp_path/'train.jsonl'
    data.write_text('{"sample_id":"approved"}\n')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'approved_training_files':['train.jsonl']}))
    policy=tmp_path/'policy.json'
    policy.write_text(json.dumps({'schema':'training-exclusions-v3','manifest_sha256':sha(manifest),'excluded_sample_ids':['unknown']}))
    with pytest.raises(ValueError,match='Unknown'):
        filter_training_pools(pools,manifest,policy)
    manifest.write_text('{}')
    with pytest.raises(ValueError,match='identity'):
        filter_training_pools(pools,manifest,policy)


def test_run_identity_tracks_policy_bytes_and_filter_source(tmp_path):
    from src.training.runtime_v2 import build_identity
    config=json.loads((DIR/'project2_opd_v3_smoke.json').read_text(encoding='utf-8'))
    policy=tmp_path/'policy.json'
    policy.write_text('{"version":1}')
    config['training_exclusions']=str(policy)
    first=build_identity(config,{},Path.cwd())
    policy.write_text('{"version":2}')
    second=build_identity(config,{},Path.cwd())
    assert first['training_exclusions_sha'] != second['training_exclusions_sha']
    assert first['training_exclusion_source_sha']==sha('src/data/training_exclusions_v3.py')
