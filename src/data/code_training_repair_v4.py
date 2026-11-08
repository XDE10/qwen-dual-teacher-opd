"""Evidence-gated new release; never promote a row by editing an old approval."""
import ast
from collections import Counter
import hashlib
import json

PRIVATE_FIELDS={'reference_code','test_cases','verification_info','solution','execution'}
REMOVED_ID='code/vfc_121257'
REPLACEMENT_ID='code/vfc_96076'


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def admit(row,static,review,source,execution):
    if row.get('sample_id')!=REPLACEMENT_ID or row.get('training_approved') is not False:
        raise ValueError('Expected the frozen, not-yet-approved replacement candidate')
    if row.get('split')!='train' or row.get('interface')!='stdin_stdout' or row.get('domain')!='code':
        raise ValueError('Candidate interface/domain/split differs')
    if static.get('candidate_row_sha256')!=canonical_sha(row) or static.get('sample_id')!=row['sample_id']:
        raise ValueError('Candidate identity changed')
    if not (review.get('verdict')=='pass' and review.get('valid_schema') is True
            and review.get('issues')==[] and review.get('requires_visual') is False
            and review.get('answer_leak') is False and review.get('request_id')==static.get('historical_request_id')
            and review.get('id')=='code/train/'+row['sample_id']):
        raise ValueError('No complete historical pass evidence')
    if static.get('historical_export_reason')!='train_interface_quota_surplus':
        raise ValueError('Candidate was not a quota-only surplus')
    required={'historical_review_pass':True,'heldout_5gram_numeric_containment_at_0_5':[],
              'legacy_protected_check':None,'current_training_3gram_at_0_85':None}
    if any(k not in static or static[k]!=v for k,v in required.items()):
        raise ValueError('Overlap candidate cannot be admitted')
    identities=static.get('current_training_same_identity_fields',{})
    if set(identities)!={'family_id','problem_hash','reference_ast_sha256'} or any(identities.values()):
        raise ValueError('Duplicate training identity')
    if static.get('distinct_source_test_inputs',0)<3 or static.get('source_test_count')!=len(row['verification_info']['test_cases']):
        raise ValueError('Source test schema evidence incomplete')
    if not static.get('student_tokenizer_files_match_audit') or static['prompt_tokens']+1536>4096:
        raise ValueError('Prompt identity or generation budget failed')
    expected_ast=hashlib.sha256(ast.dump(ast.parse(row['reference_code']),include_attributes=False).encode()).hexdigest()
    if not (source.get('reference_and_tests_byte_content_match') is True
            and source.get('problem_id')==row['sample_id'].removeprefix('code/')
            and source.get('revision')==row['source_revision']
            and source.get('ast_sha256')==row['reference_ast_sha256']==expected_ast):
        raise ValueError('Source verification failed')
    results={r['sample_id']:r for r in execution['results']}
    if results.get('source_all',{}).get('status')!='passed' or results['source_all']['checked']!=len(row['verification_info']['test_cases']):
        raise ValueError('Source tests not all passed')
    if results.get('independent_oracle_48',{}).get('status')!='passed' or results['independent_oracle_48']['checked']!=48:
        raise ValueError('Independent oracle failed')
    if any(results.get(k,{}).get('status')!='mismatch' for k in ['mutant_constant_first_output','mutant_empty_output','mutant_wrong_side_pair']):
        raise ValueError('Wrong implementations were not rejected')
    exported={k:v for k,v in row.items() if k not in PRIVATE_FIELDS}
    exported.update(training_approved=True,split='train',
        approval_request_id=review['request_id'],
        approval_method='historical_delegated_AI_pass_plus_v4_source_sandbox_oracle_and_decontamination_admission',
        human_review_performed=False,
        local_admission_version='code-training-repair-v4-20261025',
        source_id=source['problem_id'],
        verified_source_problem_id=source['problem_id'],
        execution_status='all_source_tests_and_independent_oracle_passed_in_restricted_container',
        prompt_token_count=static['prompt_tokens'])
    return exported


def replace_one(current,replacement):
    if len(current)!=2500 or sum(r['sample_id']==REMOVED_ID for r in current)!=1:
        raise ValueError('Expected exactly one removed duplicate in the original 2500')
    if replacement['sample_id'] in {r['sample_id'] for r in current}:
        raise ValueError('Replacement already present')
    result=[replacement if r['sample_id']==REMOVED_ID else r for r in current]
    if Counter(r['interface'] for r in result)!={'function':1250,'stdin_stdout':1250}:
        raise ValueError('Code interface quota changed')
    if len({r['sample_id'] for r in result})!=2500 or not all(r.get('training_approved') is True for r in result):
        raise ValueError('Invalid training approval or IDs')
    return result
