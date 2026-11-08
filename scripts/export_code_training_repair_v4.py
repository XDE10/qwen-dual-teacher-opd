"""Publish one new dataset version after evidence gates; all prior releases stay read-only."""
from collections import Counter
import json
from pathlib import Path
import shutil

from scripts.prepare_code_training_repair_v4 import OUT, rows
from src.data.code_training_repair_v4 import admit, replace_one, canonical_sha, REMOVED_ID
from src.data.approved_training_inputs import load_approved,model_view
from src.training.runtime_v2 import read,write,sha

RELEASE=Path('data/code_training_repair_20261025_v4/release')


def main():
    if RELEASE.exists():raise FileExistsError('New release already exists; never overwrite it')
    static=read(OUT/'admission_static.json');candidate=read(OUT/'candidate_private.json')
    review=read(OUT/'historical_review.json');source=read(OUT/'source_verification.json');execution=read(OUT/'execution.json')
    for p,digest in static['source_hashes'].items():
        assert sha(Path(p.replace('\\','/')))==digest,p
    assert execution['candidate_private_sha256']==sha(OUT/'candidate_private.json')
    assert execution['worker_sha256']==sha('scripts/code_eval_worker_v3.py')
    assert sha(source['source_file'])==source['source_file_sha256']
    assert all(read_path.read_text().strip()=='0' for read_path in [OUT/'preflight_final.exit.txt',OUT/'execution.exit.txt',OUT/'source_verification.exit.txt'])
    replacement=admit(candidate,static,review,source,execution)
    old_manifest=Path('data/portable_release/manifest.json')
    old=read(old_manifest)
    current=load_approved(old_manifest,'project2_code_train.jsonl')
    repaired=replace_one(current,replacement)
    protected=set(read(old_manifest.parent/old['protected_dev_ids_file'])['sample_ids'])
    assert not {r['sample_id'] for r in repaired}&protected
    approval={'version':'code-training-repair-v4-20261025','authorization':'Current user explicitly requested one replacement to restore code train2500; no budget/API authorization inferred',
        'decision':'admit_one_training_prompt','sample_id':candidate['sample_id'],
        'replaces_confirmed_duplicate':REMOVED_ID,'historical_AI_request_id':review['request_id'],
        'historical_candidate_flag_unchanged':candidate['training_approved'] is False,
        'new_exported_row_sha256':canonical_sha(replacement),
        'evidence_files':{p.as_posix():sha(p) for p in [OUT/'admission_static.json',OUT/'historical_review.json',
            OUT/'historical_request.json',OUT/'source_verification.json',OUT/'execution.json',OUT/'independent_oracle_cases.json']},
        'local_semantic_review':{'reviewer':'Codex AI; not human gold',
            'specification':'t testcases, four positive integer segment lengths; unique maximum rectangle area, stdin/stdout complete.',
            'proof':'Opposite sides constrain one dimension by the pair minimum; enumerate all pair assignments. Sorted a0*a2 is optimal and matches independent 24-permutation oracle.',
            'image_dependency':'Optional illustrated examples are not required; lengths, orientation and output are completely specified in text.',
            'private_input_boundary':'Reference program, source tests and review content remain in reports only.'},
        'checks':{'source_tests_passed':execution['source_count'],'independent_test_groups':48,
                  'independent_subcases':48*12,'wrong_programs_rejected':3,'heldout_lexical_hits_at_0_5':0},
        'limitations':['No human gold claim','Embedding near-neighbor check 未验证','Finite cases are not exhaustive proof','No GPU/model training']}
    write(OUT/'admission_decision.json',approval)
    RELEASE.mkdir(parents=True,exist_ok=False)
    for name in old['files']:
        if name!='project2_code_train.jsonl':shutil.copyfile(old_manifest.parent/name,RELEASE/name)
    # Keep 2499 original JSONL records byte-for-byte; replace only the removed slot.
    old_lines=(old_manifest.parent/'project2_code_train.jsonl').read_bytes().splitlines(keepends=True)
    slot=next(i for i,line in enumerate(old_lines) if json.loads(line)['sample_id']==REMOVED_ID)
    old_lines[slot]=(json.dumps(replacement,ensure_ascii=False)+'\n').encode('utf-8')
    (RELEASE/'project2_code_train.jsonl').write_bytes(b''.join(old_lines))
    manifest={'version':'project2-code-training-repair-v4-20261025','project':2,
        'summary':{'project2_math_train':2500,'project2_math_dev':256,'project2_code_train':{'function':1250,'stdin_stdout':1250},'project2_code_dev':{'function':100,'stdin_stdout':100},
                   'replaced_rows':1,'models_trained':0,'approval':'Historical delegated AI pass plus current evidence-gated local admission; not human gold'},
        'approved_training_files':old['approved_training_files'],
        'approved_evaluation_files':old['approved_evaluation_files'],
        'protected_dev_ids_file':old['protected_dev_ids_file'],
        'files':{p.name:sha(p) for p in RELEASE.iterdir() if p.is_file()},
        'inputs':{'data/portable_release/manifest.json':sha(old_manifest),
                  'reports/code_training_repair_20261025_v4/admission_decision.json':sha(OUT/'admission_decision.json'),
                  'reports/handover_20261024_v3/training_exclusions.json':sha('reports/handover_20261024_v3/training_exclusions.json')},
        'replacement':{'removed':REMOVED_ID,'added':replacement['sample_id'],'slot_index_zero_based':slot,
            'old_row_survives_in_original_release':True,'other_code_rows_byte_identical':2499,
            'evaluation_and_math_files_byte_identical':True},
        'approval_scope':'Only the listed files/rows. Surplus candidates remain unapproved. No old release, flags or SHA modified.'}
    write(RELEASE/'manifest.json',manifest)
    for domain in ['math','code']:
        loaded=load_approved(RELEASE/'manifest.json',f'project2_{domain}_train.jsonl')
        assert len(loaded)==2500
        for r in loaded:model_view(r,'opd')
    for name in ['smoke','math_short','code_short']:
        c=read(f'reports/handover_20261024_v3/project2_opd_v3_{name}.json')
        c['data_manifest']=(RELEASE/'manifest.json').as_posix()
        c.pop('training_exclusions',None) # Duplicate physically absent in this new version.
        c['output_dir']=c['output_dir'].replace('opd_review_v3_','opd_data_v4_')
        write(OUT/f'project2_opd_v4_{name}.json',c)
        write(Path('configs')/f'project2_opd_v4_{name}.json',c)
    write(OUT/'release_summary.json',{'release':RELEASE.as_posix(),'manifest_sha256':sha(RELEASE/'manifest.json'),
        'code_train':len(repaired),'interfaces':dict(Counter(r['interface'] for r in repaired)),
        'removed':REMOVED_ID,'added':replacement['sample_id'],'replaced_slot':slot,
        'dev_unchanged':True,'old_release_unchanged':True,'training_started':False})
    print(json.dumps(read(OUT/'release_summary.json'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
