"""Continuation identities, approved-pool order and terminal archival receipts.

The release fixture is synthetic and does not mutate project approval records.
These checks require neither model downloads, private run artifacts nor a GPU.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import archive_latent_three_checkpoint as archival
from scripts import continue_latent_mopd_three as extension


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def synthetic_release(root):
    pools = {d:[{'sample_id':f'{d}/synthetic/{i}','domain':d,'training_approved':True}
                for i in range(1056)] for d in extension.DOMAINS}
    for domain,rows in pools.items():
        write_jsonl(root/(domain+'_train.jsonl'),rows)
    schedule = []
    for step in range(1,63):
        domain = extension.DOMAINS[(step-1)%3]
        offset = ((step-1)//3)*32
        schedule.append({'step':step,'domain':domain,
                         'sample_ids':[r['sample_id'] for r in pools[domain][offset:offset+32]]})
    write_jsonl(root/'schedule.jsonl',schedule)
    (root/'protected_ids.json').write_text(json.dumps({'sample_ids':['protected-dev']}))
    files = {p.name:{'sha256':extension.sha(p)} for p in root.iterdir()}
    manifest = root/'manifest.json'
    manifest.write_text(json.dumps({'approval':{'approver':'synthetic-test-fixture'},'files':files,
                                    'protected_dev_ids_file':'protected_ids.json'}))
    return manifest


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.parent = {'config':{'steps':62,'output_dir':'/parent','seed':42,'responses':4,
                                 'max_new_tokens':16384,'rollout_batch':4,
                                 'teachers':{'math':'fixed-math','code':'fixed-code','logic':'fixed-logic'},
                                 'data_manifest':'frozen/manifest.json'},
                       'source_files':{'trainer.py':'fixed-trainer'},'generation':{'do_sample':True},
                       'pinned_packages':{'torch':'2.8.0+cu128'},'manifest_sha':'frozen',
                       'audit_sha':'audit','tokenizer_files':{'tokenizer.json':'fixed-tokenizer'}}
        self.output = Path('/content/runs/extension80/latent_mopd_three_latent')
        self.child = copy.deepcopy(self.parent)
        self.child['config'].update(steps=80,output_dir=str(self.output))
        self.lineage = {'parent_step':62,'target_step':80,'entrypoint_sha256':'entry'}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.manifest = synthetic_release(Path(temporary.name))
        for name,value in [('RELEASE_SHA',extension.sha(self.manifest)),
                           ('PARENT_SCHEDULE_SHA',extension.sha(self.manifest.with_name('schedule.jsonl')))]:
            patched = patch.object(extension,name,value)
            patched.start()
            self.addCleanup(patched.stop)

    def test_explicit_budget_and_output_migration_leaves_parent_unchanged(self):
        snapshot = copy.deepcopy(self.parent)
        result = extension.derive_hard(self.parent,self.child,self.lineage,self.output)
        self.assertEqual(result['continuation'],self.lineage)
        self.assertEqual(self.parent,snapshot)
        self.assertEqual(result['source_files'],self.parent['source_files'])
        self.assertEqual(result['generation'],self.parent['generation'])
        result['config']['seed'] = 7
        self.assertEqual(self.child['config']['seed'],42)

    def test_recipe_identity_changes_are_rejected(self):
        for field in ('seed','responses','max_new_tokens','rollout_batch','teachers','data_manifest'):
            with self.subTest(field=field):
                child = copy.deepcopy(self.child)
                child['config'][field] = 'changed'
                with self.assertRaises(ValueError):
                    extension.derive_hard(self.parent,child,self.lineage,self.output)
        for field in ('source_files','generation','pinned_packages','manifest_sha','audit_sha','tokenizer_files'):
            with self.subTest(field=field):
                child = copy.deepcopy(self.child)
                child[field] = 'changed'
                with self.assertRaises(ValueError):
                    extension.derive_hard(self.parent,child,self.lineage,self.output)

    def test_80_step_plan_preserves_62_and_uses_new_approved_rows(self):
        plan = extension.make_plan(self.manifest)
        self.assertEqual(plan[:62],extension.read_rows(self.manifest.with_name('schedule.jsonl')))
        self.assertEqual(len(plan),80)
        before = {sample for step in plan[:62] for sample in step['sample_ids']}
        after = [sample for step in plan[62:] for sample in step['sample_ids']]
        self.assertEqual(len(before),1984)
        self.assertEqual(len(after),576)
        self.assertEqual(len(set(after)),576)
        self.assertFalse(before.intersection(after))
        self.assertEqual([row['domain'] for row in plan[62:65]],['logic','math','code'])
        for domain in extension.DOMAINS:
            self.assertEqual(sum(row['domain']==domain for row in plan[62:]),6)
        self.assertEqual(plan[70]['domain'],plan[61]['domain'])
        self.assertEqual(plan[79]['domain'],plan[61]['domain'])

    def test_next_stage_80_to98_keeps_recipe_identity(self):
        parent80 = copy.deepcopy(self.child)
        parent80['continuation'] = {'schema':'explicit-chain-continuation-1','parent_step':62,'target_step':80}
        child98 = copy.deepcopy(parent80)
        child98.pop('continuation')
        child98['config']['steps'] = 98
        out = Path('/content/runs/extension98/latent_mopd_three_latent')
        child98['config']['output_dir'] = str(out)
        lineage = {'schema':'explicit-chain-continuation-1','parent_step':80,'target_step':98}
        result = extension.derive_hard(parent80,child98,lineage,out,98)
        self.assertEqual(result['continuation']['parent_step'],80)
        self.assertEqual(result['generation'],self.parent['generation'])
        self.assertEqual(result['source_files'],self.parent['source_files'])
        self.assertEqual(parent80['config']['steps'],80)

    def test_next_schedule_extends80_without_repeating_its_rows(self):
        plan80 = extension.make_plan(self.manifest,80)
        plan98 = extension.make_plan(self.manifest,98)
        self.assertEqual(plan98[:80],plan80)
        before = {sample for row in plan80 for sample in row['sample_ids']}
        extra = [sample for row in plan98[80:] for sample in row['sample_ids']]
        self.assertEqual(len(extra),576)
        self.assertFalse(before.intersection(extra))
        self.assertEqual(plan98[80]['domain'],'logic')


class Step:
    def __init__(self,value): self.value=value
    def item(self): return self.value


def optimizer_state():
    return {'step':80,'optimizer':{'state':{i:{'step':Step(80),'exp_avg':'saved','exp_avg_sq':'saved'}
                                          for i in range(339)},
                                  'param_groups':[{'lr':1e-5,'fused':True,'betas':(0.9,0.999),
                                                   'weight_decay':0.01,'eps':1e-8}]},
            'python_rng':('saved',),'numpy_rng':('saved',),'torch_rng':'saved','cuda_rng':['saved']}


class ArchiveTests(unittest.TestCase):
    def test_terminal80_retains_all_training_state(self):
        self.assertEqual(archival.validate_state(optimizer_state(),80),339)

    def test_missing_training_state_and_reset_optimizer_are_rejected(self):
        for key in ('optimizer','step','python_rng','numpy_rng','torch_rng','cuda_rng'):
            with self.subTest(key=key):
                saved=optimizer_state();del saved[key]
                with self.assertRaises(ValueError): archival.validate_state(saved,80)
        saved=optimizer_state()
        for value in saved['optimizer']['state'].values():value['step']=Step(0)
        with self.assertRaises(ValueError): archival.validate_state(saved,80)
        saved=optimizer_state();saved['cuda_rng']=[]
        with self.assertRaises(ValueError): archival.validate_state(saved,80)
        saved=optimizer_state();del saved['optimizer']['state'][0]['exp_avg']
        with self.assertRaises(ValueError): archival.validate_state(saved,80)

    def test_upload_failure_cannot_leave_a_verified_local_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'checkpoint-80_verified.json'
            with patch.object(archival,'command',side_effect=RuntimeError('transfer failed')):
                with self.assertRaises(RuntimeError): archival.commit_receipt({'step':80},target,'remote')
            self.assertFalse(target.exists())

    def test_mismatched_remote_receipt_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'checkpoint-80_verified.json'
            with patch.object(archival,'command',side_effect=['','{"step":79}']):
                with self.assertRaises(RuntimeError): archival.commit_receipt({'step':80},target,'remote')
            self.assertFalse(target.exists())

    def test_marker_is_committed_after_remote_readback(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'checkpoint-80_verified.json'
            with patch.object(archival,'command',side_effect=['','{"step":80}']) as command:
                archival.commit_receipt({'step':80},target,'remote')
            self.assertEqual(json.loads(target.read_text()),{'step':80})
            self.assertEqual(command.call_args_list[-1].args,('rclone','cat','remote'))
