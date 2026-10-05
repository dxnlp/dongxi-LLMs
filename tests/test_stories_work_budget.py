"""Original decoder equations and persistent story-work CPU acceptance."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
import torch
import test_stories_valid_target_budget as original
from dongxi_llms import stories_training as story
from dongxi_llms import story_work_budget as accounting
from dongxi_llms.decoder_lab import DecoderConfig
from dongxi_llms.run_identity import canonical_hash
from dongxi_llms.story_work_budget import (STORY_WORK_KEYS, StoryWorkBudget,
    story_work_contract, read_story_receipt)
from dongxi_llms.work_budget import WorkBudgetExceeded, WorkLedger

ROOT=original.ROOT
ARCHIVE='experiments/reports/2026-10-05-story-work-preintegration'
SPEC='experiments/specs/2026-10-05-story-persistent-work-cpu.md'
CAPS={key:100000 for key in STORY_WORK_KEYS}
BOUND=1024**2
NUMERICAL_KEYS=('update','loss','lr','gradient_norm','valid_targets','cumulative_targets',
    'first_q_max_parameter_change','processed_positions','cumulative_processed_positions',
    'processed_positions_accounting')
SOURCES=('src/dongxi_llms/stories_training.py','src/dongxi_llms/story_work_budget.py',
    'src/dongxi_llms/stories_data.py','src/dongxi_llms/decoder_lab.py',
    'src/dongxi_llms/pretraining_lab.py','src/dongxi_llms/work_budget.py',
    'src/dongxi_llms/run_identity.py','src/dongxi_llms/snapshot_io_budget.py',
    'src/dongxi_llms/stories_work_lab.py','src/dongxi_llms/batched_cache_lab.py',
    'src/dongxi_llms/dpo_lab.py','src/dongxi_llms/grpo_lab.py',
    'scripts/train_stories.py','tests/test_stories_work_budget.py',
    'tests/test_stories_pipeline.py','tests/test_stories_valid_target_budget.py',
    'tests/test_work_budget.py','pyproject.toml','uv.lock',SPEC,
    'tests/test_stories_work_lab.py','experiments/specs/2026-10-05-story-work-reader-lesson.md',
    'experiments/specs/2026-10-05-story-persistent-work.md',ARCHIVE+'/manifest.json',
    *(ARCHIVE+'/'+p+'.txt' for p in ('src/dongxi_llms/stories_training.py',
        'scripts/train_stories.py','tests/test_stories_pipeline.py',
        'tests/test_stories_valid_target_budget.py','src/dongxi_llms/stories_data.py',
        'src/dongxi_llms/decoder_lab.py','src/dongxi_llms/pretraining_lab.py')))


def write(path,value):
    with Path(path).open('x',encoding='utf8') as handle:
        handle.write(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())


def original_actor():
    """Execute trusted archived course bytes, map only source-identity paths."""
    manifest=json.loads((ROOT/ARCHIVE/'manifest.json').read_text())
    name='dongxi_stories_original_reference';module=ModuleType(name)
    module.__file__=str(ROOT/'src/dongxi_llms/stories_training.py');sys.modules[name]=module
    archived=ROOT/ARCHIVE/'src/dongxi_llms/stories_training.py.txt'
    exec(compile(archived.read_text(),str(archived),'exec'),module.__dict__)
    def identity(path):
        key='src/dongxi_llms/'+Path(path).name;row=manifest['source_files'][key]
        actual=(ROOT/ARCHIVE/row['archive']).read_bytes()
        if hashlib.sha256(actual).hexdigest()!=row['sha256']:raise AssertionError('Original source archive changed')
        return row['sha256']
    module.digest=identity
    return module


def configuration(native_sampling=False):
    return replace(original.config(),vocab=50257) if native_sampling else original.config()


def recipe(seed=909,checkpointing=False):
    return replace(original.recipe(),seed=seed,activation_checkpointing=checkpointing,accumulation=2)


def plain(seed=909,checkpointing=False,*,archived=False,native_sampling=False,cap=1000):
    module=original_actor() if archived else story
    rec=recipe(seed,checkpointing)
    if archived:rec=module.Recipe(**asdict(rec))
    return module.Session(configuration(native_sampling),rec,original.AuthoredWindows(),
                          valid_target_budget=cap),module


def fixture(directory,seed=909,checkpointing=False,*,caps=CAPS,cap=1000,native_sampling=False,
            receipt=None,journal=None,invocation='authored-story-work',recipe_overrides=None):
    directory=Path(directory);directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    cfg=configuration(native_sampling);rec=recipe(seed,checkpointing);data=original.AuthoredWindows()
    if recipe_overrides:rec=replace(rec,**recipe_overrides)
    contract=story_work_contract(caps,BOUND)
    science=story.session_contract(cfg,rec,data,valid_target_budget=cap,work_contract=contract)
    journal=directory/'work.jsonl' if journal is None else Path(journal)
    if receipt is None:
        budget=StoryWorkBudget.create(journal,contract=contract,scientific_contract=science,invocation_id=invocation)
    else:
        budget=StoryWorkBudget.open(journal,contract=contract,scientific_contract=science,
                                   invocation_id=invocation,receipt=receipt)
    try:
        session=story.Session(cfg,rec,data,valid_target_budget=cap,work_budget=budget)
    except BaseException:
        budget.ledger.close();raise
    return session,science,budget


def record(row):return {key:row[key] for key in NUMERICAL_KEYS}


def numerical(session):return original.snapshot(session)


def state_sha(session):return original.state_digest(numerical(session))


def next_ids(session):
    old=session.stream.state_dict()
    try:return session.stream.take(session.recipe.microbatch*session.recipe.accumulation)
    finally:session.stream.load_state_dict(old)


def save(session,path):
    session.save(path)
    return read_story_receipt(Path(str(path)+'.work.json'))


def changed_payload(path,receipt,output,change):
    state=torch.load(path,map_location='cpu',weights_only=True);change(state)
    with Path(output).open('xb') as handle:torch.save(state,handle)
    retained=deepcopy(receipt);raw=Path(output).read_bytes()
    retained['payload_sha256']=hashlib.sha256(raw).hexdigest();retained['payload_bytes']=len(raw)
    return retained


def command_module():
    spec=importlib.util.spec_from_file_location('dongxi_story_work_native_command',ROOT/'scripts/train_stories.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class NoMonitor:
    """Explicit CPU-fixture exclusion; no watcher thread, signal or GPU job."""
    def __init__(self,*args,**kwargs):pass
    def __enter__(self):return self
    def __exit__(self,*args):pass


class AuthoredTokenizer:
    """Original EOS/vocabulary, explicit authored prefix IDs, no downloaded BPE."""
    def encode(self,prompt,add_special_tokens=False):
        return SimpleNamespace(ids=[1]*(story.PROMPTS.index(prompt)+1))
    def decode(self,ids):return ' '.join(map(str,ids))


def later_failure(session):
    before=session.work_budget.ledger.snapshot();actual=session.summed_loss;calls=0
    def fail_second(x,y):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('authored second accumulation forward failure')
        return actual(x,y)
    try:
        with patch.object(session,'summed_loss',side_effect=fail_second):session.update()
    except OSError as error:failure=dict(type=type(error).__name__,message=str(error))
    else:raise AssertionError('Predeclared admitted partial update failure absent')
    after=session.work_budget.ledger.snapshot()
    if calls!=2 or not after['failed_tickets']:raise AssertionError('Actual admitted failure was not retained')
    return dict(before=before,after=after,actual_loss_calls=calls,error=failure,
        completed_updates=session.step,successful_targets=session.tokens,poisoned=session.poisoned)


def fresh_child(bundle_path,output):
    bundle=json.loads(Path(bundle_path).read_text());directory=Path(output);directory.mkdir(mode=0o700)
    retained=read_story_receipt(bundle['checkpoint']+'.work.json')
    session,science,budget=fixture(directory/'constructor',bundle['seed'],bundle['checkpointing'],
        receipt=retained,journal=bundle['journal'],invocation='authored-fresh-story-work')
    try:
        if science!=bundle['science']:raise ValueError('Actual fresh story science changed')
        session.restore(bundle['checkpoint'],work_receipt=retained)
        after_restore=budget.ledger.snapshot();selected=[];history=[]
        while session.step<5:
            selected.append(next_ids(session));history.append(record(session.update()))
        session.save(directory/'final.pt')
        final=numerical(session)
        with (directory/'final-state.pt').open('xb') as handle:torch.save(final,handle)
        result=dict(seed=bundle['seed'],checkpointing=bundle['checkpointing'],restored_completed_updates=2,
            process_id=os.getpid(),parent_process_id=os.getppid(),actual_argv=list(sys.orig_argv),
            completed_updates=session.step,selected_ids=selected,history=history,
            state_sha256=state_sha(session),history_sha256=original.state_digest(history),
            evaluation=session.evaluate(original.AuthoredWindows()),
            activations=story.activation_summary(session,original.AuthoredWindows()),
            after_restore=after_restore,work=budget.ledger.snapshot(),final_checkpoint=str((directory/'final.pt').resolve()))
        write(directory/'result.json',result);return result
    finally:budget.ledger.close()


def teaching_resource_observation(memory_probe=None):
    """One portable2GiB tiny-fixture sample, not Spark production clearance.

    Explicit probes are labelled injected controls, never platform evidence.
    Unknown, malformed and low observations retain a refusal rather than pass.
    """
    source=('actual-psutil-available-point-sample' if memory_probe is None
            else 'injected-unit-probe-not-platform-evidence')
    probe=(lambda:psutil.virtual_memory().available) if memory_probe is None else memory_probe
    result=dict(policy='tiny-cpu-teaching2',source=source,required_available_bytes=2*1024**3,
        observed_available_bytes=None,cleared=False,failure=None,
        measurement_boundary='Single pre-child available-memory point sample; not a continuous minimum or production clearance')
    try:
        available=probe()
        if type(available) is not int or available<0:
            raise ValueError('Available memory must be an exact nonnegative byte integer')
        result['observed_available_bytes']=available
        if available<result['required_available_bytes']:
            raise RuntimeError('Refusing tiny CPU child below2GiB sampled teaching reserve')
    except Exception as error:
        result['failure']=dict(type=type(error).__name__,message=str(error)[:256])
    else:result['cleared']=True
    return result


def replay_arm(directory,seed,checkpointing,*,resource_policy='linux-collector25',memory_probe=None):
    if resource_policy not in ('linux-collector25','tiny-cpu-teaching2'):
        raise ValueError('Explicit known story replay resource policy required')
    if resource_policy=='linux-collector25' and memory_probe is not None:
        raise ValueError('Default collector keeps its actual Linux25GiB probe')
    directory=Path(directory);directory.mkdir(mode=0o700)
    clean,science,budget=fixture(directory/'clean',seed,checkpointing)
    try:
        expected_ids=[];expected=[]
        while clean.step<5:
            expected_ids.append(next_ids(clean));expected.append(record(clean.update()))
        expected_state=state_sha(clean);expected_eval=clean.evaluate(original.AuthoredWindows())
        expected_activations=story.activation_summary(clean,original.AuthoredWindows())
        with (directory/'clean/final-state.pt').open('xb') as handle:torch.save(numerical(clean),handle)
        write(directory/'clean/history.json',expected)
        clean.save(directory/'clean/final.pt');clean_work=budget.ledger.snapshot()
    finally:budget.ledger.close()
    session,current,budget=fixture(directory/'retained',seed,checkpointing)
    if current!=science:raise AssertionError('Science changed before retained arm')
    try:
        before=[]
        for _ in range(2):before.append(record(session.update()))
        checkpoint=directory/'retained/completed2.pt';receipt=save(session,checkpoint)
        failed=later_failure(session)
        write(directory/'retained/failed-suffix.json',failed)
        bundle=dict(seed=seed,checkpointing=checkpointing,science=science,
            checkpoint=str(checkpoint.resolve()),journal=str(budget.ledger.path))
        write(directory/'bundle.json',bundle)
    finally:budget.ledger.close()
    command=[sys.executable,str(Path(__file__).resolve()),'--child',str(directory/'bundle.json'),str(directory/'fresh')]
    extra={}
    if resource_policy=='linux-collector25':
        # Preserve the original collector's guard and receipt keys unchanged.
        available=story.available_gib()
        if available<25:raise RuntimeError('Refusing fresh CPU child below25GiB sampled host reserve')
    else:
        observation=teaching_resource_observation(memory_probe)
        extra['resource_observation']=observation
        if not observation['cleared']:
            write(directory/'fresh-execution.json',dict(command=command,actual_exit_code=None,
                seconds=0.,stdout='',stderr='',deadline_seconds=60,
                sampled_prechild_available_gib=None if observation['observed_available_bytes'] is None
                    else observation['observed_available_bytes']/1024**3,**extra))
            raise RuntimeError(observation['failure']['message'])
        available=observation['observed_available_bytes']/1024**3
    started=time.monotonic();child=subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=60)
    execution=dict(command=command,actual_exit_code=child.returncode,seconds=time.monotonic()-started,
                   stdout=child.stdout,stderr=child.stderr,deadline_seconds=60,
                   sampled_prechild_available_gib=available,**extra)
    write(directory/'fresh-execution.json',execution)
    if child.returncode:raise AssertionError('Fresh story child failed: '+child.stderr)
    answer=json.loads(child.stdout)
    if answer['state_sha256']!=expected_state or answer['history_sha256']!=original.state_digest(expected[2:]):
        raise AssertionError('Actual fresh story replay differs numerically')
    if answer['selected_ids']!=expected_ids[2:] or answer['evaluation']!=expected_eval or answer['activations']!=expected_activations:
        raise AssertionError('Actual next batch/evaluation/features changed after recovery')
    if not set(failed['after']['failed_tickets'])<=set(answer['work']['failed_tickets']):
        raise AssertionError('Old checkpoint refunded later failure')
    result=dict(seed=seed,checkpointing=checkpointing,exact_fresh_replay=True,execution=execution,
        expected_state_sha256=expected_state,expected_history_sha256=original.state_digest(expected[2:]),
        expected_selected_ids=expected_ids[2:],full_successful_history=expected,
        carried_receipt=receipt,failed_suffix=failed,fresh=answer,clean_work=clean_work)
    write(directory/'arm.json',result);return result


class TeachingResourcePolicyTests(unittest.TestCase):
    """Injected branch checks do not establish a Mac or hosted run."""

    def test_exact_two_gib_boundary_and_labelled_injected_sample(self):
        for available in (2*1024**3,2*1024**3+1):
            result=teaching_resource_observation(lambda:available)
            self.assertTrue(result['cleared']);self.assertIsNone(result['failure'])
            self.assertEqual(result['observed_available_bytes'],available)
            self.assertEqual(result['required_available_bytes'],2*1024**3)
            self.assertEqual(result['source'],'injected-unit-probe-not-platform-evidence')

    def test_missing_malformed_and_nonfinite_memory_never_clear(self):
        for value in (None,True,False,-1,2.*1024**3,float('nan'),float('inf'),'available'):
            with self.subTest(value=value):
                result=teaching_resource_observation(lambda:value)
                self.assertFalse(result['cleared']);self.assertIsNone(result['observed_available_bytes'])
                self.assertEqual(result['failure']['type'],'ValueError')

    def test_low_or_failed_point_sample_retains_refusal(self):
        low=teaching_resource_observation(lambda:2*1024**3-1)
        self.assertFalse(low['cleared']);self.assertEqual(low['observed_available_bytes'],2*1024**3-1)
        self.assertIn('below2GiB',low['failure']['message'])
        def absent():raise OSError('authored unavailable point sample')
        missing=teaching_resource_observation(absent)
        self.assertFalse(missing['cleared']);self.assertIsNone(missing['observed_available_bytes'])
        self.assertEqual(missing['failure']['type'],'OSError')
        with tempfile.TemporaryDirectory(prefix='dongxi-teaching-refusal-') as temporary:
            output=Path(temporary)/'refused'
            with patch.object(subprocess,'run') as child,self.assertRaisesRegex(RuntimeError,'below2GiB'):
                replay_arm(output,909,False,resource_policy='tiny-cpu-teaching2',memory_probe=lambda:0)
            child.assert_not_called()
            retained=json.loads((output/'fresh-execution.json').read_text())
            self.assertIsNone(retained['actual_exit_code']);self.assertEqual(retained['deadline_seconds'],60)
            self.assertEqual(retained['resource_observation']['observed_available_bytes'],0)
            self.assertFalse(retained['resource_observation']['cleared'])

    def test_default_portable_provider_one_sample_and_unknown_policy_no_fixture(self):
        with patch.object(psutil,'virtual_memory',return_value=SimpleNamespace(available=3*1024**3)) as probe:
            result=teaching_resource_observation()
        probe.assert_called_once_with();self.assertTrue(result['cleared'])
        self.assertEqual(result['source'],'actual-psutil-available-point-sample')
        with patch(__name__+'.fixture') as model:
            with self.assertRaises(ValueError):replay_arm('unused',909,False,resource_policy='unknown')
            with self.assertRaises(ValueError):replay_arm('unused',909,False,memory_probe=lambda:100)
        model.assert_not_called()


class StoriesWorkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);temporary=tempfile.TemporaryDirectory(prefix='dongxi-stories-work-')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def make(self,**kwargs):
        session,science,budget=fixture(self.root/'owned',**kwargs)
        self.addCleanup(budget.ledger.close);return session,science,budget

    def test_original_archive_bytes_and_explicit21_cap_schema(self):
        manifest=json.loads((ROOT/ARCHIVE/'manifest.json').read_text())
        for row in manifest['source_files'].values():
            p=ROOT/ARCHIVE/row['archive']
            self.assertEqual(p.stat().st_size,row['bytes']);self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),row['sha256'])
        self.assertEqual(len(STORY_WORK_KEYS),21)
        for caps in (dict(CAPS,train_updates=True),dict(CAPS,extra=1),
                     {k:v for k,v in CAPS.items() if k!='save_operations'},dict(CAPS,train_updates=1.5),
                     dict(CAPS,train_updates=float('nan'))):
            with self.assertRaises(ValueError):story_work_contract(caps,BOUND)

    def test_current_accounted_unaccounted_and_archived_equations_allfourarms(self):
        for seed in (909,910):
            for mode in (False,True):
                old,module=plain(seed,mode,archived=True);reference,_=plain(seed,mode)
                value,_,budget=fixture(self.root/f'{seed}-{mode}',seed,mode)
                try:
                    for _ in range(5):
                        a,b,c=record(value.update()),record(reference.update()),record(old.update())
                        self.assertEqual(a,b);self.assertEqual(a,c)
                        self.assertEqual(state_sha(value),state_sha(reference));self.assertEqual(state_sha(value),state_sha(old))
                    data=original.AuthoredWindows()
                    self.assertEqual(value.evaluate(data),reference.evaluate(data));self.assertEqual(value.evaluate(data),old.evaluate(data))
                    self.assertEqual(story.activation_summary(value,data),module.activation_summary(old,data))
                finally:budget.ledger.close()

    def test_fresh_process_original_two_to_five_allfourarms(self):
        self.fresh_replay_results=[]
        for seed in (909,910):
            for mode in (False,True):
                with self.subTest(seed=seed,checkpointing=mode):
                    result=replay_arm(self.root/f'replay-{seed}-{mode}',seed,mode,
                        resource_policy='tiny-cpu-teaching2')
                    self.fresh_replay_results.append(result)
                    self.assertTrue(result['exact_fresh_replay'])
                    self.assertEqual(result['execution']['actual_exit_code'],0)
                    observation=result['execution']['resource_observation']
                    self.assertTrue(observation['cleared'])
                    self.assertEqual(observation['source'],'actual-psutil-available-point-sample')
                    self.assertEqual(observation['required_available_bytes'],2*1024**3)

    def test_original_native_sampling_current_archive_and_accounted(self):
        old,module=plain(archived=True,native_sampling=True);reference,_=plain(native_sampling=True)
        value,_,budget=self.make(native_sampling=True);tok=AuthoredTokenizer()
        actual=story.samples(value,tok,max_new=3)
        self.assertEqual(actual,story.samples(reference,tok,max_new=3));self.assertEqual(actual,module.samples(old,tok,max_new=3))
        costs=budget.ledger.snapshot()
        self.assertEqual(costs['reserved']['generation_sequences'],6)
        self.assertEqual(costs['reserved']['generation_calls'],18)
        self.assertEqual(costs['reserved']['generation_positions'],72)
        self.assertEqual(costs['reserved']['generation_tokens'],18)
        self.assertEqual(costs['reserved']['multinomial_draws'],9)
        self.assertEqual(costs['completed']['generation_tokens'],sum(len(row['token_ids']) for row in actual))

    def test_zero_initialization_before_model_constructor_and_rng_mutation(self):
        caps=dict(CAPS,model_initializations=0);contract=story_work_contract(caps,BOUND)
        science=story.session_contract(configuration(),recipe(),original.AuthoredWindows(),
            valid_target_budget=1000,work_contract=contract)
        owned=self.root/'zero-init';owned.mkdir(mode=0o700)
        budget=StoryWorkBudget.create(owned/'work.jsonl',contract=contract,
            scientific_contract=science,invocation_id='zero-model')
        self.addCleanup(budget.ledger.close);before=torch.get_rng_state().clone()
        with patch.object(story,'StoriesDecoder') as model,patch.object(torch,'manual_seed') as rng:
            with self.assertRaises(WorkBudgetExceeded):story.Session(configuration(),recipe(),
                original.AuthoredWindows(),valid_target_budget=1000,work_budget=budget)
        model.assert_not_called();rng.assert_not_called();self.assertTrue(torch.equal(before,torch.get_rng_state()))

    def test_whole_update_zero_dimensions_before_numerical_mutation(self):
        for key in ('train_updates','training_windows','training_valid_targets','policy_forward_calls',
                    'policy_forward_positions','backward_calls','optimizer_calls'):
            value,_,budget=fixture(self.root/key,caps=dict(CAPS,**{key:0}))
            try:
                value.model.eval();before=state_sha(value)
                with patch.object(value,'summed_loss') as model,patch.object(value.optimizer,'step') as optimizer:
                    with self.assertRaises(WorkBudgetExceeded):value.update()
                model.assert_not_called();optimizer.assert_not_called()
                self.assertEqual(state_sha(value),before);self.assertFalse(value.poisoned)
                self.assertEqual(budget.ledger.snapshot()['reserved']['train_updates'],0)
            finally:budget.ledger.close()

    def test_epoch_crossing_reservation_rollback_preserves_all_stream_rng(self):
        value,_,_=self.make(caps=dict(CAPS,backward_calls=0),recipe_overrides=dict(microbatch=3))
        before=state_sha(value);expected=next_ids(value)
        with self.assertRaises(WorkBudgetExceeded):value.update()
        self.assertEqual(state_sha(value),before);self.assertEqual(next_ids(value),expected)

    def test_original_successful_target_caps_independent_from_failed_work(self):
        for cap,steps,targets in ((8,0,0),(9,1,9),(16,1,9),(17,2,17)):
            value,_,budget=fixture(self.root/f'target-{cap}',cap=cap)
            try:
                value.stream.order=torch.arange(4)
                for _ in range(steps):value.update()
                before=state_sha(value);costs=budget.ledger.snapshot()
                with patch.object(value,'summed_loss') as model,self.assertRaises(story.ValidTargetBudgetExceeded):value.update()
                model.assert_not_called();self.assertEqual(state_sha(value),before)
                self.assertEqual((value.step,value.tokens),(steps,targets))
                self.assertEqual(budget.ledger.snapshot(),costs)
            finally:budget.ledger.close()

    def test_repeated_evaluation_and_activation_panels_accumulate(self):
        value,_,budget=self.make();data=original.AuthoredWindows()
        first=value.evaluate(data,max_windows=3);second=value.evaluate(data,max_windows=3)
        self.assertEqual(first,second)
        story.activation_summary(value,data);story.activation_summary(value,data)
        actual=budget.ledger.snapshot()
        self.assertEqual(actual['reserved']['evaluation_panels'],2)
        self.assertEqual(actual['completed']['evaluation_windows'],6)
        self.assertEqual(actual['completed']['evaluation_valid_targets'],2*first['valid_targets'])
        self.assertEqual(actual['completed']['activation_panels'],2)
        self.assertEqual(actual['completed']['activation_positions'],16)
        self.assertEqual(actual['completed']['activation_block_applications'],2)

    def test_observation_zero_caps_before_forward_features_or_mode_change(self):
        for key in ('evaluation_panels','activation_panels'):
            value,_,budget=fixture(self.root/key,caps=dict(CAPS,**{key:0}))
            try:
                before=state_sha(value)
                with patch.object(value,'summed_loss') as forward,patch.object(value.model,'apply_block') as features:
                    with self.assertRaises(WorkBudgetExceeded):
                        if key=='evaluation_panels':value.evaluate(original.AuthoredWindows())
                        else:story.activation_summary(value,original.AuthoredWindows())
                forward.assert_not_called();features.assert_not_called();self.assertEqual(state_sha(value),before)
            finally:budget.ledger.close()

    def test_sampling_zero_before_features_or_random_draws(self):
        value,_,_=self.make(native_sampling=True,caps=dict(CAPS,generation_calls=0));before=state_sha(value)
        with patch.object(value.model,'features') as features,patch.object(torch,'multinomial') as draw:
            with self.assertRaises(WorkBudgetExceeded):story.samples(value,AuthoredTokenizer(),max_new=3)
        features.assert_not_called();draw.assert_not_called();self.assertEqual(state_sha(value),before)

    def test_forced_eos_artificial_control_keeps_unused_generation_reservations(self):
        value,_,budget=self.make(native_sampling=True)
        def only_eos(hidden):
            logits=torch.full((*hidden.shape[:-1],50257),-torch.inf)
            logits[...,50256]=0.;return logits
        with patch.object(value.model.lm_head,'forward',side_effect=only_eos):
            answer=story.samples(value,AuthoredTokenizer(),max_new=3)
        self.assertTrue(all(row['token_ids']==[50256] and row['ended_with_eos'] for row in answer))
        actual=budget.ledger.snapshot()
        self.assertEqual(actual['reserved']['generation_calls'],18)
        self.assertEqual(actual['completed']['generation_calls'],6)
        self.assertEqual(actual['completed']['generation_positions'],18)
        self.assertEqual(actual['completed']['multinomial_draws'],3)

    def test_partial_forward_poison_refuses_new_work_observation_and_publication(self):
        value,_,budget=self.make();value.update();path=self.root/'completed1.pt';save(value,path)
        failed=later_failure(value);self.assertTrue(failed['poisoned']);self.assertEqual(value.step,1)
        with patch.object(torch,'save') as serializer:
            for call in (value.update,lambda:value.evaluate(original.AuthoredWindows()),
                         lambda:story.activation_summary(value,original.AuthoredWindows()),
                         lambda:value.save(self.root/'poison.pt')):
                with self.assertRaises(RuntimeError):call()
        serializer.assert_not_called();self.assertFalse((self.root/'poison.pt').exists())
        self.assertEqual(budget.ledger.snapshot()['completed']['train_updates'],1)

    def test_nonfinite_and_partial_optimizer_failures_poison_but_not_commit_targets(self):
        for kind in ('nonfinite','optimizer'):
            value,_,budget=fixture(self.root/kind)
            try:
                if kind=='nonfinite':
                    def failure(*args):return torch.tensor(float('nan'),requires_grad=True)
                    target=value;method='summed_loss';error=FloatingPointError
                else:
                    def failure(*args,**kwargs):
                        with torch.no_grad():next(value.model.parameters()).add_(.01)
                        raise OSError('authored partially mutating optimizer')
                    target=value.optimizer;method='step';error=OSError
                with patch.object(target,method,side_effect=failure),self.assertRaises(error):value.update()
                self.assertTrue(value.poisoned);self.assertEqual((value.step,value.tokens,value.positions),(0,0,0))
                self.assertEqual(budget.ledger.snapshot()['reserved']['train_updates'],1)
                self.assertTrue(budget.ledger.snapshot()['failed_tickets'])
            finally:budget.ledger.close()

    def test_mandatory_original_hook_cannot_be_removed(self):
        value,_,_=self.make();value.work_budget=None
        with patch.object(value.model,'forward') as model:
            with self.assertRaises(ValueError):value.update()
        model.assert_not_called()

    def test_independent_receipt_metadata_bounds_and_no_follow(self):
        fifo=self.root/'receipt-fifo';os.mkfifo(fifo)
        duplicate=self.root/'duplicate.json';duplicate.write_bytes(b'{"schema":1,"schema":2}')
        oversized=self.root/'huge.json';oversized.write_bytes(b' '*65537)
        symlink=self.root/'alias';symlink.symlink_to(duplicate)
        for path in (fifo,duplicate,oversized,symlink):
            with self.assertRaises((ValueError,OSError)):read_story_receipt(path)

    def test_saved_prefix_is_pre_reservation_and_later_save_charge_stays_spent(self):
        value,_,budget=self.make();value.update();before=budget.ledger.snapshot()
        path=self.root/'completed1.pt';receipt=save(value,path)
        state=torch.load(path,map_location='cpu',weights_only=True)
        self.assertEqual(receipt['work_prefix'],before);self.assertEqual(state['story_work_prefix'],before)
        after=budget.ledger.snapshot();self.assertEqual(after['reserved']['save_operations'],1)
        self.assertEqual(after['completed']['save_operations'],1)
        self.assertGreater(after['sequence'],receipt['work_prefix']['sequence'])
        value.restore(path,work_receipt=receipt)
        later=budget.ledger.snapshot();self.assertEqual(later['reserved']['save_operations'],1)
        self.assertEqual(later['reserved']['restore_operations'],1)

    def test_save_and_restore_zero_before_serializer_payload_read_or_model_application(self):
        for key in ('save_operations','restore_operations'):
            value,_,budget=fixture(self.root/key,caps=dict(CAPS,**{key:0}))
            try:
                path=self.root/key/'completed0.pt'
                receipt=save(value,path) if key=='restore_operations' else None
                before=state_sha(value)
                with patch.object(torch,'save') as serializer,patch.object(torch,'load') as loader,\
                     patch.object(story,'checked_payload') as payload,patch.object(value.model,'load_state_dict') as apply:
                    with self.assertRaises(WorkBudgetExceeded):
                        if key=='save_operations':value.save(path)
                        else:value.restore(path,work_receipt=receipt)
                serializer.assert_not_called();loader.assert_not_called();payload.assert_not_called();apply.assert_not_called()
                self.assertEqual(state_sha(value),before);self.assertFalse(value.poisoned)
                self.assertEqual(budget.ledger.snapshot()['reserved'][key],0)
            finally:budget.ledger.close()

    def test_receipt_schema_cursor_science_and_prefix_mutations_before_payload(self):
        value,science,budget=self.make();value.update();path=self.root/'completed1.pt';receipt=save(value,path)
        cases=[]
        for key,item in (('schema','obsolete'),('completed_updates',True),('completed_updates',6),
                         ('contract_sha256','0'*64),('payload_bytes',False)):
            changed=deepcopy(receipt);changed[key]=item;cases.append(changed)
        changed=deepcopy(receipt);changed['work_prefix']['reserved']['train_updates']+=1;cases.append(changed)
        changed=deepcopy(receipt);changed['work_prefix']['sequence']=True;cases.append(changed)
        for altered in cases:
            before=budget.ledger.snapshot()
            with patch.object(story,'checked_payload') as payload,patch.object(value.model,'load_state_dict') as apply:
                with self.assertRaises(ValueError):value.restore(path,work_receipt=altered)
            payload.assert_not_called();apply.assert_not_called();self.assertEqual(budget.ledger.snapshot(),before)

    def test_copied_or_new_physical_journal_cannot_replenish_saved_allowance(self):
        value,science,budget=self.make();value.update();path=self.root/'completed1.pt';receipt=save(value,path)
        journal=budget.ledger.path;contract=budget.contract;budget.ledger.close()
        copied=self.root/'copied.jsonl';shutil.copyfile(journal,copied);copied.chmod(0o600)
        with self.assertRaises(ValueError):StoryWorkBudget.open(copied,contract=contract,
            scientific_contract=science,invocation_id='copied',receipt=receipt)
        new=StoryWorkBudget.create(self.root/'new.jsonl',contract=contract,
            scientific_contract=science,invocation_id='new')
        try:
            with self.assertRaises(ValueError):new.ledger.validate_snapshot(receipt['work_prefix'])
        finally:new.ledger.close()
        for changed in (dict(contract,schema='obsolete'),dict(contract,extra=1),
                        dict(contract,limits=dict(CAPS,train_updates=100001))):
            with self.assertRaises(ValueError):StoryWorkBudget.open(journal,contract=changed,
                scientific_contract=science,invocation_id='changed',receipt=receipt)

    def test_payload_bytes_prefix_rng_adam_and_noncanonical_stream_before_application(self):
        for kind in ('bytes','prefix','rng','adam','adamhalf','adamsparse','cursor','targets',
                     'positions','missingpositions','derivedpositions'):
            value,_,budget=fixture(self.root/kind)
            try:
                for _ in range(2):value.update()
                path=self.root/kind/'completed2.pt';receipt=save(value,path)
                actual=torch.load(path,map_location='cpu',weights_only=True)
                self.assertEqual((actual['stream']['cursor'],actual['stream']['epoch']),(4,0))
                output=self.root/kind/'malformed.pt'
                def change(state):
                    if kind=='bytes':state['authored_byte_mismatch']=1
                    elif kind=='prefix':state['story_work_prefix']['reserved']['train_updates']+=1
                    elif kind=='rng':state['rng']=torch.zeros(3,dtype=torch.float32)
                    elif kind=='adam':next(iter(state['optimizer']['state'].values()))['step']=torch.tensor(True)
                    elif kind=='adamhalf':next(iter(state['optimizer']['state'].values()))['step']=torch.tensor(2.,dtype=torch.float16)
                    elif kind=='adamsparse':next(iter(state['optimizer']['state'].values()))['step']=torch.tensor(2.).to_sparse()
                    elif kind=='cursor':state['stream']['cursor']=0;state['stream']['epoch']=1
                    elif kind=='targets':state['tokens']=-1
                    elif kind=='positions':state['processed_positions']=-1
                    elif kind=='missingpositions':del state['processed_positions']
                    elif kind=='derivedpositions':state['processed_positions_accounting']='derived-legacy-fixed-geometry-plus-measured-continuation'
                modified=changed_payload(path,receipt,output,change)
                if kind=='bytes':modified=receipt
                with patch.object(value.model,'load_state_dict') as model,patch.object(value.optimizer,'load_state_dict') as optimizer:
                    with self.assertRaises(ValueError):value.restore(output,work_receipt=modified)
                model.assert_not_called();optimizer.assert_not_called();self.assertTrue(value.poisoned)
                self.assertEqual(budget.ledger.snapshot()['reserved']['restore_operations'],1)
                self.assertTrue(budget.ledger.snapshot()['failed_tickets'])
            finally:budget.ledger.close()

    def test_admitted_partial_restore_poison_prevents_further_work(self):
        value,_,budget=self.make();value.update();path=self.root/'completed1.pt';receipt=save(value,path)
        with patch.object(value.optimizer,'load_state_dict',side_effect=OSError('authored partial restore')):
            with self.assertRaises(OSError):value.restore(path,work_receipt=receipt)
        self.assertTrue(value.poisoned);self.assertTrue(budget.ledger.snapshot()['failed_tickets'])
        with patch.object(value,'summed_loss') as model,self.assertRaises(RuntimeError):value.update()
        model.assert_not_called()

    def test_partial_serializer_and_publication_preserve_old_durable_boundary(self):
        for kind in ('serialization','link','receipt'):
            value,_,budget=fixture(self.root/kind)
            try:
                old=self.root/kind/'completed0.pt';receipt=save(value,old)
                oldbytes=old.read_bytes();oldreceipt=Path(str(old)+'.work.json').read_bytes()
                value.update();new=self.root/kind/'completed1.pt'
                def serializer(state,handle):handle.write(b'partial');handle.flush();raise OSError('authored serializer')
                target=torch if kind=='serialization' else (story.os if kind=='link' else story)
                method='save' if kind=='serialization' else ('link' if kind=='link' else 'publish_story_receipt')
                effect=serializer if kind=='serialization' else OSError('authored publication refusal')
                with patch.object(target,method,side_effect=effect),self.assertRaises(OSError):value.save(new)
                self.assertEqual(old.read_bytes(),oldbytes);self.assertEqual(Path(str(old)+'.work.json').read_bytes(),oldreceipt)
                self.assertEqual(budget.last_receipt,receipt);self.assertFalse(Path(str(new)+'.work.json').exists())
                if kind=='receipt':self.assertTrue(new.exists());self.assertFalse(new.with_suffix('.pt.tmp').exists())
                else:self.assertTrue(new.with_suffix('.pt.tmp').exists());self.assertFalse(new.exists())
                self.assertEqual(budget.ledger.snapshot()['reserved']['save_operations'],2)
                self.assertEqual(budget.ledger.snapshot()['completed']['save_operations'],1)
                self.assertTrue(budget.ledger.snapshot()['failed_tickets'])
            finally:budget.ledger.close()

    def test_observation_failures_retain_admitted_panels_without_poisoning_model(self):
        for kind in ('evaluate','activation'):
            value,_,budget=fixture(self.root/kind)
            try:
                before=state_sha(value)
                target=value if kind=='evaluate' else value.model
                method='summed_loss' if kind=='evaluate' else 'apply_block'
                actual=getattr(target,method);calls=0
                def failure(*args,**kwargs):
                    nonlocal calls
                    calls+=1
                    if kind=='activation' or calls==2:raise OSError('authored observation failure')
                    return actual(*args,**kwargs)
                with patch.object(target,method,side_effect=failure),self.assertRaises(OSError):
                    if kind=='evaluate':value.evaluate(original.AuthoredWindows())
                    else:story.activation_summary(value,original.AuthoredWindows())
                self.assertEqual(state_sha(value),before);self.assertFalse(value.poisoned)
                self.assertTrue(budget.ledger.snapshot()['failed_tickets'])
                self.assertEqual(budget.ledger.snapshot()['reserved']['evaluation_panels' if kind=='evaluate' else 'activation_panels'],1)
            finally:budget.ledger.close()

    def test_live_recipe_config_data_device_cap_and_replaced_hook_before_model(self):
        for kind in ('recipe','config','data','device','cap','hook'):
            value,science,budget=fixture(self.root/kind)
            second=None
            try:
                if kind=='recipe':value.recipe=replace(value.recipe,peak_lr=.006)
                elif kind=='config':value.model.cfg=replace(value.model.cfg,max_length=7)
                elif kind=='data':value.train_data=original.AuthoredWindows();value.train_data.identity='changed-authored-ids'
                elif kind=='device':value.device=torch.device('meta')
                elif kind=='cap':value.valid_target_budget=1001
                else:
                    second=StoryWorkBudget.create(self.root/kind/'new-allowance.jsonl',contract=budget.contract,
                        scientific_contract=science,invocation_id='replacement');value.work_budget=second
                with patch.object(value,'summed_loss') as model,self.assertRaises(ValueError):value.update()
                model.assert_not_called()
            finally:
                if second:second.close()
                budget.ledger.close()

    def test_actual_cli_missing_pair_and_zero_initialization_preallocation(self):
        command=command_module();caps=self.root/'limits.json';write(caps,story_work_contract(dict(CAPS,model_initializations=0),BOUND))
        base=['train_stories.py','train','--data',str(self.root/'authored-data'),
            '--output',str(self.root/'out'),'--device','cpu','--total','5','--accumulation','2',
            '--valid-target-budget','1000','--work-limits',str(caps)]
        with patch.object(sys,'argv',base),patch.object(story,'Windows') as windows,\
             patch.object(story,'StoriesDecoder') as model,self.assertRaises(ValueError):command.main()
        windows.assert_not_called();model.assert_not_called()
        with patch.object(sys,'argv',base+['--work-journal',str(self.root/'work.jsonl')]),\
             patch.object(story,'Windows',side_effect=lambda *args:original.AuthoredWindows()),\
             patch.object(story,'baseline',return_value=configuration()),patch.object(story,'MemoryMonitor',NoMonitor),\
             patch.object(story,'StoriesDecoder') as model,patch.object(torch,'load') as payload,\
             patch.object(torch,'manual_seed') as rng,self.assertRaises(WorkBudgetExceeded):command.main()
        model.assert_not_called();payload.assert_not_called();rng.assert_not_called()
        self.assertTrue((self.root/'work.jsonl').exists())

    def test_actual_cli_durable_initial_observer_failure_and_same_journal_retry_resume(self):
        command=command_module();caps=self.root/'limits.json';write(caps,story_work_contract(CAPS,BOUND))
        journal=self.root/'native-work.jsonl'
        data=self.root/'authored-data';data.mkdir();write(data/'manifest.json',{'train':{'files':{}},'valid':{'files':{}}})
        class NativeTokenizer:
            @classmethod
            def from_file(cls,path):return AuthoredTokenizer()
        def invoke(output,stop,resume=None,receipt=None):
            argv=['train_stories.py','train','--data',str(self.root/'authored-data'),
                '--output',str(output),'--device','cpu','--total','5','--stop-after',str(stop),
                '--warmup','1','--accumulation','2','--valid-windows','2','--sample-tokens','3',
                '--checkpoint-every','2','--peak-lr','0.003','--floor-lr','0.0003',
                '--valid-target-budget','1000','--work-limits',str(caps),'--work-journal',str(journal)]
            if resume is not None:argv+=['--resume',str(resume),'--resume-work-receipt',str(receipt)]
            with patch.object(sys,'argv',argv),patch.object(story,'Windows',side_effect=lambda *args:original.AuthoredWindows()),\
                 patch.object(story,'baseline',return_value=configuration(native_sampling=True)),\
                 patch.object(story,'MemoryMonitor',NoMonitor),patch('tokenizers.Tokenizer',NativeTokenizer),patch('builtins.print'):
                return command.main()
        actual=story.Session.summed_loss
        def observer_fail(session,*args,**kwargs):raise OSError('authored first native CLI observer failure')
        with patch.object(story.Session,'summed_loss',autospec=True,side_effect=observer_fail),self.assertRaises(OSError):
            invoke(self.root/'failed-observer',2)
        first=self.root/'failed-observer/update-000000.pt';retained=read_story_receipt(str(first)+'.work.json')
        self.assertEqual(retained['completed_updates'],0)
        self.assertFalse((self.root/'failed-observer/initial.json').exists())
        self.assertEqual(torch.load(first,weights_only=True)['story_work_prefix'],retained['work_prefix'])
        invoke(self.root/'retry',2,first,Path(str(first)+'.work.json'))
        second=self.root/'retry/update-000002.pt'
        invoke(self.root/'resume',5,second,Path(str(second)+'.work.json'))
        result=json.loads((self.root/'resume/completion.json').read_text())
        work=result['work_ledger']
        self.assertTrue(result['schedule_complete']);self.assertEqual(result['completed_updates'],5)
        self.assertEqual(result['cumulative_processed_positions'],80)
        self.assertEqual(work['reserved']['train_updates'],5)
        self.assertEqual(work['completed']['train_updates'],5)
        self.assertEqual(work['reserved']['model_initializations'],3)
        self.assertEqual(work['reserved']['restore_operations'],2)
        self.assertEqual(work['reserved']['evaluation_panels'],6)
        self.assertEqual(work['completed']['evaluation_panels'],5)
        self.assertTrue(work['failed_tickets'])
        self.assertEqual(len((self.root/'retry/metrics.jsonl').read_text().splitlines()),2)
        self.assertEqual(len((self.root/'resume/metrics.jsonl').read_text().splitlines()),3)

    def test_actual_cli_terminal_or_mismatched_receipt_before_journal_payload_and_model(self):
        value,_,budget=self.make();path=self.root/'completed0.pt';retained=save(value,path)
        caps=self.root/'limits.json';write(caps,budget.contract);budget.ledger.close()
        data=self.root/'authored-data';data.mkdir();write(data/'manifest.json',{'train':{'files':{}},'valid':{'files':{}}})
        command=command_module()
        for kind in ('terminal','missing','caps'):
            receipt=deepcopy(retained)
            if kind=='terminal':receipt['completed_updates']=5
            elif kind=='caps':receipt['contract_sha256']='0'*64
            receiptpath=self.root/f'{kind}-receipt.json';write(receiptpath,receipt)
            argv=['train_stories.py','train','--data',str(data),'--output',str(self.root/kind),
                '--device','cpu','--total','5','--accumulation','2','--resume',str(path),
                '--work-limits',str(caps),'--work-journal',str(budget.ledger.path),'--valid-target-budget','1000']
            if kind!='missing':argv+=['--resume-work-receipt',str(receiptpath)]
            with patch.object(sys,'argv',argv),patch.object(story,'Windows',side_effect=lambda *args:original.AuthoredWindows()),\
                 patch.object(story,'baseline',return_value=configuration()),patch.object(story,'StoriesDecoder') as model,\
                 patch.object(torch,'load') as loader,patch.object(story,'MemoryMonitor') as monitor,self.assertRaises(ValueError):command.main()
            model.assert_not_called();loader.assert_not_called();monitor.assert_not_called()
            self.assertFalse((self.root/kind).exists())

    def test_unaccounted_legacy_checkpoint_remains_explicit_not_adopted(self):
        reference,_=plain();reference.update();path=self.root/'legacy.pt';reference.save(path)
        value,_,_=self.make()
        with patch.object(torch,'load') as loader,self.assertRaises(ValueError):value.restore(path)
        loader.assert_not_called()
        unaccounted,_=plain();unaccounted.restore(path)
        self.assertEqual((unaccounted.step,unaccounted.tokens),(reference.step,reference.tokens))

    def test_unaccounted_session_cannot_attach_existing_accounted_allowance(self):
        legacy,_=plain();value,_,budget=self.make();legacy.work_budget=budget
        with patch.object(legacy,'summed_loss') as model,self.assertRaises(ValueError):legacy.update()
        model.assert_not_called();self.assertEqual(budget.ledger.snapshot()['reserved']['train_updates'],0)

    def test_actual_cli_designated_payload_aliases_before_any_bootstrap_content(self):
        command=command_module()
        for role in ('caps','receipt','journal','manifest.json','tokenizer.json','train.bin',
                     'valid.bin','train.windows.npy','valid.windows.npy','source'):
            for hardlink in (False,True):
                with self.subTest(role=role,hardlink=hardlink):
                    directory=self.root/f'{role}-{int(hardlink)}';directory.mkdir()
                    data=directory/'data';data.mkdir();payload=directory/'payload.pt';payload.write_bytes(b'authored-no-read')
                    caps=directory/'caps.json';receipt=directory/'receipt.json';journal=directory/'journal.jsonl'
                    designated={'caps':caps,'receipt':receipt,'journal':journal}.get(role,data/role)
                    if role=='source':
                        payload=ROOT/'src/dongxi_llms/stories_training.py'
                        if hardlink:
                            alias=directory/'source-alias.pt';os.link(payload,alias);payload=alias
                    elif hardlink:os.link(payload,designated)
                    elif role=='caps':caps=payload
                    elif role=='receipt':receipt=payload
                    elif role=='journal':journal=payload
                    else:payload.rename(designated);payload=designated
                    argv=['train_stories.py','train','--data',str(data),'--output',str(directory/'out'),
                        '--device','cpu','--total','5','--resume',str(payload),
                        '--work-limits',str(caps),'--work-journal',str(journal),'--resume-work-receipt',str(receipt)]
                    with patch.object(sys,'argv',argv),patch.object(story,'read_story_limits') as limits,\
                         patch.object(story,'read_story_receipt') as receipter,patch.object(story,'Windows') as windows,\
                         patch.object(story,'StoriesDecoder') as model,patch.object(torch,'load') as loader,\
                         patch.object(StoryWorkBudget,'open') as opener,self.assertRaises(ValueError):command.main()
                    for spy in (limits,receipter,windows,model,loader,opener):spy.assert_not_called()
                    self.assertFalse((directory/'out').exists())

    def test_actual_cli_additional_manifest_input_alias_before_native_file_hashes(self):
        command=command_module()
        for hardlink in (False,True):
            directory=self.root/f'extra-input-{int(hardlink)}';directory.mkdir()
            data=directory/'data';data.mkdir();payload=directory/'payload.pt';payload.write_bytes(b'authored-no-read')
            extra=data/'additional-authored.bin'
            if hardlink:os.link(payload,extra)
            else:payload.rename(extra);payload=extra
            write(data/'manifest.json',{'train':{'files':{'additional-authored.bin':'0'*64}},'valid':{'files':{}}})
            argv=['train_stories.py','train','--data',str(data),'--output',str(directory/'out'),
                '--device','cpu','--total','5','--resume',str(payload),'--work-limits',str(directory/'caps.json'),
                '--work-journal',str(directory/'journal.jsonl'),'--resume-work-receipt',str(directory/'receipt.json')]
            with patch.object(sys,'argv',argv),patch.object(story,'read_story_limits') as limits,\
                 patch.object(story,'read_story_receipt') as receipt,patch.object(story,'Windows') as windows,\
                 patch.object(story,'digest') as hasher,patch.object(story,'StoriesDecoder') as model,\
                 patch.object(torch,'load') as loader,self.assertRaises(ValueError):command.main()
            for spy in (limits,receipt,windows,hasher,model,loader):spy.assert_not_called()

    def test_public_loss_outside_wrong_operation_and_duplicate_native_callback(self):
        value,_,budget=self.make();x,y=value.train_data.batch([0])
        with patch.object(value.model,'forward',wraps=value.model.forward) as model:
            with self.assertRaises(RuntimeError):value.summed_loss(x,y)
            self.assertEqual(model.call_count,0)
            with self.assertRaises(RuntimeError):
                with value.operation(dict(save_operations=1),'authored-non-policy-operation'):
                    value.summed_loss(x,y)
            self.assertEqual(model.call_count,0)
            with self.assertRaises(RuntimeError):
                with value.operation(dict(save_operations=1),'authored-non-policy-operation'):
                    value._call_loss(x,y)
            self.assertEqual(model.call_count,0)
            actual=value.summed_loss
            def duplicate(a,b):
                first=actual(a,b);actual(a,b);return first
            with patch.object(value,'summed_loss',side_effect=duplicate),self.assertRaises(RuntimeError):value.update()
            self.assertEqual(model.call_count,1)
        self.assertTrue(value.poisoned);self.assertIsNone(value._loss_permit)
        self.assertEqual((value.step,value.tokens,value.positions),(0,0,0))
        work=budget.ledger.snapshot();self.assertEqual(work['reserved']['train_updates'],1)
        self.assertEqual(work['completed']['backward_calls'],0);self.assertTrue(work['failed_tickets'])


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--child':
        print(json.dumps(fresh_child(sys.argv[2],sys.argv[3]),sort_keys=True,allow_nan=False));return
    unittest.main()


if __name__=='__main__':main()
