"""Pure standard-library authored archive controls; not model-run evidence."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('story_publication_archive',ROOT/'scripts/archive_native_story_publication.py')
lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)


class StoryPublicationArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(prefix='dongxi-story-archive-authored-');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.evidence=self.root/'experiments/reports'/lab.STEM
        self.evidence.mkdir(parents=True);self.output=self.root/'experiments/reports'/'authored-archive'
        self.runtime=self.root/'authored-python';self.runtime.write_text('inert authored interpreter identity\n')
        self.lock=self.root/'authored.lock';self.lock.write_text('inert authored lock\n')
        for name in (*lab.TRAINING_SOURCES,*lab.EVALUATION_SOURCES,*lab.ARCHIVE_SOURCES):
            path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('authored archive source '+name+'\n')
        for current in (patch.object(lab,'ROOT',self.root),patch.object(lab,'INTERPRETER',str(self.runtime)),patch.object(lab,'LOCK',str(self.lock))):
            current.start();self.addCleanup(current.stop)
        contract=dict(id='authored-publication-control',items=[dict(id=f'story-{i:02d}',source_group=f'original-opening-{i:02d}',
            prompt=f'Authored control opening{i}.') for i in range(1,13)],decoding=[{'authored_index':i} for i in range(4)],
            predetermined_updates=list(lab.UPDATES),context_window=1024,max_new_tokens=256)
        fingerprint=lab.canonical_hash(contract);contract['logical_contract_sha256']=fingerprint
        current=patch.object(lab,'ORIGINAL_CONTRACT_SHA',fingerprint);current.start();self.addCleanup(current.stop)
        interface=dict(evaluation_execution=lab.EXECUTION,observed_publication_prefixes={item['id']:[50256,1] for item in contract['items']})
        bindings=dict(training=dict(sources={name:lab.digest(self.root/name) for name in lab.TRAINING_SOURCES},
            interpreter=dict(lab.digest(self.runtime),actual_path=str(self.runtime)),environment_lock=dict(lab.digest(self.lock),actual_path=str(self.lock)),
            data={'train.bin':{'sha256':'a'*64,'scope':'Historical fixture declaration, body intentionally absent'}}),
            evaluation={name:lab.digest(self.root/name) for name in lab.EVALUATION_SOURCES})
        self.declarations={};plans=[]
        for cid in lab.ALL_CHECKPOINTS:
            arm,_,update=cid.partition('-update-');available=cid in lab.AVAILABLE
            declaration=dict(checkpoint_id=cid,arm=arm,update=int(update),status='available' if available else 'missing-checkpoint',
                payload_identity={'sha256':'a'*64,'bytes':99,'path':'/authored/unavailable-model-body'} if available else None)
            hashes={}
            if available:
                path=self.evidence/(cid+'-receipt.json');self.write(path,dict(completed_updates=int(update),payload_sha256='a'*64,payload_bytes=99))
                declaration['receipt_identity']=lab.digest(path);hashes[path.name]=declaration['receipt_identity']['sha256']
                hashes['update-'+update+'.pt']='a'*64
            path=self.evidence/(cid+'-declaration.json');self.write(path,declaration);hashes[path.name]=lab.digest(path)['sha256']
            self.declarations[cid]=declaration
            plans.append(dict(checkpoint_id=cid,update=int(update),checkpoint_files=hashes,provenance='model-generated',
                interface_sha256=lab.canonical_hash(interface),generation_source_sha256={name:bindings['evaluation'][name]['sha256'] for name in lab.EVALUATION_SOURCES}))
        caps={'limits':{'evaluation_windows':128,'generation_sequences':48}}
        for name,value in (('contract.json',contract),('interface.json',interface),('checkpoints.json',plans),('work-caps.json',caps)):
            self.write(self.evidence/name,value)
        self.prepared=dict(schema='dongxi-fixed-native-story-evaluation-v1',run_id=lab.RUN_ID,bindings=bindings,
            contract=contract,interface=interface,checkpoint_plan=plans,checkpoints=self.declarations,caps=caps,
            files={name:lab.digest(self.evidence/name) for name in ('contract.json','interface.json','checkpoints.json','work-caps.json')},
            execution_labels=dict(evaluation=lab.EXECUTION,actual_producer_entries={arm:dict(event='actual-story-deterministic-entry',
                attention_backend='SDPA MATH only',deterministic_algorithms=True,tf32=False) for arm in lab.ARMS}))
        self.sign_prepared();rows=[];invocations=[]
        for cid in lab.ALL_CHECKPOINTS:
            if cid not in lab.AVAILABLE:
                invocations.append(dict(checkpoint_id=cid,status='missing-checkpoint',actual_exit_code=None));continue
            supervision=dict(schema='dongxi-native-profile-watchdog-result-v1',status='completed',actual_exit_code=0,
                actual_native_profile_executed=True,child_command=[str(self.runtime),str(self.root/'scripts/run_native_story_evaluation.py'),
                    '--run-id',lab.RUN_ID,'--checkpoint-child',cid],limits=dict(external_seconds=900,reserve_bytes=25*1024**3),
                observations=[{'authored_sample':True}],child_seconds=.1)
            self.write(self.evidence/(cid+'-returned-supervision.json'),supervision);invocations.append(dict(checkpoint_id=cid,**supervision))
            root=self.root/'outputs'/lab.STEM/cid
            nll={split:dict(nll=2.5+index,selected_window_ids=list(range(index*100,index*100+64)),selection_seed=seed,
                valid_targets=32,physical_positions=65536,windows=64,complete_prepared_split=False,
                evaluation_execution=lab.EXECUTION,observed_logits_dtypes=['torch.bfloat16'],scope='Authored matched64-window measurement control')
                for index,(split,seed) in enumerate((('train',409),('valid',909)))}
            self.write(root/'fixed-nll.json',nll)
            self.write(root/'child-result.json',dict(status='completed',checkpoint_id=cid,evaluation_execution=lab.EXECUTION,
                work=dict(open_tickets=[],failed_tickets=[],limits=caps['limits'],completed=dict(evaluation_windows=128,evaluation_panels=2,
                    evaluation_valid_targets=64,policy_forward_calls=8,policy_forward_positions=131072,generation_sequences=48)),
                cuda_peak_allocated_bytes=100,cuda_peak_reserved_bytes=200,scope='Authored retained measurements, no actual CUDA work'))
            plan=next(plan for plan in plans if plan['checkpoint_id']==cid)
            for item in contract['items']:
                for d in range(4):
                    text='Authored negative continuation'
                    rows.append(dict(checkpoint_id=cid,item_id=item['id'],decoding_index=d,attempt_index=0,record_id=f'{cid}-{item["id"]}-d{d}',
                        contract_sha256=fingerprint,checkpoint_sha256=lab.canonical_hash(plan),opening_sha256=lab.hashlib.sha256(item['prompt'].encode()).hexdigest(),
                        source_group=item['source_group'],text=text,text_sha256=lab.hashlib.sha256(text.encode()).hexdigest(),error=None,
                        stop_reason='natural-eos',generated_tokens=1,token_ids=[50256],cost={'generation_tokens':1}))
        (self.evidence/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        expected=[dict(checkpoint_id=cid,item_id=item['id'],decoding_index=d,record_id=f'{cid}-{item["id"]}-d{d}' if cid in lab.AVAILABLE else None)
            for cid in lab.ALL_CHECKPOINTS for item in contract['items'] for d in range(4)]
        self.coverage=dict(status='generated-awaiting-independent-ratings',actual_records=192,missing_cells=288,generation_failures=0,
            expected_cells=expected,invocations=invocations,artifact_identities={name:lab.digest(self.evidence/name)
                for name in ('contract.json','checkpoints.json','records.jsonl')},scope='Authored incomplete original geometry')
        self.write(self.evidence/'coverage.json',self.coverage)

    def write(self,path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')

    def sign_prepared(self):
        self.prepared['preparation_sha256']=lab.canonical_hash({key:value for key,value in self.prepared.items() if key!='preparation_sha256'})
        self.write(self.evidence/'preparation.json',self.prepared)

    def nll_path(self,cid=None):return self.root/'outputs'/lab.STEM/(cid or lab.AVAILABLE[0])/'fixed-nll.json'

    def test_archive_embeds_raw_measurements_receipts_and_original_missing_slice(self):
        originals={cid:json.loads(self.nll_path(cid).read_text()) for cid in lab.AVAILABLE};report=lab.archive(self.output)
        self.assertEqual((report['actual_available_checkpoints'],report['actual_records'],report['missing_records']),(4,192,288))
        self.assertEqual(report['raw']['coverage'],self.coverage);self.assertEqual(set(report['raw']['measurements']),set(lab.AVAILABLE))
        for cid in lab.AVAILABLE:
            raw=report['raw']['measurements'][cid];self.assertEqual(raw['fixed_nll'],originals[cid]);self.assertEqual(raw['child_result']['cuda_peak_reserved_bytes'],200)
            self.assertEqual(raw['returned_supervision'],json.loads((self.evidence/(cid+'-returned-supervision.json')).read_text()))
        self.assertEqual(report['archive_sha256'],lab.canonical_hash({key:value for key,value in report.items() if key!='archive_sha256'}))
        accepted=json.loads((self.output/'acceptance.json').read_text());self.assertEqual(accepted['archive'],lab.digest(self.output/'archive.json',lab.TOTAL_BYTES))
        self.assertFalse((self.output/'failure.json').exists());self.assertFalse(any(self.root.rglob('*.pt')))

    def test_existing_output_and_outside_reports_refuse_without_overwrite(self):
        self.output.mkdir();sentinel=self.output/'keep.txt';sentinel.write_text('preserve')
        with self.assertRaises(FileExistsError):lab.archive(self.output)
        self.assertEqual(sentinel.read_text(),'preserve')
        with self.assertRaisesRegex(ValueError,'experiments/reports'):lab.archive(self.root/'outside')

    def test_changed_actual_source_refuses_and_preserves_raw_measurements(self):
        path=self.root/lab.EVALUATION_SOURCES[0];path.write_text('changed actual source')
        original=self.nll_path().read_bytes()
        with self.assertRaisesRegex(ValueError,'source changed'):lab.archive(self.output)
        self.assertEqual(self.nll_path().read_bytes(),original);self.assertTrue((self.output/'failure.json').exists())

    def test_missing_or_failed_child_cannot_be_archived_as_complete(self):
        path=self.nll_path().with_name('child-result.json');child=json.loads(path.read_text());child['status']='failed';self.write(path,child)
        with self.assertRaisesRegex(ValueError,'completed selected child'):lab.archive(self.output)
        self.assertFalse((self.output/'archive.json').exists())

    def test_missing_ignored_nll_document_refuses_without_touching_other_inputs(self):
        self.nll_path().unlink();retained=self.nll_path(lab.AVAILABLE[1]).read_bytes()
        with self.assertRaises(FileNotFoundError):lab.archive(self.output)
        self.assertEqual(self.nll_path(lab.AVAILABLE[1]).read_bytes(),retained)
        self.assertTrue((self.output/'failure.json').exists());self.assertFalse((self.output/'acceptance.json').exists())

    def test_failed_supervision_even_rebound_coverage_refuses(self):
        cid=lab.AVAILABLE[0];path=self.evidence/(cid+'-returned-supervision.json');supervision=json.loads(path.read_text())
        supervision['actual_exit_code']=7;self.write(path,supervision)
        self.coverage['invocations'][0]=dict(checkpoint_id=cid,**supervision);self.write(self.evidence/'coverage.json',self.coverage)
        with self.assertRaisesRegex(ValueError,'supervisor receipt'):lab.archive(self.output)

    def test_duplicate_records_refuse_even_with_rebound_byte_hash(self):
        path=self.evidence/'records.jsonl';rows=path.read_text().splitlines();rows[1]=rows[0];path.write_text('\n'.join(rows)+'\n')
        self.coverage['artifact_identities']['records.jsonl']=lab.digest(path);self.write(self.evidence/'coverage.json',self.coverage)
        with self.assertRaisesRegex(ValueError,'Duplicate/unknown'):lab.archive(self.output)

    def test_changed_record_bytes_or_coverage_counts_refuse(self):
        self.coverage['actual_records']=191;self.write(self.evidence/'coverage.json',self.coverage)
        with self.assertRaisesRegex(ValueError,'coverage changed'):lab.archive(self.output)

    def test_record_whitespace_change_is_rejected_by_original_coverage_hash(self):
        path=self.evidence/'records.jsonl';path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'byte identity changed'):lab.archive(self.output)

    def test_duplicate_original_coverage_cell_refuses_even_with_correct_counts(self):
        self.coverage['expected_cells'][1]=deepcopy(self.coverage['expected_cells'][0]);self.write(self.evidence/'coverage.json',self.coverage)
        with self.assertRaisesRegex(ValueError,'coverage changed'):lab.archive(self.output)

    def test_nonfinite_nll_and_whole_split_claim_refuse(self):
        nll=json.loads(self.nll_path().read_text());nll['train']['nll']=float('nan');self.write(self.nll_path(),nll)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):lab.archive(self.output)
        nll['train']['nll']=2.5;nll['train']['complete_prepared_split']=True;self.write(self.nll_path(),nll)
        with self.assertRaisesRegex(ValueError,'64-window'):lab.archive(self.output.with_name('authored-archive-02'))

    def test_changed_nll_window_selection_or_child_counters_refuse(self):
        path=self.nll_path(lab.AVAILABLE[1]);nll=json.loads(path.read_text());nll['train']['selected_window_ids'].reverse();self.write(path,nll)
        with self.assertRaisesRegex(ValueError,'match across'):lab.archive(self.output)

    def test_automatic_publication_attention_must_not_be_relabelled_math_training(self):
        path=self.nll_path();nll=json.loads(path.read_text());nll['train']['evaluation_execution']=dict(lab.EXECUTION,attention_declaration='SDPA MATH only');self.write(path,nll)
        with self.assertRaisesRegex(ValueError,'automatic-SDPA'):lab.archive(self.output)

    def test_before_after_hashes_refuse_input_mutation_during_collection(self):
        collect=lab.collect
        def mutate(capture):
            report=collect(capture);self.nll_path().write_text(self.nll_path().read_text()+' ');return report
        with patch.object(lab,'collect',side_effect=mutate),self.assertRaisesRegex(ValueError,'bytes changed before publication'):
            lab.archive(self.output)
        self.assertFalse((self.output/'archive.json').exists());self.assertTrue((self.output/'failure.json').exists())

    def test_duplicate_json_fields_and_explicit_byte_caps_refuse(self):
        self.nll_path().write_text('{"train":{},"train":{}}\n')
        with self.assertRaisesRegex(ValueError,'Duplicate JSON'):lab.archive(self.output)
        path=self.root/'bounded.json';path.write_text('x'*100)
        with self.assertRaisesRegex(ValueError,'Bounded nonempty'):lab.digest(path,99)


if __name__=='__main__':unittest.main()
