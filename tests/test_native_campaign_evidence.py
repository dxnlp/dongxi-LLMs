"""Small authored CPU snapshot controls; never actual campaign/GPU results."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack,redirect_stdout
import io
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_campaign_snapshot',ROOT/'scripts/assemble_native_campaign_evidence.py')
adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)


def entry(path,identifier='authored',stages=('story-control-smoke','story-control-recovery'),kind='recovery'):
    return dict(id=identifier,stage_ids=list(stages),path=str(path),kind=kind,
        scope='Authored inert CPU evidence control, not a model experiment',output_roots=[],selected=True)


def supervisor(*,code=0,native=True,script='run_chapter09_spark_sft.py',pid=17):
    return dict(status='completed' if code==0 else 'failed',actual_exit_code=code,
        actual_native_profile_executed=native,child_pid=pid,child_seconds=.01,
        child_command=['/authored/inert/python',script],minimum_sampled_available_bytes=30000000000)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n')


def story_rating_bundle_fixture(root,*,seed=909):
    """Real producer/consumer schemas, all tokens/scores explicitly authored CPU controls."""
    from dongxi_llms.staged_campaign import evaluation_contracts
    from dongxi_llms.story_rubric import prepare_packet,evaluate_ratings,write_bundle,DIMENSIONS,SCHEMA
    publication=root/'experiments/reports/native-story-publication-20261005-01';publication.mkdir(parents=True)
    bundle=publication/'blind-packet-01';evaluation=publication/'ratings-evaluation-01'
    contract=evaluation_contracts(ROOT)['story-publication-v1']
    plans=[dict(checkpoint_id=arm+'-update-000400',update=400,provenance='model-generated',
        checkpoint_files={'explicit-authored-CPU-schema-control':'a'*64},interface_sha256='b'*64,
        generation_source_sha256={'explicit-authored-CPU-schema-control':'c'*64}) for arm in ('control','half-lr')]
    example=json.loads((ROOT/'fixtures/story-rubric/authored-records.jsonl').read_text().splitlines()[0])
    records=[];item=contract['items'][0]
    for plan in plans:
        record=deepcopy(example);record.update(record_id='authored-schema-'+plan['checkpoint_id'],
            contract_sha256=contract['logical_contract_sha256'],checkpoint_id=plan['checkpoint_id'],
            checkpoint_sha256=adapter.canonical_hash(plan),item_id=item['id'],source_group=item['source_group'],
            opening_sha256=hashlib.sha256(item['prompt'].encode()).hexdigest(),decoding_index=0,attempt_index=0,
            text='<|endoftext|>',text_sha256=hashlib.sha256(b'<|endoftext|>').hexdigest(),token_ids=[50256],
            prompt_tokens=15,generated_tokens=1,selected_likelihoods=[-1.],stop_reason='natural-eos',truncated=False,
            error=None,cost=dict(generation_tokens=1,wall_seconds=.01))
        records.append(record)
    raters=[dict(rater_id='codex-blind-reader-'+suffix,provenance='ai',shared_consultation=False,
        independence_declaration='Authored CPU schema declaration only; no actual reviewer evidence.') for suffix in ('a','b')]
    packet,book=prepare_packet(contract,plans,records,raters,seed=seed,
        comparisons=[[plans[0]['checkpoint_id'],plans[1]['checkpoint_id']]])
    save(publication/'contract.json',contract);save(publication/'checkpoints.json',plans)
    (publication/'records.jsonl').write_text(''.join(json.dumps(value)+'\n' for value in records))
    raters_path=root/'experiments/specs/2026-10-05-story-publication-raters.json';save(raters_path,raters)
    write_bundle(bundle,{'packet.json':packet,'private-codebook.json':book},input_paths=[
        publication/'contract.json',publication/'checkpoints.json',publication/'records.jsonl',raters_path])
    ratings=[]
    for rater,name in zip(raters,adapter.STORY_RATING_FILES):
        document=dict(schema_version=SCHEMA,packet_sha256=packet['packet_sha256'],rubric_sha256=book['rubric_sha256'],
            rater_id=rater['rater_id'],ratings=[dict(candidate_id=row['candidate_id'],text_sha256=row['text_sha256'],
                scores={dimension:1 for dimension in DIMENSIONS},abstention_reason=None,
                note='Explicit authored numeric CPU schema control, not actual AI or human assessment.') for row in packet['candidates']])
        save(bundle/name,document);ratings.append(document)
    report=evaluate_ratings(packet,book,ratings,draws=800,seed=1010)
    write_bundle(evaluation,{'report.json':report},input_paths=[bundle/'packet.json',bundle/'private-codebook.json',
        *(bundle/name for name in adapter.STORY_RATING_FILES)])
    saved=adapter.SavedEvidence();observations={}
    for identifier,path in (('story-publication',publication),('story-blind-packet',bundle),('story-ratings',evaluation)):
        observations[identifier]=adapter.collect_entry(entry(path,identifier=identifier),saved)
    saved.bind(raters_path)
    return publication,report,observations,saved


class NativeCampaignSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive=json.loads((ROOT/adapter.ARCHIVE).read_text())
        cls.rows=cls.archive['campaign']['stage_rows']

    def test_import_has_no_model_gpu_or_network_dependency(self):
        code="import sys,runpy;sys.modules['torch']=None;sys.modules['transformers']=None;runpy.run_path(sys.argv[1],run_name='not_main')"
        result=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'scripts/assemble_native_campaign_evidence.py')],
            capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_all45_original_rows_and_optional_dependencies_stay_unchanged(self):
        original=deepcopy(self.rows)
        rows,changes=adapter.original_rows(self.archive,adapter.stage_rows())
        self.assertEqual(rows,original);self.assertEqual(self.rows,original)
        self.assertEqual(adapter.canonical_hash(rows),adapter.ARCHIVED_ROWS_SHA256)
        self.assertEqual(len(rows),45);self.assertEqual(sum(not row['optional'] for row in rows),41)
        self.assertEqual({row['id'] for row in rows if row['optional']},set(adapter.OPTIONAL_IDS))
        self.assertTrue(all(all(value is None for value in row['actual'].values()) for row in rows))
        self.assertTrue(all('execution_dependency_scopes' not in row for row in rows))
        self.assertTrue(all(change['execution_dependency_scopes'] is not None for change in changes))

    def test_changed_archive_budget_dependency_intervention_or_gate_refuses(self):
        changed=deepcopy(self.archive);changed['campaign']['stage_rows'][0]['proposed_budget']['maximum_updates']=41
        with self.assertRaisesRegex(ValueError,'archived45'):adapter.original_rows(changed,adapter.stage_rows())
        for key,value in (('dependencies',[]),('intervention',{'seed':1}),('gates',[]),('optional',True)):
            current=adapter.stage_rows();current[0][key]=value
            with self.assertRaisesRegex(ValueError,'criteria changed'):adapter.original_rows(self.archive,current)

    def test_missing_future_and_optional_rows_are_null_not_full_completion(self):
        view=adapter.stage_observations(self.rows,{}, {})
        self.assertEqual(set(view),{row['id'] for row in self.rows})
        self.assertTrue(all(value is None for value in view.values()))
        for key in adapter.OPTIONAL_IDS:self.assertIsNone(view[key])

    def test_first400_receipt_does_not_complete_the14000_horizon(self):
        root=Path('/authored/story-control');observation=entry(root,stages=('story-control-pilot',),kind='story-first-tranche')
        docs={str(root/'acceptance.json'):dict(status='passed',completion_level='first400-tranche-not-full14000',full_schedule_complete=False)}
        view=adapter.stage_observations(self.rows,{'control':dict(observation,receipt_paths=list(docs))},docs)
        actual=view['story-control-pilot']
        self.assertEqual(actual['actual_scope'],'first400-tranche');self.assertEqual(actual['original_maximum_updates'],14000)
        self.assertFalse(actual['reported_full_schedule_complete']);self.assertIsNone(actual['full14000_schedule_completion'])
        self.assertEqual(actual['later_checkpoint_observations'],{'4000':None,'8000':None,'14000':None})
        self.assertIsNone(view['story-half-lr-pilot'])
        self.assertTrue(all(value is None for value in self.rows[3]['actual'].values()))

    def test_shared_smoke_recovery_receipt_counts_one_child_not_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'report';terminal=root/'supervision-source/result.json'
            save(terminal,supervisor());save(root/'returned-supervision-source.json',supervisor())
            save(root/'launch-source.json',dict(argv=['/authored/inert/python']))
            save(root/'acceptance.json',dict(status='passed',invocations=[supervisor()]))
            saved=adapter.SavedEvidence();observed=adapter.collect_entry(entry(root),saved)
            aliases={'smoke':observed,'recovery':deepcopy(observed)}
            counts=adapter.child_accounting(aliases,saved.documents)
            self.assertEqual(counts['distinct_native_model_children'],1)
            self.assertEqual(counts['outer_adapter_count'],1)
            self.assertEqual(len(counts['native_model_children'][0]['stage_observation_ids']),2)
            self.assertEqual(saved.bytes_hashed,sum(value['bytes'] for value in saved.bindings.values()))

    def test_failed_native_children_and_cpu_or_outer_processes_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'report'
            save(root/'supervision-1/result.json',supervisor(code=1,pid=18))
            save(root/'supervision-2/result.json',supervisor(native=False,script='cpu_fixture.py',pid=19))
            save(root/'cpu-comparison-process.json',dict(exit=1,CUDA_VISIBLE_DEVICES=''))
            save(root/'failure.json',dict(status='failed',message='authored numerical mismatch'))
            saved=adapter.SavedEvidence();observed=adapter.collect_entry(entry(root),saved)
            counts=adapter.child_accounting({'recovery':observed},saved.documents)
            self.assertEqual(counts['distinct_native_model_children'],1)
            self.assertEqual(counts['native_child_statuses'],{'failed':1})
            self.assertEqual(counts['distinct_cpu_comparison_receipts'],1)
            self.assertEqual(len(counts['unclassified_or_non_native_supervisors']),1)
            self.assertIn(str(root/'failure.json'),observed['failure_paths'])

    def test_returned_logging_failure_dominates_provisional_completed_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'report';provisional=supervisor()
            provisional.update(journal_error=None,final_record_retained=True)
            returned=dict(provisional,status='failed',final_record_retained=False,
                journal_error=dict(type='TimeoutError',message='Final logging timed out'))
            save(root/'supervision-1/result.json',provisional)
            save(root/'returned-supervision-1.json',returned)
            saved=adapter.SavedEvidence();observed=adapter.collect_entry(entry(root),saved)
            count=adapter.child_accounting({'pilot':observed},saved.documents)
            self.assertEqual(count['distinct_native_model_children'],1)
            self.assertEqual(count['native_child_statuses'],{'failed':1})
            self.assertEqual(count['native_model_children'][0]['actual_exit_code'],0)
            self.assertEqual(count['native_model_children'][0]['journal_error'],returned['journal_error'])
            self.assertFalse(count['native_model_children'][0]['final_record_retained'])
            self.assertEqual(observed['terminal_supervisor_paths'],[str(root/'returned-supervision-1.json')])
            self.assertIn(str(root/'supervision-1/result.json'),observed['receipt_paths'])

    def test_native_flag_without_actual_pid_or_known_command_does_not_count(self):
        self.assertFalse(adapter.native_child(supervisor(pid=None)))
        self.assertFalse(adapter.native_child(supervisor(script='authored_fixture.py')))
        self.assertFalse(adapter.native_child(supervisor(native=False)))
        self.assertTrue(adapter.native_child(supervisor(code=-15)))
        comparison=supervisor(script='run_native_chosen_stages.py');comparison['child_command'].append('--comparison-child')
        self.assertFalse(adapter.native_child(comparison))
        legacy=supervisor(script=str(ROOT/'experiments/reports/2026-10-05-native-lora-merge-reload/merge-child.py'))
        self.assertTrue(adapter.native_child(legacy))

    def test_native_watchdog_cpu_comparator_is_not_a_model_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'report';comparison=supervisor(script='run_native_dpo_cpu8_child.py')
            comparison['child_command'].append('--comparison-child');save(root/'supervision-comparison/result.json',comparison)
            saved=adapter.SavedEvidence();observed=adapter.collect_entry(entry(root),saved)
            count=adapter.child_accounting({'comparison':observed},saved.documents)
            self.assertEqual(count['distinct_native_model_children'],0)
            self.assertEqual(count['distinct_cpu_comparison_receipts'],1)
            self.assertEqual(len(count['unclassified_or_non_native_supervisors']),1)

    def test_actual_failed_native_null_exit_retains_cleanup_failure_not_outer_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'report'
            failed=supervisor(script='run_chapter11_spark_dpo.py',pid=275819)
            failed.update(status='failed',actual_exit_code=None,child_seconds=600.6833,seconds=616.25,
                cleanup_errors=['Owned leader reap timed out'],stop_reason='external-deadline',outer_wrapper_exit=1)
            save(root/'supervision-resumed/result.json',failed)
            refused=supervisor(script='run_chapter11_spark_dpo.py',pid=None)
            refused.update(status='failed',actual_exit_code=None,actual_native_profile_executed=False)
            save(root/'supervision-refused/result.json',refused)
            save(root/'failure.json',dict(status='failed',outer_wrapper_exit=1))
            saved=adapter.SavedEvidence();observed=adapter.collect_entry(entry(root),saved)
            count=adapter.child_accounting({'dpo01':observed},saved.documents)
            self.assertEqual(count['distinct_native_model_children'],1)
            self.assertEqual(count['native_child_statuses'],{'failed':1})
            actual=count['native_model_children'][0]
            self.assertIsNone(actual['actual_exit_code']);self.assertEqual(actual['actual_exit_observation'],'unknown-not-observed')
            self.assertEqual(actual['child_pid'],275819);self.assertEqual(actual['child_seconds'],600.6833)
            self.assertEqual(actual['seconds'],616.25);self.assertEqual(actual['stop_reason'],'external-deadline')
            self.assertEqual(actual['cleanup_errors'],['Owned leader reap timed out'])
            self.assertEqual(len(count['unclassified_or_non_native_supervisors']),1)
            self.assertFalse(adapter.native_child(refused));self.assertTrue(adapter.native_child(failed))
            inconsistent=dict(failed,status='completed')
            self.assertFalse(adapter.native_child(inconsistent))
            self.assertEqual(saved.documents[str(root/'failure.json')]['outer_wrapper_exit'],1)

    def test_empty_failed_journal_and_malformed_failure_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);empty=root/'responses.jsonl';empty.touch();bad=root/'failure.json';bad.write_bytes(b'{"partial":')
            saved=adapter.SavedEvidence();binding=saved.bind(empty);value=saved.read(bad)
            self.assertEqual(binding['bytes'],0);self.assertIn('retained_json_error',value)
            self.assertEqual(saved.bindings[str(bad)]['sha256'],hashlib.sha256(bad.read_bytes()).hexdigest())
            saved.verify()

    def test_duplicate_nonfinite_or_symlinked_evidence_is_not_interpreted(self):
        for raw in (b'{"a":1,"a":2}',b'{"cost":NaN}'):
            with self.assertRaises(ValueError):adapter.parse_json(raw)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);real=root/'actual';real.write_bytes(b'actual');link=root/'link';link.symlink_to(real)
            with self.assertRaises(OSError):adapter.stream_binding(link)

    def test_missing_and_existing_receipt_drift_breaks_closing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'receipt.json';save(path,{'version':1})
            saved=adapter.SavedEvidence();saved.read(path);save(path,{'version':2})
            with self.assertRaisesRegex(ValueError,'changed'):saved.verify()
            saved=adapter.SavedEvidence();missing=root/'missing.json';self.assertIsNone(saved.read(missing));save(missing,{})
            with self.assertRaisesRegex(ValueError,'appeared'):saved.verify()

    def test_new_directory_file_or_orphan_size_is_visible_at_closing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'orphan.pt').write_bytes(b'authored incomplete tensor bytes')
            saved=adapter.SavedEvidence();layout=saved.directory(root)
            self.assertEqual(layout[0]['bytes'],len(b'authored incomplete tensor bytes'))
            self.assertEqual(saved.bytes_hashed,0)
            (root/'orphan.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'inventory changed'):saved.verify()

    def test_nested_preference_panels_raw_events_identity_and_summary_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/'report';output=root/'outputs/preference'
            save(report/'comparison.json',dict(status='passed',scope='authored CPU control'))
            for role in ('unchanged','chosen100','dpo100'):
                for panel in ('location','assistant','reasoning'):
                    folder=output/role/panel
                    save(folder/'input-identity.json',dict(authored=True,role=role,panel=panel))
                    save(folder/'summary.json',dict(status='completed',authored=True))
                    (folder/'responses.jsonl').write_bytes(b'{"authored":"raw1"}\n')
                    (folder/'events.jsonl').write_bytes(b'{"authored":"cost"}\n')
                    (folder/'do-not-load.pt').write_bytes(b'authored inert tensor-name fixture')
            value=entry(report,identifier='preference-comparison',stages=('assistant-preference-comparison',),kind='evaluation')
            value['output_roots']=[str(output)];saved=adapter.SavedEvidence()
            observed=adapter.collect_entry(value,saved);saved.verify()
            for role in ('unchanged','chosen100','dpo100'):
                for panel in ('location','assistant','reasoning'):
                    folder=output/role/panel
                    for name in ('input-identity.json','summary.json','responses.jsonl','events.jsonl'):
                        self.assertIn(str(folder/name),saved.bindings)
                    self.assertIn(str(folder/'input-identity.json'),observed['receipt_paths'])
                    self.assertNotIn(str(folder/'do-not-load.pt'),saved.bindings)
            changed=output/'dpo100/assistant/responses.jsonl'
            changed.write_bytes(b'{"authored":"raw2"}\n')
            with self.assertRaisesRegex(ValueError,'changed'):saved.verify()

    def test_nested_preference_missing_panel_remains_null_and_cannot_appear_at_closing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);report=root/'report';output=root/'outputs/preference';output.mkdir(parents=True)
            save(report/'partial-comparison.json',dict(scope='authored partial CPU control'))
            value=entry(report,identifier='preference-comparison');value['output_roots']=[str(output)]
            saved=adapter.SavedEvidence();adapter.collect_entry(value,saved)
            missing=output/'chosen100/reasoning';self.assertIn(str(missing),saved.missing)
            saved.verify();missing.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError,'appeared|inventory changed'):saved.verify()
            saved=adapter.SavedEvidence();saved.directory(root);save(root/'new-outcome.json',{})
            with self.assertRaisesRegex(ValueError,'inventory changed'):saved.verify()

    def test_legitimate_pinned_cache_link_binds_actual_blob_and_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'hub';model=cache/'models--Authored--Inert';root=model/'snapshots/pinned';root.mkdir(parents=True)
            blob=model/'blobs'/'a';blob.parent.mkdir();blob.write_bytes(b'authored inert cache bytes')
            path=root/'model.safetensors';path.symlink_to('../../blobs/a')
            files={path.name:hashlib.sha256(blob.read_bytes()).hexdigest()};saved=adapter.SavedEvidence()
            with patch.object(adapter,'CACHE_ROOT',cache):
                inventories=adapter.bind_inventories({'cache-control':dict(path=str(root),files=files)},saved)
                self.assertEqual(inventories['cache-control']['actual_file_bindings'][path.name]['actual_path'],str(blob))
                saved.verify();path.unlink();path.symlink_to('../../blobs/not-the-saved-blob')
                with self.assertRaises(ValueError):saved.verify()

    def test_cache_link_cannot_escape_its_pinned_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'hub';root=cache/'model/snapshots/pinned';root.mkdir(parents=True)
            outside=Path(tmp)/'outside';outside.write_bytes(b'authored')
            link=root/'model.safetensors';link.symlink_to(outside)
            with patch.object(adapter,'CACHE_ROOT',cache):
                with self.assertRaisesRegex(ValueError,'model blobs'):adapter.SavedEvidence().artifact(link,root)

    def test_final_small_export_is_streamed_byte_bound_without_deserialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policy=root/'outputs/accepted/policy';policy.mkdir(parents=True)
            (policy/'model.safetensors').write_bytes(b'authored inert bytes, not a tensor archive')
            save(policy/'config.json',{'authored':True});save(policy/'course-genealogy.json',{'kind':'authored CPU control'})
            files={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in policy.iterdir()}
            value=dict(path=str(policy),files=files,acceptance_receipt_path='/authored/acceptance.json')
            saved=adapter.SavedEvidence()
            with patch.object(adapter,'ROOT',root):inventories=adapter.bind_inventories({'test-export':value},saved)
            self.assertEqual(inventories['test-export']['files'],files)
            self.assertEqual(inventories['test-export']['genealogy'],{'kind':'authored CPU control'})
            saved.verify()
            (policy/'model.safetensors').write_bytes(b'authored changed bytes')
            with self.assertRaisesRegex(ValueError,'changed'):saved.verify()

    def test_conflicting_export_inventory_or_arbitrary_path_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);policy=root/'outputs/policy';policy.mkdir(parents=True)
            (policy/'model.safetensors').write_bytes(b'authored')
            files={'model.safetensors':hashlib.sha256(b'authored').hexdigest()}
            value=dict(path=str(policy),files=files)
            with patch.object(adapter,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'conflict'):adapter.bind_inventories({'one':value,
                    'two':dict(value,files={'model.safetensors':'f'*64})},adapter.SavedEvidence())
                with self.assertRaisesRegex(ValueError,'roots'):adapter.bind_inventories({'one':dict(value,path='/arbitrary/private')},adapter.SavedEvidence())

    def test_exact_assistant_and_instruct_genealogy_branches_are_separate(self):
        base=dict(path='/authored/Base',files={'base':'a'*64})
        instruct=dict(path='/authored/Instruct',files={'instruct':'b'*64})
        full=dict(path='/authored/full400',files={'full':'c'*64},genealogy={'base_checkpoint_files':base['files']})
        lora=dict(path='/authored/adapter400',files={'adapter':'d'*64},genealogy={'base_checkpoint_files':base['files']})
        inventories={'base-parent':base,'instruct-parent':instruct,'assistant-full-pilot':full,'assistant-lora-pilot':lora,
            'assistant-lora-merge':dict(path='/authored/merge',genealogy={'parent_adapter':lora['path'],'adapter_files':lora['files']}),
            'dpo-pilot':dict(path='/authored/dpo100',genealogy={'parent_checkpoint':full['path'],'parent_checkpoint_sha256':full['files']}),
            'chosen-pilot':dict(path='/authored/chosen100',genealogy={'parent_checkpoint':full['path'],'parent_checkpoint_sha256':full['files']}),
            'rlvr-g4-pilot':dict(path='/authored/g4',genealogy={'parent_local_source_hashes':instruct['files']}),
            'rlvr-g8-pilot':dict(path='/authored/g8',genealogy={'parent_local_source_hashes':instruct['files']})}
        edges=adapter.genealogy_edges(inventories)
        self.assertEqual(len(edges),7);self.assertTrue(all(edge['exact_recorded_parent_inventory_join'] for edge in edges))
        inventories['rlvr-g8-pilot']['genealogy']['parent_local_source_hashes']=base['files']
        self.assertFalse(adapter.genealogy_edges(inventories)[-1]['exact_recorded_parent_inventory_join'])

    def test_actual_story_checkpoint_needs_independent_receipt_and_keeps_later_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);payload=root/'outputs/native-story-control-pilot-20261005-01/pilot/update-000400.pt'
            payload.parent.mkdir(parents=True);payload.write_bytes(b'authored inert story checkpoint bytes')
            receipt=Path(str(payload)+'.work.json');save(receipt,dict(completed_updates=400,
                payload_bytes=payload.stat().st_size,payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest()))
            saved=adapter.SavedEvidence()
            with patch.object(adapter,'ROOT',root):result=adapter.bind_story_checkpoints(saved)
            self.assertEqual(len(result),10);self.assertIsNotNone(result['control-update-000400'])
            self.assertIsNone(result['control-update-014000']);self.assertIsNone(result['half-lr-update-000400'])
            payload.write_bytes(b'changed')
            with patch.object(adapter,'ROOT',root):
                with self.assertRaisesRegex(ValueError,'differs'):adapter.bind_story_checkpoints(adapter.SavedEvidence())

    def test_only_actual_completed_rating_receipt_is_a_scientific_slice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publication,report,observed,saved=story_rating_bundle_fixture(root)
            report_path=str(publication/'ratings-evaluation-01/report.json')
            with patch.object(adapter,'ROOT',root):
                self.assertEqual(adapter.completed_story_ratings(observed,saved.documents,saved.bindings),report)
                for missing in ('report.json','receipt.json'):
                    partial=dict(saved.documents);partial.pop(str(publication/'ratings-evaluation-01'/missing))
                    self.assertIsNone(adapter.completed_story_ratings(observed,partial,saved.bindings))
                authored=deepcopy(saved.documents);authored[report_path]['provenance']='authored-control'
                self.assertIsNone(adapter.completed_story_ratings(observed,authored,saved.bindings))
                changed=dict(saved.bindings);changed[report_path]=dict(sha256='b'*64)
                with self.assertRaisesRegex(ValueError,'completed offline receipt'):
                    adapter.completed_story_ratings(observed,saved.documents,changed)

    def test_other_packet_report_or_unjoined_rating_file_cannot_be_promoted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publication,report,observed,saved=story_rating_bundle_fixture(root)
            other_publication,other_report,other_observed,other_saved=story_rating_bundle_fixture(root/'other',seed=1909)
            report_path=str(publication/'ratings-evaluation-01/report.json')
            receipt_path=str(publication/'ratings-evaluation-01/receipt.json')
            with patch.object(adapter,'ROOT',root):
                self.assertNotEqual(report['packet_sha256'],other_report['packet_sha256'])
                # A physically copied, valid other-packet report plus matching output-byte receipt.
                save(Path(report_path),other_report)
                receipt=deepcopy(saved.documents[receipt_path])
                receipt['artifact_sha256']['report.json']=hashlib.sha256(Path(report_path).read_bytes()).hexdigest()
                save(Path(receipt_path),receipt)
                changed_saved=adapter.SavedEvidence();changed_saved.read(report_path);changed_saved.read(receipt_path)
                wrong=dict(saved.documents,**changed_saved.documents);wrong_bindings=dict(saved.bindings,**changed_saved.bindings)
                with self.assertRaisesRegex(ValueError,'another packet'):
                    adapter.completed_story_ratings(observed,wrong,wrong_bindings)
                changed=deepcopy(saved.documents)
                changed[receipt_path]['input_sha256'][str(publication/'blind-packet-01'/adapter.STORY_RATING_FILES[0])]='f'*64
                with self.assertRaisesRegex(ValueError,'input receipt mismatch'):
                    adapter.completed_story_ratings(observed,changed,saved.bindings)
                missing=dict(saved.bindings);missing.pop(str(publication/'blind-packet-01'/adapter.STORY_RATING_FILES[1]))
                self.assertIsNone(adapter.completed_story_ratings(observed,saved.documents,missing))
                templates=deepcopy(saved.documents);inputs=templates[receipt_path]['input_sha256']
                sha=inputs.pop(str(publication/'blind-packet-01'/adapter.STORY_RATING_FILES[0]))
                inputs[str(publication/'blind-packet-01/ratings-0.json')]=sha
                with self.assertRaisesRegex(ValueError,'empty templates'):
                    adapter.completed_story_ratings(observed,templates,saved.bindings)

    def test_blind_bundle_input_and_artifact_receipts_join_actual_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publication,report,observed,saved=story_rating_bundle_fixture(root)
            receipt_path=str(publication/'blind-packet-01/receipt.json')
            with patch.object(adapter,'ROOT',root):
                for section,key in (('input_sha256',str(publication/'records.jsonl')),('artifact_sha256','packet.json')):
                    changed=deepcopy(saved.documents);changed[receipt_path][section][key]='f'*64
                    with self.assertRaisesRegex(ValueError,'receipt mismatch'):
                        adapter.completed_story_ratings(observed,changed,saved.bindings)

    def test_negative_actual_acceptance_is_not_a_quality_gate(self):
        observation=entry(Path('/authored/preference'),stages=('assistant-preference-comparison',),kind='evaluation')
        document=dict(status='passed',checks={'complete_actual_records':True},negative_difference=-1.)
        docs={'/authored/preference/acceptance.json':document}
        view=adapter.stage_observations(self.rows,{'preference':dict(observation,receipt_paths=list(docs))},docs)
        self.assertEqual(view['assistant-preference-comparison']['retained_outcome_statuses'],{'preference':'passed'})
        self.assertIn('Scoped',view['assistant-preference-comparison']['interpretation'])

    def test_closed_catalog_includes_all_four_interfaces_and_two_caps(self):
        entries=adapter.catalog();index={value['id']:value for value in entries}
        for role in ('base-raw','base-chat','instruct-thinking-off','instruct-thinking-on'):
            for cap in (32,128):self.assertIn('reasoning-'+role+'-cap'+str(cap),index)
        for group in (4,8):
            for cap in (32,128):
                key=f'rlvr-g{group}-evaluation-cap{cap}'
                self.assertIn('run-02',index[key]['path']);self.assertTrue(index[key]['selected'])
                self.assertIn('run-01',index[key+'-accepted-default-unrun']['path'])
                self.assertFalse(index[key+'-accepted-default-unrun']['selected'])
        self.assertNotIn('2026-10-05-story-rubric/packet-01',{Path(value['path']).name for value in entries})
        self.assertTrue(index['assistant-comparison-2']['selected']);self.assertFalse(index['assistant-comparison-1']['selected'])
        self.assertTrue(index['dpo-replay']['selected']);self.assertIn('run-03',index['dpo-replay']['path'])
        for number in (1,2):
            historical=index[f'dpo-replay-historical-{number:02d}']
            self.assertFalse(historical['selected']);self.assertIn(f'run-{number:02d}',historical['path'])
        self.assertIn('run-01',index['dpo-pilot']['path'])

    def test_explicit_g4_retry_preserves_original_failure_and_stage_declarations(self):
        index={value['id']:value for value in adapter.catalog()}
        failed=index['rlvr-g4-recovery'];retry=index['rlvr-g4-recovery-retry02']
        self.assertFalse(failed['selected']);self.assertTrue(retry['selected'])
        self.assertIn('run-01',failed['path']);self.assertIn('run-02',retry['path'])
        self.assertEqual(failed['stage_ids'],retry['stage_ids'])
        self.assertEqual(len(failed['output_roots']),3);self.assertEqual(len(retry['output_roots']),3)
        self.assertTrue(index['rlvr-g8-recovery']['selected'])
        self.assertIn('run-01',index['rlvr-g8-recovery']['path'])

    def test_dpo03_catalog_is_predeclared_not_an_actual_replay_or_new_science(self):
        values=adapter.catalog();index={value['id']:value for value in values}
        self.assertEqual(len(index),len(values))
        chosen=index['dpo-replay']
        for number in (1,2):
            historical=index[f'dpo-replay-historical-{number:02d}']
            self.assertEqual(historical['stage_ids'],chosen['stage_ids'])
            self.assertFalse(historical['selected'])
        self.assertEqual(chosen['stage_ids'],['assistant-dpo-smoke','assistant-dpo-recovery'])
        self.assertEqual(chosen['output_roots'],[str(ROOT/'outputs'/
            ('native-dpo-replay-20261005-run-03-'+role)) for role in ('clean','source','resumed')])
        self.assertIn('Predeclared',chosen['scope']);self.assertIn('default remains serial',chosen['scope'])
        self.assertIn('Execution-only',chosen['scope']);self.assertIn('unchanged600s',chosen['scope'])
        self.assertFalse(any(key in chosen for key in ('acceptance','checks','actual_exit_code','child_pid')))
        self.assertTrue(index['dpo-pilot']['selected']);self.assertIn('run-01',index['dpo-pilot']['path'])

    def test_dpo_failed01_and02_are_retained_counted_without_filling_missing03(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(adapter,'ROOT',root):
                index={value['id']:value for value in adapter.catalog()}
                selected={name:index[name] for name in
                    ('dpo-replay-historical-01','dpo-replay-historical-02','dpo-replay')}
                for number in (1,2):
                    value=selected[f'dpo-replay-historical-{number:02d}'];path=Path(value['path'])
                    save(path/'failure.json',dict(status='failed',outer_wrapper_exit=1,
                        scope='Authored retained historical deadline schema control'))
                    for ordinal,role in enumerate(('clean','source','resumed')):
                        result=supervisor(script='run_native_dpo_cpu8_child.py',pid=4000+number*10+ordinal)
                        if role=='resumed':
                            result.update(status='failed',actual_exit_code=None,child_seconds=600.68,
                                stop_reason='external-deadline',cleanup_errors=['Owned leader reap timed out'])
                        save(path/f'supervision-{role}/result.json',result)
                        save(path/f'returned-supervision-{role}.json',result)
                        save(path/f'launch-{role}.json',dict(argv=result['child_command']))
                    orphan=Path(value['output_roots'][-1])/'orphan.pt'
                    orphan.parent.mkdir(parents=True);orphan.write_bytes(b'authored retained incomplete bytes')
                saved=adapter.SavedEvidence()
                observations={key:adapter.collect_entry(value,saved) for key,value in selected.items()}
                self.assertIsNone(observations['dpo-replay'])
                for number in (1,2):
                    key=f'dpo-replay-historical-{number:02d}';value=observations[key]
                    self.assertIsNotNone(value);self.assertFalse(value['selected'])
                    self.assertIn(str(Path(value['path'])/'failure.json'),value['failure_paths'])
                    orphan=Path(selected[key]['output_roots'][-1])/'orphan.pt'
                    self.assertNotIn(str(orphan),saved.bindings)
                    self.assertTrue(any(item['name']=='orphan.pt' for item in saved.directories[str(orphan.parent)]))
                view=adapter.stage_observations(self.rows,observations,saved.documents)
                self.assertIsNone(view['assistant-dpo-smoke']);self.assertIsNone(view['assistant-dpo-recovery'])
                counted=adapter.child_accounting(observations,saved.documents)
                self.assertEqual(counted['distinct_native_model_children'],6)
                self.assertEqual(counted['native_child_statuses'],{'completed':4,'failed':2})
                failed=[value for value in counted['native_model_children'] if value['status']=='failed']
                self.assertTrue(all(value['actual_exit_code'] is None for value in failed))
                self.assertTrue(all(value['actual_exit_observation']=='unknown-not-observed' for value in failed))
                self.assertTrue(all(value['cleanup_errors']==['Owned leader reap timed out'] for value in failed))
                self.assertIn(selected['dpo-replay']['path'],saved.missing)
                self.assertTrue(all(path in saved.missing for path in selected['dpo-replay']['output_roots']))
                self.assertFalse(any('run-03' in path for path in saved.documents))
                saved.verify()

    def test_actual_runtime_hash_configuration_is_retained_separate_from_historical_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'src/dongxi_llms/artifact_budget.py'
            source.parent.mkdir(parents=True);source.write_bytes(b'# authored current CPU source\n')
            path=root/'experiments/reports/authored-replay';receipt=path/'preparation.json'
            old_source=dict(path=str(source),sha256='a'*64,bytes=31)
            original=dict(source_bindings={'src/dongxi_llms/artifact_budget.py':old_source},
                cpu_settings=dict(omp_num_threads='8',torch_num_threads=8,torch_num_interop_threads=1),
                snapshot_hash_workers=4,science=dict(updates=2,seed=1818,lr=5e-7,accumulation=4,
                    maximum_length=512,beta=.1,generation_cap=64))
            save(receipt,original);saved=adapter.SavedEvidence()
            observed=adapter.collect_entry(entry(path,identifier='authored-runtime'),saved)
            with patch.object(adapter,'ROOT',root):
                retained=adapter.provenance({'authored-runtime':observed},saved.documents,saved)
            record=retained['retained_producer_source_environment_interface_records']['authored-runtime'][str(receipt)]
            self.assertEqual(record['source_bindings'],original['source_bindings'])
            self.assertEqual(record['cpu_settings'],original['cpu_settings'])
            self.assertEqual(record['snapshot_hash_workers'],4);self.assertEqual(record['science'],original['science'])
            current=retained['current_source_environment_file_bindings'][str(source)]
            self.assertEqual(current['sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertNotEqual(current['sha256'],old_source['sha256'])
            self.assertIn('execution configuration',retained['boundary'])
            self.assertEqual(saved.documents[str(receipt)],original);saved.verify()

    def test_actual_scoped_comparison_maps_capstone_without_new_model_or_maxima_claim(self):
        observed={'assistant-comparison-2':entry(Path('/authored/comparison'),stages=(),kind='cpu-comparison')}
        view=adapter.stage_observations(self.rows,observed,{})
        self.assertEqual(view['capstone-branch-comparison']['observation_ids'],['assistant-comparison-2'])
        self.assertIn('story-ratings',view['capstone-branch-comparison']['missing_observation_ids'])
        self.assertIn('no new model child',view['capstone-branch-comparison']['interpretation'])

    def test_authored_complete_small_snapshot_and_failure_prefix_are_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);save(root/adapter.ARCHIVE,self.archive)
            source=root/'authored-source.py';source.write_bytes(b'# authored CPU fixture\n')
            lock=root/'authored.lock';lock.write_bytes(b'authored local lock bytes\n')
            (root/'experiments/reports').mkdir(parents=True,exist_ok=True)
            with ExitStack() as stack:
                for name,value in (('ROOT',root),('LOCK',str(lock)),('SOURCES',('authored-source.py',)),
                        ('DOCUMENTS',(adapter.ARCHIVE,)),('catalog',lambda:[])):
                    stack.enter_context(patch.object(adapter,name,value))
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(adapter.assemble('run-01'),0)
                target=adapter.output_path('run-01');snapshot=json.loads((target/'snapshot.json').read_text())
                self.assertEqual(snapshot['original_archived_stage_rows'],self.rows)
                self.assertEqual(snapshot['child_accounting']['distinct_native_model_children'],0)
                self.assertTrue(all(value is None for value in snapshot['stage_observations'].values()))
                closing=json.loads((target/'closing-bindings.json').read_text())
                self.assertEqual(closing['status'],'unchanged')
                self.assertEqual(closing['snapshot_binding'],adapter.stream_binding(target/'snapshot.json',adapter.SNAPSHOT_BYTES))
                self.assertEqual(closing['snapshot_gzip_binding'],adapter.stream_binding(target/'snapshot.json.gz',adapter.SNAPSHOT_BYTES))
                self.assertEqual(gzip.decompress((target/'snapshot.json.gz').read_bytes()),(target/'snapshot.json').read_bytes())
                self.assertEqual(closing['archive_output_limits']['decompressed_bytes'],1024**3)
                self.assertFalse(closing['compression']['automatic_extraction'])
                self.assertEqual(closing['compression']['original_filename'],'')
                self.assertEqual(closing['compression']['mtime'],0)
                self.assertFalse((target/'failure.json').exists())
                with patch.object(adapter,'bind_story_checkpoints',side_effect=ValueError('authored failure')):
                    with self.assertRaisesRegex(ValueError,'authored failure'):adapter.assemble('run-02')
                self.assertTrue((adapter.output_path('run-02')/'failure.json').exists())
                self.assertFalse((adapter.output_path('run-02')/'snapshot.json').exists())
                original_compress=adapter.compress_snapshot
                def failed_read(*args,**kwargs):
                    with patch.object(adapter.os,'read',side_effect=OSError('authored compression read fault')):
                        return original_compress(*args,**kwargs)
                with patch.object(adapter,'compress_snapshot',side_effect=failed_read):
                    with self.assertRaisesRegex(OSError,'authored compression read fault'):adapter.assemble('run-03')
                failed=adapter.output_path('run-03')
                self.assertEqual(json.loads((failed/'snapshot.json').read_text())['original_archived_stage_rows'],self.rows)
                self.assertTrue((failed/'snapshot.json.gz').exists());self.assertTrue((failed/'failure.json').exists())
                self.assertFalse((failed/'closing-bindings.json').exists())

    def test_cli_is_exclusive_closed_and_never_overwrites_previous_snapshot(self):
        for run in ('../escape','latest','run-1','run-001'):
            with self.assertRaises(ValueError):adapter.output_path(run)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'experiments/reports/native-campaign-evidence-20261005-run-01';target.mkdir(parents=True)
            original=target/'retained';original.write_bytes(b'preserve')
            with patch.object(adapter,'ROOT',root):
                with self.assertRaises(FileExistsError):adapter.assemble('run-01')
            self.assertEqual(original.read_bytes(),b'preserve')
        with patch.object(adapter,'assemble',return_value=0) as assemble:
            self.assertEqual(adapter.main(['--run-id','run-02']),0);assemble.assert_called_once_with('run-02')


class NativeCampaignArchiveOutput(unittest.TestCase):
    """Small authored byte controls, not an actual campaign serialization."""
    def test_archive_output_ceiling_is_separate_from_unchanged_inputs(self):
        self.assertEqual(adapter.JSON_BYTES,64*1024**2)
        self.assertEqual(adapter.JOURNAL_BYTES,256*1024**2)
        self.assertEqual(adapter.EXPORT_FILE_BYTES,4*1024**3)
        self.assertEqual(adapter.SNAPSHOT_BYTES,1024**3)
        self.assertEqual(adapter.MAX_FILES,6000)

    def test_streamed_snapshot_matches_original_json_bytes(self):
        value={'fixture':'authored \u6570\u636e','rows':[{'kept':True},None,0]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.json';adapter.retain_snapshot(path,value)
            expected=(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
            self.assertEqual(path.read_bytes(),expected)

    def test_snapshot_ceiling_is_admitted_before_write_and_includes_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'snapshot.json'
            with self.assertRaisesRegex(ValueError,'Archive-output ceiling crossed'):
                adapter.retain_snapshot(path,{'kept':'x'*100},maximum=32)
            prefix=path.read_bytes();self.assertGreater(len(prefix),0);self.assertLessEqual(len(prefix),32)
            with self.assertRaises(FileExistsError):adapter.retain_snapshot(path,{})
            self.assertEqual(path.read_bytes(),prefix)
            newline=root/'newline.json'
            with self.assertRaisesRegex(ValueError,'Archive-output ceiling crossed'):
                adapter.retain_snapshot(newline,{},maximum=2)
            self.assertEqual(newline.read_bytes(),b'{}')

    def test_invalid_archive_limits_reject_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'authored')
            for maximum in (False,True,0,-1,1.,'1024',adapter.SNAPSHOT_BYTES+1):
                with self.subTest(maximum=maximum):
                    raw=root/'raw';compressed=root/'compressed'
                    with self.assertRaisesRegex(ValueError,'archive-output ceiling'):
                        adapter.retain_snapshot(raw,{},maximum=maximum)
                    with self.assertRaisesRegex(ValueError,'archive-output ceiling'):
                        adapter.compress_snapshot(source,compressed,maximum=maximum)
                    self.assertFalse(raw.exists());self.assertFalse(compressed.exists())

    def test_nonfinite_json_preserves_failed_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'snapshot.json'
            with self.assertRaises(ValueError):adapter.retain_snapshot(path,{'kept':1,'invalid':float('nan')})
            self.assertTrue(path.exists());self.assertIn(b'"kept": 1',path.read_bytes())
            self.assertNotIn(b'NaN',path.read_bytes())

    def test_gzip_is_same_environment_deterministic_with_no_filename_or_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'snapshot.json'
            adapter.retain_snapshot(source,{'kept':'authored \u6570\u636e','failures':[None,{'exit':-15}]})
            a=root/'a.gz';b=root/'different-original-name.gz'
            adapter.compress_snapshot(source,a);adapter.compress_snapshot(source,b)
            encoded=a.read_bytes();self.assertEqual(encoded,b.read_bytes())
            self.assertEqual(encoded[:3],b'\x1f\x8b\x08');self.assertEqual(encoded[3],0)
            self.assertEqual(encoded[4:8],b'\x00'*4)
            self.assertEqual(gzip.decompress(encoded),source.read_bytes())

    def test_gzip_rejects_raw_cap_symlink_directory_and_fifo_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'authored')
            linked=root/'link';linked.symlink_to(source);fifo=root/'fifo';os.mkfifo(fifo)
            for value,maximum in ((source,3),(linked,64),(root,64),(fifo,64)):
                with self.subTest(source=value):
                    output=root/'out.gz'
                    with self.assertRaises((OSError,ValueError)):
                        adapter.compress_snapshot(value,output,maximum=maximum)
                    self.assertFalse(output.exists())

    def test_gzip_compressed_cap_preserves_bounded_partial_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'abcdefghijklmnopqrst')
            output=root/'out.gz'
            with self.assertRaisesRegex(ValueError,'Archive-output ceiling crossed'):
                adapter.compress_snapshot(source,output,maximum=20)
            self.assertTrue(output.exists());self.assertGreater(output.stat().st_size,0)
            self.assertLessEqual(output.stat().st_size,20)
            self.assertEqual(source.read_bytes(),b'abcdefghijklmnopqrst')

    def test_gzip_never_overwrites_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'authored')
            output=root/'out.gz';output.write_bytes(b'preserve prior gzip/failure bytes')
            with self.assertRaises(FileExistsError):adapter.compress_snapshot(source,output)
            self.assertEqual(output.read_bytes(),b'preserve prior gzip/failure bytes')

    def test_gzip_refuses_source_mutation_during_streaming(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'authored')
            output=root/'out.gz';read=adapter.os.read;changed=False
            def mutating_read(fd,size):
                nonlocal changed
                value=read(fd,size)
                if value and not changed:
                    changed=True
                    with source.open('ab') as handle:handle.write(b'x')
                return value
            with patch.object(adapter.os,'read',side_effect=mutating_read):
                with self.assertRaisesRegex(ValueError,'Snapshot bytes/path changed'):
                    adapter.compress_snapshot(source,output)
            self.assertTrue(output.exists());self.assertEqual(source.read_bytes(),b'authoredx')

    def test_gzip_refuses_same_bytes_rebound_to_another_source_inode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.write_bytes(b'authored')
            output=root/'out.gz';read=adapter.os.read;changed=False
            def replacing_read(fd,size):
                nonlocal changed
                value=read(fd,size)
                if value and not changed:
                    changed=True;source.rename(root/'retained-original');source.write_bytes(b'authored')
                return value
            with patch.object(adapter.os,'read',side_effect=replacing_read):
                with self.assertRaisesRegex(ValueError,'Snapshot bytes/path changed'):
                    adapter.compress_snapshot(source,output)
            self.assertTrue(output.exists());self.assertEqual((root/'retained-original').read_bytes(),b'authored')

    def test_only_exact_uncompressed_campaign_snapshot_ignore_rule_is_added(self):
        lines=(ROOT/'.gitignore').read_text().splitlines()
        self.assertEqual(lines.count('/experiments/reports/native-campaign-evidence-*/snapshot.json'),1)
        self.assertNotIn('/experiments/reports/native-campaign-evidence-*',lines)
        self.assertNotIn('*.gz',lines)


if __name__=='__main__':unittest.main()
