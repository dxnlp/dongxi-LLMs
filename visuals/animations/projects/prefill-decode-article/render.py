"""Reuse the verified article exporter; validate our own mechanisms and metadata."""
import importlib.util
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[3]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


export=load('article_export',P.parent/'kv-article-loops/render.py')
fixture_module=load('pd_checks',ROOT/'publications/x-articles/x-prefill-decode-001/check.py')
CLIPS=[('01-lifecycle','Lifecycle'),('02-parallel','ParallelPositions'),
       ('03-first-token','FirstToken'),('04-decode-loop','DecodeLoop'),
       ('05-workloads','Workloads'),('06-scheduling','Scheduling'),('07-handoff','Handoff')]


def validate(stem,timeline):
    assert all(timeline['checks'].values())
    e={v['name']:v for v in timeline['events']}
    assert 'complete' in e and 5<timeline['duration_seconds']<35
    if stem=='01-lifecycle':
        assert e['first_output']['prompt_cache']==4
        assert e['prompt_kv_ready']['seconds']<e['first_output']['seconds']<e['output_2']['seconds']
    elif stem=='02-parallel':
        assert e['causal_reads']['reads']==[[1],[1,2]] and not e['causal_reads']['uses_O1']
        assert e['qkv_ready']['seconds']<e['outputs_together']['seconds']
        assert not e['outputs_together']['same_layer_output_dependency']
    elif stem=='03-first-token':
        assert [e[n]['cache'] for n in ['y1_selected','y1_kv_appended','y2_selected']]==[4,5,5]
    elif stem=='04-decode-loop':
        assert [e[n]['cache'] for n in ['y1_selected','y1_input','y1_layer1','y1_layer2','y2_selected',
                'y2_input','y2_layer1','y2_layer2','y3_selected']]==[[4,4],[4,4],[5,4],[5,5],[5,5],[5,5],[6,5],[6,6],[6,6]]
        assert e['y2_selected']['old_cache_unchanged'] and e['y3_selected']['old_cache_unchanged']
    elif stem=='05-workloads':
        assert e['inputs']['input_lines']==[7,2] and e['outputs']['output_lines']==[3,7]
        assert not e['outputs']['measured']
    elif stem=='06-scheduling':
        assert e['requests']['separate_state']
        assert [e[f'chunk_{i}_retained']['B_chunks'] for i in [1,2,3]]==[1,2,3]
        assert e['B_handoff']['transferred_request']=='B'
    else:
        assert e['transfer_tail']['received']==[1,2] and not e['transfer_tail']['decode_started']
        assert e['L2_ready']['transfer_in_flight']==[1]
        assert e['decode_started']['received']==[1,2,3]
        assert e['all_received']['seconds']>e['L3_ready']['seconds']


if __name__=='__main__':
    fixture=fixture_module.mechanism()
    export.P=P;export.CLIPS=CLIPS;export.validate=validate
    export.main()
    path=P/('preview-metadata.json' if '--preview' in sys.argv else 'metadata.json')
    report=json.loads(path.read_text())
    report['task']='ANIM-PD-001';report['fixture']=fixture
    for p in [P.parent/'kv-article-loops/render.py',ROOT/'publications/x-articles/x-prefill-decode-001/check.py',
              P.parents[1]/'STYLE_GUIDE.md']:
        report['sources'].append({'path':str(p.relative_to(ROOT)),'sha256':export.sha(p)})
    report['limitations']=[
        'Schematic dependencies and token identities, not measured activations, quality or speed.',
        'Fade/reset is editorial, not cache eviction or reverse generation.',
        'Parallel clip shows one attention head; other block operations omitted.',
        'First-token clip aggregates per-layer KV; decode loop separates two illustrative layers.',
        'Decode loop compresses embeddings, normalization, positional handling, residual/MLP and vocabulary selection.',
        'Workload strokes are text-volume symbols, not actual tokens or a timed performance comparison.',
        'Scheduling retains separate A/B state and B chunk dependencies; budgets/placement are conceptual.',
        'Handoff is an EXO-inspired KV overlap schematic, not a local benchmark or turnkey software claim.',
        'Handoff continuation assumes compatible weights, cache, positions and required generation metadata.']
    path.write_text(json.dumps(report,indent=2)+'\n')
