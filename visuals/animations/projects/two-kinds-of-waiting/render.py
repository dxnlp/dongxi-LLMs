"""Standalone export. Hash-protect both the article and its seven existing loops."""
import importlib.util
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
ROOT=P.parents[3]
spec=importlib.util.spec_from_file_location('exporter',P.parent/'kv-article-loops/render.py')
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


def validate(stem,timeline):
    assert timeline['checks']['inside_canvas']
    e={v['name']:v for v in timeline['events']}
    assert 17<timeline['duration_seconds']<25
    assert e['request_sent']['seconds']==e['ttft_start']['seconds']
    assert e['ttft_start']['visible_outputs']==0
    for i in range(4):
        start=e['ttft_start' if i==0 else f'itl_{i}_start']
        end=e[f'output_{i+1}_received']
        assert start['seconds']<end['seconds']
        assert start['end_x']==end['x']
        assert end['visible_outputs']==i+1
        if i:
            previous=e[f'output_{i}_received']
            assert start['seconds']==previous['seconds']
            assert start['start_x']==previous['x']


if __name__=='__main__':
    roots=[ROOT/'publications/x-articles/x-prefill-decode-001',P.parent/'prefill-decode-article']
    protected={p:export.sha(p) for root in roots for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    export.P=P;export.CLIPS=[('two-waits','TwoWaits')];export.validate=validate
    export.main()
    assert all(export.sha(p)==sha for p,sha in protected.items())
    mf=P/('preview-metadata.json' if '--preview' in sys.argv else 'metadata.json')
    report=json.loads(mf.read_text())
    report['task']='ANIM-PD-002'
    report['checks']['article_and_original_animations_unchanged']=True
    report['protected_files']={str(p.relative_to(ROOT)):sha for p,sha in protected.items()}
    for src in [P/'example.json',P.parent/'kv-article-loops/render.py']:
        report['sources'].append({'path':str(src.relative_to(ROOT)),'sha256':export.sha(src)})
    report['limitations']=['Authored project-document example, not a measured model response.',
      'Each readable word card stands in for one token arrival; not actual tokenizer segmentation.',
      'TTFT spans request send to first arrival; each ITL spans adjacent arrivals. Durations are illustrative.',
      'TTFT can include queueing, prefill, sampling and transport; Prefill label is not a latency breakdown.',
      'Ordinary causal autoregressive generation; actual model internals and serving overhead omitted.',
      'Fade is an editorial reset, not cache eviction.']
    mf.write_text(json.dumps(report,indent=2)+'\n')
