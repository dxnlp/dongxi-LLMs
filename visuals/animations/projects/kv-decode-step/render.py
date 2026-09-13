"""Export a close-up without changing existing article animations."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

P=Path(__file__).resolve().parent
shared=P.parent/'kv-article-loops/render.py'
spec=importlib.util.spec_from_file_location('loop_export',shared)
export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)


def validate(stem,timeline):
    assert stem=='decode-step' and all(timeline['checks'].values())
    e=timeline['events'];f=timeline['fixture']
    assert [v['name'] for v in e]==['new_hidden_state','new_qkv','append_before_read',
        'query_matches_all_keys','softmax_weights','weighted_value_sum','cache_retained']
    assert [v['cached'] for v in e]==[2,2,3,3,3,3,3]
    assert e[2]['self_included'] and e[3]['read_positions']==[1,2,3] and not e[-1]['query_retained']
    scores=np.array(f['q'])@np.array(f['K']).T/np.sqrt(2)
    weights=np.exp(scores-scores.max());weights/=weights.sum()
    np.testing.assert_allclose(weights,f['weights'])
    np.testing.assert_allclose(weights@np.array(f['V']),f['output'])


if __name__=='__main__':
    protected={p:export.sha(p) for name in ['kv-prefill-decode','kv-article-loops','kv-memory-growth']
               for p in (P.parent/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    export.P=P;export.CLIPS=[('decode-step','DecodeStep')];export.validate=validate
    export.main()
    assert all(export.sha(p)==h for p,h in protected.items())
    path=P/('preview-metadata.json' if '--preview' in sys.argv else 'metadata.json')
    m=json.loads(path.read_text());m['checks']['all_existing_loops_unchanged']=True
    for p in [shared,P/'example.py']:
        m['sources'].append(dict(path=str(p.relative_to(export.ROOT)),sha256=export.sha(p)))
    m['limitations']=['One layer/head, one new position; no padding or positional operations.',
        'Explicit toy fixture, not measured model activations; old vectors remain unchanged.',
        'Softmax bar widths and flow strokes use the calculated toy weights.',
        'Head output is a weighted V sum, not logits; multi-head output projection and model remainder omitted.',
        'Cache retention does not require retaining past queries for ordinary decode.',
        'Final fade is an editorial loop reset, not cache eviction.']
    path.write_text(json.dumps(m,indent=2)+'\n')
