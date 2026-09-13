"""Reuse the established media exporter without rerendering existing loops."""
import importlib.util
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent
shared=P.parent/'kv-article-loops/render.py'
spec=importlib.util.spec_from_file_location('loop_export',shared)
export=importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def validate(stem,timeline):
    assert stem=='memory-growth' and all(timeline['checks'].values())
    events=timeline['events']
    assert [e['name'] for e in events]==['baseline','double_positions','comparison_baseline','half_heads']
    assert [(e['positions'],e['heads'],e['mib']) for e in events]==[
        (4096,8,192),(8192,8,384),(4096,8,192),(4096,4,96)]
    for e in events:
        assert e['mib']==2*24*e['positions']*e['heads']*64*2//2**20
        assert e['bar_fraction']==e['mib']/384


if __name__=='__main__':
    protected={p:export.sha(p) for project in ['kv-prefill-decode','kv-article-loops']
               for p in (P.parent/project).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    export.P=P
    export.CLIPS=[('memory-growth','MemoryGrowth')]
    export.validate=validate
    export.main()
    assert all(export.sha(p)==h for p,h in protected.items())
    manifest=P/('preview-metadata.json' if '--preview' in sys.argv else 'metadata.json')
    m=json.loads(manifest.read_text())
    m['checks']['all_existing_loops_unchanged']=True
    m['sources'].append(dict(path=str(shared.relative_to(export.ROOT)),sha256=export.sha(shared)))
    m['limitations']=[
        'Calculated separate-K/V tensor payload, not measured GPU allocations.',
        'Fixed batch 1, 24 layers, head width 64, 2 bytes per element; MiB = 2^20 bytes.',
        'Each column groups 1024 positions; one paired K/V glyph per KV head.',
        'Layer, feature and batch axes are omitted from the schematic, included in the calculation.',
        'Fewer KV heads compares model configurations; it is not runtime head deletion.',
        'Fades between comparisons and at loop end are editorial resets, not cache eviction.',
        'Model weights, temporary buffers, cache metadata and allocator overhead excluded.']
    manifest.write_text(json.dumps(m,indent=2)+'\n')
