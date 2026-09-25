"""Structural checks plus a tiny causal-attention cache fixture, not a benchmark."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re

import numpy as np
from PIL import Image

P = Path(__file__).resolve().parent
Z = P/'zh'


def mechanism():
    rng=np.random.default_rng(14)
    d,vocab,layers=8,19,2
    embeddings=rng.normal(size=(vocab,d))*0.3
    weights=rng.normal(size=(layers,3,d,d))*0.2
    head=rng.normal(size=(d,vocab))*0.2
    # Evaluate position 2 before position 1 using shared input-derived K/V.
    q,k,v=[embeddings[:2]@w for w in weights[0]]
    scores=q@k.T/np.sqrt(d)
    scores[np.triu_indices(2,1)]=-np.inf
    probs=np.exp(scores-scores.max(axis=1,keepdims=True))
    probs/=probs.sum(axis=1,keepdims=True)
    parallel=probs@v
    reversed_rows=np.empty_like(parallel)
    for row in [1,0]:
        row_scores=q[row]@k[:row+1].T/np.sqrt(d)
        row_probs=np.exp(row_scores-row_scores.max())
        row_probs/=row_probs.sum()
        reversed_rows[row]=row_probs@v[:row+1]
    np.testing.assert_allclose(parallel,reversed_rows,atol=1e-12,rtol=1e-12)

    def forward(ids, old=None, causal=True):
        prefix=0 if old is None else len(old[0][0])
        positions=np.arange(prefix,prefix+len(ids))
        x=embeddings[ids]+0.03*np.sin(positions[:,None]+np.arange(d))
        caches=[]
        for layer,(wq,wk,wv) in enumerate(weights):
            q,k,v=x@wq,x@wk,x@wv
            if old is not None:
                k=np.concatenate([old[layer][0],k])
                v=np.concatenate([old[layer][1],v])
            scores=q@k.T/np.sqrt(d)
            if causal:
                allowed=np.arange(len(k))[None,:]<=positions[:,None]
                scores=np.where(allowed,scores,-np.inf)
            probs=np.exp(scores-scores.max(axis=-1,keepdims=True))
            probs/=probs.sum(axis=-1,keepdims=True)
            x=x+np.tanh(probs@v)
            caches.append((k.copy(),v.copy()))
        return x@head,caches

    prompt=[1,4,7,9]
    logits,cache=forward(prompt)
    snapshot=[(k.copy(),v.copy()) for k,v in cache]
    y1=int(logits[-1].argmax())
    assert all(len(k)==4 for k,v in cache)
    one,next_cache=forward([y1],cache)
    full,full_cache=forward(prompt+[y1])
    errors=[float(np.max(np.abs(one[-1]-full[-1])))]
    np.testing.assert_allclose(one[-1],full[-1],atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(logits,full[:4],atol=1e-12,rtol=1e-12)
    for i in range(layers):
        assert len(next_cache[i][0])==5
        for part in range(2):
            np.testing.assert_array_equal(cache[i][part],snapshot[i][part])
            np.testing.assert_allclose(next_cache[i][part],full_cache[i][part],atol=1e-12,rtol=1e-12)
    # Two-token prefill chunks need the cached-prefix offset in the mask.
    _,first_chunk=forward(prompt[:2])
    second_logits,chunk_cache=forward(prompt[2:],first_chunk)
    np.testing.assert_allclose(second_logits,logits[2:],atol=1e-12,rtol=1e-12)
    for i in range(layers):
        for part in range(2):
            np.testing.assert_allclose(chunk_cache[i][part],cache[i][part],atol=1e-12,rtol=1e-12)
    # Deliberately broken full attention changes old states after an append.
    broken,_=forward(prompt,causal=False)
    broken_more,_=forward(prompt+[y1],causal=False)
    broken_delta=float(np.max(np.abs(broken-broken_more[:4])))
    assert broken_delta>1e-6
    return {'fixture':'two-layer residual causal attention, untrained float64',
            'seed':14,'prefill_entries':4,'entries_after_selection':4,
            'entries_after_y1_forward':5,'cached_vs_full_max_error':max(errors),
            'chunked_prefill_matches':True,'reversed_row_order_matches_batched_attention':True,
            'broken_noncausal_prefix_delta':broken_delta,
            'benchmark':False}


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images=[]
        self.links=[]
        self.h2=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img':
            assert attrs.get('alt')
            self.images.append(attrs['src'])
        if tag=='a':
            self.links.append(attrs.get('href'))
        self.h2+=tag=='h2'


def main():
    source=(Z/'x-editor-draft-body-with-image-placeholders.md').read_text()
    clean=(Z/'x-editor-clean-body.md').read_text()
    metadata=json.loads((P/'metadata.json').read_text())
    animated=metadata['animation_render']
    assert source.startswith('# Prefill and Decode\n')
    assert len(re.findall(r'^## ',source,re.M))==6
    assert re.findall(r'\[IMAGE (\d+)\]',source)==['01','02','03','04','05','06','07']
    assert '[IMAGE' not in clean and not clean.startswith('# ')
    assert not re.search(r'https?://|\]\(',source)
    assert not re.search(r'(?:不是|并非|并不是)[\s\S]{0,180}(?:而是|而在于)',source)
    assert not re.search(r'本地审阅版|ANIM-|excalidraw|manim',source,re.I)
    plan=json.loads((Z/'x-editor-clean-body-image-plan.json').read_text())
    assert plan['image_count']==7 and plan['missing_image_count']==0
    assert plan['images'][1]['path_relative']=='../assets/'+('02-parallel.gif' if animated else '07-parallel-positions.png')
    assert plan['images'][3]['path_relative']=='../assets/'+('04-decode-loop.gif' if animated else '06-decode-loop.png')
    clean_html=(Z/'x-editor-clean-body.html').read_text()
    for item in plan['images']:
        assert Path(item['path']).is_file()
        assert item['target_block_html'] in clean_html
        assert item['caption']==item['caption_in_body']
        if animated:
            with Image.open(item['path']) as gif:
                assert gif.size==(960,540) and gif.n_frames>30 and gif.info['loop']==0
    page=Page()
    page.feed((Z/'review.html').read_text())
    assert page.h2==6 and len(page.images)==8 and not page.links
    for i,src in enumerate(page.images):
        with Image.open(Z/src) as im:
            assert im.size==((2000,800) if i==0 else ((1920,1080) if animated else (1600,900)))
    if animated:
        review=(Z/'review.html').read_text()
        assert review.count('class="motion"')==7 and review.count('aria-controls="motion-')==7
        assert 'prefers-reduced-motion' in review
        root=P.parents[2]
        anim=root/'visuals/animations/projects/prefill-decode-article'
        assert hashlib.sha256((anim/'metadata.json').read_bytes()).hexdigest()==metadata['animation_manifest_sha256']
        manifest=json.loads((anim/'metadata.json').read_text())
        assert len(manifest['clips'])==7
        for item in manifest['sources']:
            assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
        for clip in manifest['clips']:
            for item in clip['outputs']:
                assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256'],item['path']
    for group in ['sources','outputs']:
        for name,sha in metadata[group].items():
            assert hashlib.sha256((P/name).read_bytes()).hexdigest()==sha,name
    generated=[p for p in Z.iterdir() if p.is_file()]
    qa={'status':'pass','sections':6,'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',clean)),
        'clean_body_blocks':plan['total_blocks'],'inline_figures':7,'cover':[2000,800],
        'animated':animated,
        'mechanism':mechanism(),'numpy':np.__version__,
        'browser_visual_qa':'not performed; prior explicit local-URL policy block respected',
        'sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [P/'check.py',*generated]}}
    (P/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps({k:v for k,v in qa.items() if k!='sha256'},indent=2))


if __name__=='__main__':
    main()
