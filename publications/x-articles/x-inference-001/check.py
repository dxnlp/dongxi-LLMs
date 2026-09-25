"""Audit article completeness and static media without browser or model execution."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re

from PIL import Image

P=Path(__file__).resolve().parent
ZH=P/'zh'


class AuditHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images=[]
        self.links=[]
        self.headings=0
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='img': self.images.append(attrs)
        if tag=='a': self.links.append(attrs)
        if tag=='h2': self.headings+=1


def main():
    source=(ZH/'x-editor-draft-body-with-image-placeholders.md').read_text()
    plan=json.loads((ZH/'x-editor-clean-body-image-plan.json').read_text())
    meta=json.loads((P/'metadata.json').read_text())
    assert not re.search(r'(不是|并非)[^。！？\n]{0,160}(而是|而在于)',source)
    assert 'http' not in source and '```' not in source
    assert len(re.findall(r'^\[IMAGE \d{2}\]',source,re.M))==7
    assert plan['image_count']==7 and plan['missing_image_count']==0
    assert [i['no'] for i in plan['images']]==list(range(1,8))
    clean=(ZH/'x-editor-clean-body.html').read_text()
    assert '[IMAGE' not in clean and '<a ' not in clean and '<h1' not in clean
    for item in plan['images']:
        assert Path(item['path']).exists()
        assert item['target_after_text_full'] and item['next_text_full']
        assert clean.count(item['target_block_html'])==1
    audit=AuditHTML()
    review=(ZH/'review.html').read_text()
    audit.feed(review)
    assert len(audit.images)==8 and audit.headings==8 and not audit.links
    assert '[IMAGE' not in review and '<script' not in review
    assert 'name="viewport"' in review
    for i,item in enumerate(audit.images):
        assert item.get('alt')
        path=(ZH/item['src']).resolve()
        with Image.open(path) as im:
            assert im.size==((2000,800) if i==0 else (1600,900))
            im.verify()
    for group in ['sources','outputs']:
        for rel,digest in meta[group].items():
            assert hashlib.sha256((P/rel).read_bytes()).hexdigest()==digest,rel
    assert 8_000_000_000*16//8 == meta['calculated_weight_bytes']['16bit']
    assert 8_000_000_000*4//8 == meta['calculated_weight_bytes']['4bit']
    clean_md=(ZH/'x-editor-clean-body.md').read_text()
    paragraphs=[s for s in clean_md.split('\n\n') if s and not s.startswith('#')]
    qa={
        'date':'2026-09-13','status':'local_draft_for_user_review',
        'checks':{'media_dimensions':True,'source_and_output_hashes':True,
                  'calculated_weight_payload':True,'image_order_and_anchors':True,
                  'public_body_link_free':True,'no_prohibited_contrast_pattern':True,
                  'no_placeholders_in_review_or_clean_body':True,'html_has_responsive_css':True},
        'counts':{'inline_static_figures':7,'covers':1,'cover_ratio':'5:2','h2_sections':audit.headings,
                  'clean_body_blocks':plan['total_blocks'],
                  'chinese_characters':len(re.findall(r'[\u4e00-\u9fff]',clean_md)),
                  'max_prose_paragraph_characters':max(len(s) for s in paragraphs if not s.startswith('- '))},
        'visual_inspection':{'contact_sheet':'inspected; fixed columns, readable full-size labels, no observed overlap',
                             'full_size_figures':['03-phases.png','04-speculation.png','06-parallelism.png'],
                             'result':'inspected; no observed clipping or overlap'},
        'browser_review':{'status':'blocked_by_browser_url_security_policy',
                          'attempt':'direct file URL in in-app browser',
                          'workaround_attempted':False,'mobile_layout_verified':False},
        'boundaries':{'new_animation_rendered':False,'model_or_gpu_run':False,
                      'X_draft_url':None,'uploaded':False,'published':False,
                      'existing_KV_article_modified':False}
    }
    (P/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(qa['counts'],ensure_ascii=False))
    print('PASS: structure, text style, image order, payload arithmetic and all media/source hashes.')


if __name__=='__main__':
    main()
