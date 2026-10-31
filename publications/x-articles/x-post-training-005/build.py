"""Build HTML bodies and local review pages for X-POST-005 (English root, Chinese zh/).

uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-post-training-005/build.py
Then, for draft transfer preparation:
python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-post-training-005
python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-post-training-005/zh
"""
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import re

import mistune

P = Path(__file__).resolve().parent
ORDER_RE = re.compile(r"^\s*(\d+)\.\s+`([^`]+)`\s+.\s+(.+?)\s*$")
BANNED_ZH = re.compile(r'不是|而是|并非|不在于|与其|不如说')

CSS = '''*{box-sizing:border-box}html{color-scheme:light}
body{margin:0;background:#fff;color:#111827;font-family:Arial,"Helvetica Neue",sans-serif}
body.zh{font-family:"PingFang SC","Songti SC",STSong,serif}
main{max-width:860px;margin:36px auto 110px;padding:0 28px}
nav{display:flex;justify-content:space-between;align-items:center;font-size:14px;color:#64748b;margin-bottom:18px}
nav a{color:#2563eb;text-decoration:none;border:1px solid #cbd5e1;border-radius:6px;padding:5px 10px}
header img{width:100%;height:auto;display:block;border-radius:10px}
h1{font-size:42px;line-height:1.25;font-weight:700;margin:34px 0 30px}
h2{font-size:29px;line-height:1.4;margin:58px 0 22px}
h3{font-size:23px;line-height:1.4;margin:40px 0 16px}
p,li{font-size:19px;line-height:1.85;overflow-wrap:anywhere}
p{margin:0 0 20px} ul{padding-left:24px;margin:0 0 22px} li{margin-bottom:8px}
body.zh p,body.zh li{font-size:19px;line-height:1.95}
blockquote{margin:26px 0;padding:4px 0 4px 20px;border-left:4px solid #cbd5e1;color:#1f2937}
blockquote p{margin:8px 0;font-size:20px}
figure{margin:34px -10px}
figure img{width:100%;height:auto;display:block;border:1px solid #e2e8f0;border-radius:10px}
figcaption{font-size:15px;line-height:1.75;color:#64748b;padding:12px 8px}
.note{font-size:14px;color:#64748b;border-top:1px solid #e2e8f0;margin-top:60px;padding-top:16px}
@media(max-width:600px){main{padding:0 16px;margin-top:16px}h1{font-size:30px}h2{font-size:24px;margin-top:42px}
h3{font-size:20px}p,li,body.zh p,body.zh li{font-size:17px}figure{margin:26px -6px}}'''


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(lang):
    d = P if lang == 'en' else P / 'zh'
    source = (d / 'x-editor-draft-body-with-image-placeholders.md').read_text()
    if lang == 'zh':
        assert not BANNED_ZH.search(source), BANNED_ZH.search(source)
    title_line, body = source.split('\n', 1)
    title = title_line.removeprefix('# ').strip()
    raw = mistune.html(body)
    # The X body-plan builder reads one block per line; keep the placeholder alone in its paragraph.
    (d / 'x-editor-draft-body-with-image-placeholders.html').write_text(raw)
    order = {}
    for line in (d / 'image-upload-order.md').read_text().splitlines():
        m = ORDER_RE.match(line)
        if m:
            order[int(m[1])] = (m[2], m[3])
    placeholders = [int(n) for n in re.findall(r'\[IMAGE (\d\d)\]', source)]
    assert placeholders == list(range(1, len(order) + 1)), (placeholders, sorted(order))
    page_body = raw
    for no, (path, caption) in order.items():
        assert (d / path).is_file(), path
        fig = (f'<figure><img src="{html.escape(path)}" loading="lazy" alt="{html.escape(caption, quote=True)}">'
               f'<figcaption>{html.escape(caption)}</figcaption></figure>')
        page_body, n = re.subn(rf'<p>\[IMAGE {no:02d}\]</p>', lambda _: fig, page_body)
        assert n == 1, no
    assert '[IMAGE' not in page_body
    cover = 'assets/cover-post-training-05.png' if lang == 'en' else '../assets/cover-post-training-05.png'  # one English cover
    assert (d / cover).is_file()
    other = ('zh/review.html', '中文版') if lang == 'en' else ('../review.html', 'English')
    note = ('Local review page. Not uploaded or published. Paper figures are published results; all other numbers are illustrative, with exact arithmetic.'
            if lang == 'en' else '本地审阅页面，尚未上传或发布。论文数字为公开结果；其余数字均为示意值，计算精确。')
    page = (f'<!doctype html>\n<html lang="{"en" if lang == "en" else "zh-CN"}"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{CSS}</style></head><body class="{lang}"><main>'
            f'<nav><span>X-POST-005 · post-training series 5/6</span><a href="{other[0]}">{other[1]}</a></nav>'
            f'<header><img src="{cover}" width="2000" height="800" alt="{html.escape(title, quote=True)}"></header>'
            f'<h1>{html.escape(title)}</h1><article>{page_body}</article><p class="note">{note}</p></main></body></html>\n')
    (d / 'review.html').write_text(page)
    text = re.sub(r'<[^>]+>', '', raw)
    return dict(lang=lang, title=title, images=len(order),
                words=len(re.findall(r"[A-Za-z0-9'’-]+", text)) if lang == 'en' else None,
                cjk_chars=len(re.findall(r'[一-鿿]', text)) if lang == 'zh' else None,
                outputs={str((d / n).relative_to(P)): sha(d / n)
                         for n in ['x-editor-draft-body-with-image-placeholders.html', 'review.html']})


if __name__ == '__main__':
    results = [build('en'), build('zh')]
    assets = sorted([*P.glob('assets/*'), *P.glob('zh/assets/*')])
    meta = dict(task='X-POST-005', command='uv run --project visuals/animations --with mistune==3.3.4 python '
                'publications/x-articles/x-post-training-005/build.py',
                dependencies={'mistune': importlib.metadata.version('mistune')},
                builds=results, assets={str(a.relative_to(P)): dict(sha256=sha(a), bytes=a.stat().st_size) for a in assets},
                sources={n: sha(P / n) for n in ['build.py', 'build_cover.py', 'source-map.md', 'plan.md',
                                                  'x-editor-draft-body-with-image-placeholders.md',
                                                  'zh/x-editor-draft-body-with-image-placeholders.md']},
                animation_metadata='visuals/animations/projects/post-training-reward-models/metadata.json',
                boundary='No model run, X upload or publication.')
    (P / 'metadata.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n')
    for r in results:
        print(r['lang'], r['title'], 'images', r['images'], 'words', r['words'], 'cjk', r['cjk_chars'])
