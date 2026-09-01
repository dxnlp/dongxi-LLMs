#!/usr/bin/env python3
"""Build the Chinese X transfer source and complete browser-review page."""

from __future__ import annotations

import html
import re
from pathlib import Path

import mistune


ARTICLE_DIR = Path(__file__).resolve().parent
SOURCE = ARTICLE_DIR / "x-editor-draft-body-with-image-placeholders.md"
TRANSFER = ARTICLE_DIR / "x-editor-draft-body-with-image-placeholders.html"
REVIEW = ARTICLE_DIR / "review.html"
IMAGES = {number: f"assets/{number:02d}-{name}.png" for number, name in {
    1: "six-units",
    2: "tiny-bpe-training",
    3: "byte-coverage-compression",
    4: "interface-surprises",
    5: "multilingual-measurement",
}.items()}

STYLE = """
:root {
  color-scheme: light;
  --foreground: #111827;
  --muted: #64748b;
  --grid: #cbd5e1;
  --base: #2563eb;
  --soft: #f8fafc;
}
* { box-sizing: border-box; }
html, body { background: #fff; }
body {
  margin: 0;
  color: var(--foreground);
  font-family: "Songti SC", STSong, serif;
  font-size: 19px;
  line-height: 1.82;
  text-rendering: optimizeLegibility;
}
.review-banner {
  position: sticky;
  top: 0;
  z-index: 10;
  padding: 10px 20px;
  color: var(--muted);
  background: rgba(248, 250, 252, 0.96);
  border-bottom: 1px solid var(--grid);
  font-family: Arial, "Songti SC", sans-serif;
  font-size: 14px;
  letter-spacing: 0.04em;
  text-align: center;
  backdrop-filter: blur(8px);
}
main { width: min(920px, calc(100% - 40px)); margin: 40px auto 96px; }
.cover { margin: 0 0 48px; }
.cover img, figure img {
  display: block;
  width: 100%;
  height: auto;
  border: 1px solid #e2e8f0;
  border-radius: 16px;
  background: #fff;
}
h1 {
  max-width: 880px;
  margin: 0 0 18px;
  font-size: clamp(40px, 6.4vw, 66px);
  font-weight: 400;
  line-height: 1.18;
  letter-spacing: -0.02em;
}
h2 {
  margin: 72px 0 18px;
  font-size: clamp(29px, 4.2vw, 40px);
  font-weight: 400;
  line-height: 1.35;
}
p, ul, pre, blockquote { margin: 0 0 24px; }
p:first-of-type { color: var(--muted); font-size: 21px; }
strong { font-weight: 600; }
a { color: var(--base); text-underline-offset: 3px; }
ul { padding-left: 28px; }
li { margin: 8px 0; }
blockquote {
  margin: 32px 0;
  padding: 20px 24px;
  border-left: 4px solid var(--base);
  background: var(--soft);
  border-radius: 0 12px 12px 0;
}
blockquote p { margin: 0; color: var(--foreground); font-size: 20px; }
code {
  padding: 0.12em 0.34em;
  border-radius: 5px;
  background: #f1f5f9;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.88em;
}
pre {
  overflow-x: auto;
  padding: 22px 24px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: var(--soft);
  line-height: 1.55;
}
pre code { padding: 0; background: transparent; }
figure { margin: 42px 0 52px; }
figcaption {
  max-width: 820px;
  margin: 13px auto 0;
  color: var(--muted);
  font-size: 14px;
  line-height: 1.65;
  text-align: center;
}
.sources { margin-top: 80px; padding-top: 1px; border-top: 1px solid #e2e8f0; }
@media (max-width: 640px) {
  body { font-size: 17px; }
  main { width: min(100% - 24px, 920px); margin-top: 20px; }
  .cover { margin-bottom: 32px; }
  .cover img, figure img { border-radius: 10px; }
  h2 { margin-top: 56px; }
  pre { padding: 16px; }
  figure { margin: 34px 0 44px; }
}
"""


def build_review(source_text: str, markdown: mistune.Markdown) -> None:
    def image_figure(match: re.Match[str]) -> str:
        number = int(match.group(1))
        caption = match.group(2).strip()
        return (
            f'<figure id="image-{number:02d}">\n'
            f'  <img src="{IMAGES[number]}" alt="{html.escape(caption, quote=True)}">\n'
            f'  <figcaption>{html.escape(caption)}</figcaption>\n'
            f'</figure>'
        )

    review_markdown = re.sub(
        r"^\[IMAGE\s+(\d{2})\]\s+(.+)$",
        image_figure,
        source_text,
        flags=re.M,
    )
    article_html = markdown(review_markdown)
    article_html = article_html.replace(
        "<h2>可复现来源</h2>",
        '<section class="sources">\n<h2>可复现来源</h2>',
    ) + "\n</section>"
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Token 到底是什么？— 中文版本地审阅</title>
  <style>{STYLE}</style>
</head>
<body>
  <div class="review-banner">本地编辑审阅 · 尚未传入 X · 尚未发布</div>
  <main>
    <figure class="cover">
      <img src="assets/cover.png" alt="Token 到底是什么？从 Unicode 字节到 BPE 与 Token ID">
    </figure>
{article_html}
  </main>
</body>
</html>
"""
    REVIEW.write_text(page, encoding="utf-8")


def main() -> None:
    markdown = mistune.create_markdown(escape=False)
    source_text = SOURCE.read_text(encoding="utf-8")
    rendered = markdown(source_text)
    rendered = re.sub(r"\A<h1>.*?</h1>\s*", "", rendered, count=1, flags=re.S)
    TRANSFER.write_text(
        '<!doctype html>\n<meta charset="utf-8">\n' + rendered,
        encoding="utf-8",
    )
    build_review(source_text, markdown)
    print(TRANSFER)
    print(REVIEW)


if __name__ == "__main__":
    main()
