#!/usr/bin/env python3
"""Build the X transfer source and a complete browser-review page."""

import html
from pathlib import Path
import re

import mistune


ARTICLE_DIR = Path(__file__).resolve().parent
SOURCE = ARTICLE_DIR / "x-editor-draft-body-with-image-placeholders.md"
TARGET = ARTICLE_DIR / "x-editor-draft-body-with-image-placeholders.html"
REVIEW = ARTICLE_DIR / "review.html"

IMAGES = {
    1: "assets/01-six-units.png",
    2: "assets/02-tiny-bpe-training.png",
    3: "assets/03-byte-coverage-compression.png",
    4: "assets/04-interface-surprises.png",
    5: "assets/05-multilingual-measurement.png",
}

REVIEW_STYLE = """
:root {
  color-scheme: light;
  --background: #ffffff;
  --foreground: #111827;
  --muted: #64748b;
  --grid: #cbd5e1;
  --base: #2563eb;
  --soft: #f8fafc;
}
* { box-sizing: border-box; }
html { background: var(--background); }
body {
  margin: 0;
  color: var(--foreground);
  background: var(--background);
  font-family: Arial, sans-serif;
  font-size: 19px;
  line-height: 1.68;
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
  font-size: 14px;
  letter-spacing: 0.04em;
  text-align: center;
  text-transform: uppercase;
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
  max-width: 820px;
  margin: 0 0 16px;
  font-size: clamp(42px, 7vw, 72px);
  font-weight: 400;
  line-height: 1.04;
  letter-spacing: -0.035em;
}
h2 {
  margin: 72px 0 18px;
  font-size: clamp(29px, 4.2vw, 40px);
  font-weight: 400;
  line-height: 1.15;
  letter-spacing: -0.02em;
}
p, ul, pre, blockquote { margin: 0 0 24px; }
p:first-of-type { color: var(--muted); font-size: 22px; }
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
  line-height: 1.5;
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


def build_review(markdown_source: str, markdown: mistune.Markdown) -> None:
    def image_figure(match: re.Match[str]) -> str:
        number = int(match.group(1))
        caption = match.group(2).strip()
        source = IMAGES[number]
        return (
            f'<figure id="image-{number:02d}">\n'
            f'  <img src="{source}" alt="{html.escape(caption, quote=True)}">\n'
            f'  <figcaption>{html.escape(caption)}</figcaption>\n'
            f'</figure>'
        )

    review_markdown = re.sub(
        r"^\[IMAGE\s+(\d{2})\]\s+(.+)$",
        image_figure,
        markdown_source,
        flags=re.M,
    )
    article_html = markdown(review_markdown)
    article_html = article_html.replace(
        "<h2>Reproducible sources</h2>",
        '<section class="sources">\n<h2>Reproducible sources</h2>',
    ) + "\n</section>"
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>What Is a Token, Really? — Local Review</title>
  <style>{REVIEW_STYLE}</style>
</head>
<body>
  <div class="review-banner">Local editorial review · not transferred to X · not published</div>
  <main>
    <figure class="cover">
      <img src="assets/cover.png" alt="What Is a Token, Really? From Unicode bytes to BPE and token IDs">
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
    # X Articles has a separate title field. Keep the H1 in the Markdown review
    # source, but make the rich-text transfer source body-only.
    rendered = re.sub(r"\A<h1>.*?</h1>\s*", "", rendered, count=1, flags=re.S)
    TARGET.write_text("<!doctype html>\n<meta charset=\"utf-8\">\n" + rendered, encoding="utf-8")
    build_review(source_text, markdown)
    print(TARGET)
    print(REVIEW)


if __name__ == "__main__":
    main()
