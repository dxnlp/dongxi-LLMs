# X-BPE-001 — local X Article package

Status: English and Chinese packages prepared for editorial review. Nothing has
been transferred to X and nothing has been published.

Working title: **What Is a Token, Really? From Unicode Bytes to BPE and Token IDs**

## Package

- `x-editor-draft-body-with-image-placeholders.md`: canonical review draft.
- `x-editor-draft-body-with-image-placeholders.html`: rich-text transfer source.
- `x-editor-clean-body.md` and `.html`: generated transfer bodies without title
  or image placeholders.
- `review.html`: responsive local reading version with the cover, all five inline
  images, captions, article text, and source links in final reading order.
- `image-upload-order.md`: ordered inline-image list.
- `x-editor-clean-body-image-plan.md` and `.json`: generated bottom-to-top image
  insertion plan.
- `source-map.md`: claim-to-evidence map and editorial limits.
- `terminology.md`: canonical bilingual NLP terminology shared by both versions.
- `build_article_visuals.py`: editable source for the static visual package.
- `assets/cover.png`: proposed 2000×800 (5:2) X Article cover; it is not inserted
  into the body.
- `assets/01`–`05`: inline images in upload order. Image 03 is copied from the
  approved Manim animation still by the build script.
- `assets/metadata.json`: dimensions, hashes, fonts, palette, and visual limits.
- `zh/`: complete natural-Chinese adaptation with its own localized 5:2 cover,
  five localized inline figures, claim map, review HTML, transfer bodies, and
  insertion plan.

## Rebuild

From the repository root:

```bash
python3 publications/x-articles/x-bpe-001/build_article_visuals.py

python3 publications/x-articles/x-bpe-001/build_html.py

python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py \
  publications/x-articles/x-bpe-001
```

The visual builder uses Arial for English, Songti SC for Chinese, a white canvas,
and the semantic palette in `visuals/animations/STYLE_GUIDE.md`. The reviewed
local build used Pillow 11.2.1 and Mistune 2.0.5.

## Review boundary

The English claims and sequence should be reviewed before any Chinese adaptation
or X editor transfer. The user retains final control of title, cover, draft
transfer, and publication.
