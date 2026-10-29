# The Birth of a Token (一颗 token 的诞生)

`X-TOKEN-001`. The learner requested this standalone article for general readers on 2026-10-29. It is
outside the post-training series. It follows one chat request through inference: chat template,
tokenize, prefill, pick, decode, detokenize. It ends with what this loop explains for everyday
users: streaming, varied answers, two kinds of waiting, letter counting, token-based limits, and the
fact that the model does not learn while you chat.

## Status

- 2026-10-29: English and Chinese drafts and the cover written; awaiting learner review.
  - Drafts: `x-editor-draft-body-with-image-placeholders.md` and `zh/…`. Each has 9 H2 sections,
    7 GIFs and no quotation marks. The English has 4 quote blocks. After reviewing the Chinese, the
    learner asked for more quotes and clearer paragraphs, so the Chinese now has 8 quote blocks, one
    per main section.
  - Review pages: `review.html` and `zh/review.html`. Serve this folder (port 8770 in
    `.claude/launch.json`).
  - Every claim maps to `source-map.md`.
- 2026-10-29: the Chinese version was transferred to X draft 2115777748404760576 and
  reload-verified (`X_DRAFT_TRANSFER.md`).
- 2026-10-29: the learner published the Chinese version on X. The English version has not been
  transferred.
- No publication, commit or push.

## Figures

- The seven GIFs come from `CAND-ANIM-040`, `visuals/animations/projects/inference-prompt-to-text/`.
  They were reviewed frame by frame by the agent; the learner rated the set "generally very good".
  `build.py` copies them into `assets/` when the rendered files change.
- Cover: `build_cover.py`. It keeps the Dongxi cover style (deep ink, white Avenir title, gold
  labels) with its own motif: a field of vocabulary scores, one bar rising into a token, and a
  dashed arc returning it to the empty slot at the end of the sequence. The top-left label is
  INSIDE AN LLM, since the article is outside the series.

## Writing rules applied

- No quotation marks; 4 quote blocks; few adverbs; no figurative language beyond the title the
  learner chose.
- The Chinese version uses no 不是/而是 patterns.
- Core terms stay in English in the Chinese text: token, logits, softmax, temperature, top-p,
  greedy decoding, sampling, prefill, decode, KV cache, LM head, embedding, chat template.
- Real numbers come from the local Qwen3-0.6B tokenizer and config. Illustrative numbers are
  labelled as such.

## Build

```sh
uv run --project visuals/animations python publications/x-articles/x-token-001/build_cover.py
uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-token-001/build.py
```
