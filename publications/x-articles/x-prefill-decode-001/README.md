# Prefill and Decode

Draft transfer, 2026-09-14: uploaded to the user-designated X draft; title, body,
seven animations/captions and cover verified after reload. Not published.
See `X_DRAFT_TRANSFER.md` for the exact URL, audit and GIF/video adaptation.
Earlier local-only notes below describe production before this transfer.

`X-PD-001`: independent Chinese local article, derived from Chapters 4 §4.10–4.11
and 5 §5.18. Mechanisms first; a short Spark/Mac illustration last. Existing KV
and inference-overview articles are unchanged. Active Day 8 is unchanged.

- `zh/review.html`: local review with seven GIF loops and per-figure static switches.
- `zh/x-editor-draft-body-with-image-placeholders.md`: editable manuscript.
- `zh/x-editor-clean-body.*`, `zh/x-editor-clean-body-image-plan.*`:
  draft-only transfer preparation.
- `assets/`: English GIFs/posters, retained original figures, single-line 2000×800 cover.
- `source-map.md`: provenance and limits; public body intentionally has no links.
- `visual-plan.md`: storyboard and seven-clip production return.
- Current cover: `assets/cover-prefill-decode-v2.png`, matching the KV Cache
  article's black-and-white stippled editorial series. Exact 2000×800, one-line
  title, abstract parallel-input-to-sequential-output motif. Built-in generation
  prompt and source are in `cover-concepts/prefill-decode-v2.md`. Original white
  `assets/cover.png` remains available; rebuilding keeps the new cover selected.

2026-09-14: added a static decode feedback loop to section 3. The learner
explicitly deferred all animations for later production together; no animation
was rendered. A subsequent clarification adds input-derived Q/K/V dependencies
and parallel attention rows to section 2. Seven-image upload order is independent
of stable asset filenames. This was the static-review stage; the later approval
and animation return below supersede its production deferral.

Editorial revision: question-led Chinese narrative following a meeting/report
request, a hypothetical token continuation and a long-speech contrast. Figure 05
(stable filename `03-shapes.png`) now shows workload volumes rather than matrix
blocks; figure 06 names the active answer A and incoming document B. The earlier
parallel-position explanation remains. No changes to the previous KV article.

Regenerate from the repository root:

```sh
uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-prefill-decode-001/build.py --animations
python3 /Users/yongchao/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-prefill-decode-001/zh
uv run --project visuals/animations python publications/x-articles/x-prefill-decode-001/check.py
```

Animation return, 2026-09-14: all seven scenes rendered in
`visuals/animations/projects/prefill-decode-article/`. Each has a locally retained
1080p MP4, 960×540 looping GIF, final poster and contact sheet. The cover remains
static. GIFs play by default with JavaScript; each button switches to a static
poster, rather than freezing the current GIF frame. Reduced-motion preference or
disabled JavaScript starts with posters. The original article PNGs remain too.
Run the build without `--animations` to regenerate the original static review.

No model-scale experiment, measured timings, hardware setup, browser/X transfer,
commit or push. Prior browser URL-policy block respected: PNGs
are inspected directly; automated browser/mobile HTML visual QA is not claimed.
`test-review-player.cjs` checks seven controls and reduced-motion behavior using
Node DOM stubs; this is a unit test, not browser playback verification.
Repo began on dirty `main` at `ca2cfe1`; existing work retained and pull skipped.
