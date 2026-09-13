# X-ATTN-KV-001: KV cache article

Local editorial package; not published. The latest Chinese body was uploaded
to X draft `2098863329817051136` after the user cleared its body, and verified
after reload. The cloud cover and original title were preserved; the local title
differs intentionally. See `zh/x-transfer-2026-09-12.md` for current evidence.
Production began
2026-09-11 on `codex/content/x-attn-kv-guide` from `d1374d1` after user approval.

- English: [visual review](review.html), [editable manuscript](x-editor-draft-body-with-image-placeholders.md).
- Chinese adaptation: [visual review](zh/review.html), [editable manuscript](zh/x-editor-draft-body-with-image-placeholders.md).
- Both languages have body-only Markdown/HTML and ordered image-insertion plans.
- Article bodies are self-contained and link-free. Source URLs remain only in
  the separate provenance map. The review's language switch is outside the article.
- The prefill/decode section uses a close-up animation instead of code. Its
  executable NumPy companion is preserved in `projects/kv-decode-step/example.py`
  under the animation directory and tested against full causal attention.
- Current cover: `assets/cover-kv-cache-v2.png`, an AI-generated black-and-white
  stippled editorial concept with one-line “KV Cache”, exported at 2000×800 (5:2).
  Exact prompt/original output/export notes are in `cover-concepts/kv-cache-v2.md`.
  The previous code-native `assets/cover.png` remains available for comparison.
- Five mechanism GIFs cover prefill/decode,
  append/edit, one decode step, memory growth and global KV sharing. Original
  still diagrams remain available. Shared
  English technical labels are intentional; Chinese captions explain the figures.
- Two additional typeset equation cards cover attention and
  cache payload. Fractions, radicals, transpose marks and subscripts are rendered
  locally with Matplotlib MathText, preserving identical appearance in HTML and X.
  Exact formula sources and renderer version are retained in `metadata.json` and
  `build.py`. Inline code/shape notation remains literal; no CDN math script is needed.
- Existing DeepSeek animations appear in the English review's expandable companion section only.
  They are not assumed to be native X Article embeds. Upload plans contain seven
  media items (two equation PNGs and five GIFs); the cover is separate. MP4s remain ignored and local.
- The first loop is about 9 seconds, 960×540, and 1.15 MB, with a PNG fallback,
  pause/resume button, and reduced-motion support in the local review. Editable
  source and the 1080p MP4 live in `visuals/animations/projects/kv-prefill-decode/`.
- The two additional loops live in `visuals/animations/projects/kv-article-loops/`:
  append/edit is 7.67 seconds / 1.07 MB, global sharing 7.87 seconds / 1.04 MB.
  Both use 960×540 GIFs and 1080p MP4s. The first approved loop is unchanged.
- The memory-growth loop lives in `visuals/animations/projects/kv-memory-growth/`.
  It compares retained length, then KV-head configurations, using calculated
  payloads (192→384 MiB and 192→96 MiB). Existing loops remain unchanged.
  The 960×540 GIF is 9.67 seconds / 1.47 MB; the local 1080p MP4 is 9.63 seconds.
- The code-replacement close-up lives in `visuals/animations/projects/kv-decode-step/`:
  8.60-second GIF / 0.745 MB and 8.63-second 1080p MP4. Its toy-weight fixture,
  preserved NumPy companion and cache-geometry checks are included.

## Reproduce

Run from the repository root:

```bash
uv run --project visuals/animations python visuals/animations/projects/kv-prefill-decode/render.py
uv run --project visuals/animations python visuals/animations/projects/kv-article-loops/render.py
uv run --project visuals/animations python visuals/animations/projects/kv-memory-growth/render.py
uv run --project visuals/animations python visuals/animations/projects/kv-decode-step/render.py
uv run --project visuals/animations --with mistune==3.1.3 python publications/x-articles/x-attn-kv-001/build.py
uv run --project visuals/animations python publications/x-articles/x-attn-kv-001/test_example.py
node publications/x-articles/x-attn-kv-001/verify.mjs
```

The extra mistune dependency uses an isolated uv overlay; the animation lock and
environment are not modified.
The builder uses installed Arial, whole-string text shaping, fixed anchors and
code-native geometry. It copies the separately rendered Manim loops only after
checking their render manifests' media hashes; equation cards stay static.
The local X Article Drafter skill's body-plan builder is called for each language.
Browser verification uses the existing looped-transformer Playwright dependency.
It checks both equation cards on desktop/mobile, ordered insertion of all
seven inline images, and the absence of unrendered math source and code blocks.
It also tests all five GIFs' advancement, pause/resume and reduced-motion fallback.
The earlier Chinese revision's X transfer retained its title, cover, 49 text
blocks and seven correctly placed media items after reload; all five GIFs played
in X's preview. That verification does not apply to the newer local wording.

## Chinese story-led revision — 2026-09-12

- Approved editorial plan: a project-document scenario, question-led sections,
  concrete append/edit and cross-request examples, shorter paragraphs, and
  observation prompts before existing animations. Keep accurate NLP terms.
- Local title: **AI 每生成一个 token，都要把前文重新算一遍吗？**
- All seven media slots, English labels, equation cards and the cover are retained.
  English prose and animation sources/media were not edited or rerendered.
- At the learner's follow-up request, removed the Chinese closing advanced
  details and companion-video section, and supplied brief answers to the four
  closing questions. Original video files and the seven inline media remain.
  Reuse prerequisites and Bounded Replay's approximation and evidence limits
  remain beside their claims. No new empirical claims were introduced.
- The prior Chinese source is preserved in
  `zh/revisions/2026-09-12-before-story-led.md`.
- Regenerate only Chinese text/HTML/transfer plans, without rendering assets:
  import `build.py` and call `pages(('zh',))` in the documented Python environment.
- Run `node verify_zh_editorial.mjs` for static content/media integrity checks.
  Results are in `zh/editorial-qa.json`. The browser URL policy blocked local-page
  access during this revision; fresh desktop/mobile visual QA is not claimed.
- Next: learner reviews `zh/review.html`. X draft synchronization, publication,
  commit and push are separate actions; none are included in this revision.
- Follow-up clarification: the shared-document section distinguishes prefill
  (computation phase), KV cache (stored states), and prefix caching (reuse
  strategy), with vLLM's official repeated annual-report/manual workload adapted
  to the existing project-document example. Source verified and recorded in
  `source-map.md`; body remains link-free. New-question prefill and answer decode
  are explicitly retained; no benchmark was run.

The user removed the small-experiment/stale-cache section on 2026-09-12.
Its projection-count equation and evidence diagram are excluded from both article
bodies and upload plans. Their source and assets remain available for course reuse.

## Evidence boundaries

See [source map](source-map.md). Recorded experiment results are attributed to
the existing CPU float64 course report, not a newly executed model experiment.
The builder recalculates memory payloads and projection-row counts. Sentence
tokenization, continuations and the long-prompt example are explicitly schematic.
DeepSeek performance/quality claims remain attributed; Bounded Replay is approximate.
The preserved executable companion is tested against full causal attention on
six fixed-seed input vectors. It does not reproduce or replace the recorded
two-layer experiment's logit values.

Editorial choices: natural Chinese with standard English NLP terms; no forced
translation or “不是……而是……” construction; no contract/interface framing.
Keep the main article focused on exact ordinary cache reuse. Modern variants are
a closing case study. No trained model run, speed benchmark, commit or push is
authorized by drafting this package. Final transfer/publication remain separate.
