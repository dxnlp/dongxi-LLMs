# Why Does a Language Model Need Post-Training?

`X-POST-001`, first article of the post-training series (five articles). Bilingual: English at
the package root, an independently written Chinese version in `zh/`.

**Read locally:** [`review.html`](review.html) (English) and
[`zh/review.html`](zh/review.html) (Chinese). Serve the folder so images load:

```bash
python3 -m http.server 8765 --directory publications/x-articles/x-post-training-001
```

Then open `http://localhost:8765/review.html`.

Series plan (2026-10-20): this article stays beginner-level — why post-training,
how it differs from pretraining, and a very short, easy introduction to the
three techniques named by SLP3 Chapter 8. Each later article takes one technique
in depth:

| ID | Topic | Status |
|---|---|---|
| `X-POST-001` | Why post-training (this package) | Chinese version transferred to an X draft and verified after reload (not published, see `X_DRAFT_TRANSFER.md`); English version ready, not transferred |
| `X-POST-002` | Instruction tuning (SFT) | outline agreed 2026-10-22 |
| `X-POST-003` | LoRA and QLoRA (split out of SFT, 2026-10-22) | planned |
| `X-POST-004` | RL terms for LLMs (inserted 2026-10-27, before preference alignment) | planned |
| `X-POST-005` | Reward models (solely; revised 2026-10-30, plan in `../x-post-training-005/plan.md`) | planned |
| — | Policy optimization: PPO, GRPO, DPO (split out of `X-POST-005` 2026-10-30; slot to be decided) | not scheduled |
| `X-POST-006` | RLVR | planned |

## Files

- `x-editor-draft-body-with-image-placeholders.md` / `zh/…` — manuscripts with six `[IMAGE NN]` slots.
- `image-upload-order.md`, `zh/image-upload-order.md` — six images with captions; cover listed separately.
- `assets/` — five GIFs (1280×720, 20 fps, shared by both versions, English labels),
  their posters, English comparison PNG and the shared English cover. `zh/assets/` —
  Chinese comparison PNG.
- `build_cover.py` — procedural 2000×800 cover (vector ribbons via pycairo, gate colors
  matching the animations, series/brand labels); one English cover for both versions.
- `build.py` — HTML bodies, review pages and `metadata.json` (hashes).
- `x-editor-clean-body.*`, `x-editor-clean-body-image-plan.*` (root and `zh/`) —
  draft-transfer preparation from `x-article-drafter`.
- `visual-plan.md`, `source-map.md`.
- Animation source: `visuals/animations/projects/post-training-intro/`.

## Rebuild

```bash
uv run --project visuals/animations python visuals/animations/projects/post-training-intro/render.py
uv run --project visuals/animations python publications/x-articles/x-post-training-001/build_cover.py
uv run --project visuals/animations --with mistune==3.3.4 python publications/x-articles/x-post-training-001/build.py
python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-post-training-001
python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py publications/x-articles/x-post-training-001/zh
```

## Checks done

- Every placeholder maps to an existing asset; both image plans report 6 images, 0 missing.
- Chinese text scanned for 不是/而是/并非/不在于/与其/不如说: none.
- All seven images load at full resolution in both review pages (local HTTP, in-app browser).
- Animation timelines validated by `render.py`; contact sheets and final frames inspected.

The cover is procedural (no image-generation model was used, unlike earlier
series covers); revised 2026-10-20 at the learner's request with color and series labels. No model runs or publication. Chinese X draft created 2026-10-20 (draft only).
