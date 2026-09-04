# ANIM-BPE-002 — A meteor-shower corpus becomes tokens

Status: local review candidate ready; awaiting user visual approval. This is a supporting
Chapter 2 animation, separate from the Day 3 experiments and signature releases.

## Production brief

- Base commit: `d91f5eb` (record full identity in the render metadata).
- Branch: `codex/visuals/meteor-bpe`.
- Request: animate `流`, `星`, `雨`, `流星`, `流星雨` from the five supplied
  strings and use the established minimal Manim style.
- Learning objective: follow measured pair counts into two BPE training merges,
  retain constituent vocabulary entries, then encode with the learned rules and
  a fixed token-to-ID mapping.
- Source: Chapter 2; `src/dongxi_llms/tiny_bpe.py`; the user's five strings.
- Allowed scope: this animation project, its executable corpus trace and tests,
  the related learning artifact, and production indexes. Do not alter the
  ongoing Day 3 course state or replace the existing BPE animation.

## Fixed experiment specification and predictions (before execution)

Mode: smoke / transparent CPU-only teaching mechanism. No model weights or GPU.
No normalization. Each complete supplied string is a separate pre-tokenized
segment; pairs never cross segment boundaries. Each string occurs once:

```text
流星蝴蝶剑
带你去看流星雨
流水
星辰大海
雨一直下
```

Initialize from Unicode characters, then perform exactly two BPE merges.
Select the highest weighted adjacent-pair count. Among tied maxima only, use the
predeclared preference `(流星, 雨)`; otherwise break ties by Unicode
lexicographic order. The preference deliberately makes the requested teaching
example deterministic; it is not a standard or uniquely determined BPE rule.
Never change frequencies or omit competing pairs to produce the desired result.

Predictions:

1. 22 initial tokens and 17 distinct base characters.
2. `(流, 星)` is the unique maximum at count 2 in round 1.
3. After that merge, all 15 remaining pair candidates have count 1. The declared
   tie preference selects `(流星, 雨)` in round 2.
4. The corpus token totals become 22 → 20 → 19.
5. The vocabulary retains all 17 characters and gains two entries: 19 total.
6. Assign illustrative IDs after training: `流=1, 星=2, 雨=3, 流星=4,
   流星雨=5`; assign the other 14 characters IDs 6–19 in first-appearance order.
7. Encoding `流星雨` applies both learned merges and yields `[5]`; encoding
   `星雨` yields `[2, 3]`. All five supplied strings decode exactly.

## Storyboard and precision

1. Reveal the five unchanged strings, then segment them into character tokens.
2. Highlight both occurrences of `(流, 星)`, count them, and fuse the pair in
   both corpus rows. Copy the new entry into the visible vocabulary subset.
3. Highlight `(流星, 雨)` once, display the tied-maximum condition, and merge
   only that occurrence. Keep all constituent vocabulary entries visible.
4. Assign the five requested teaching IDs to the visible vocabulary subset.
5. Switch explicitly from tokenizer training to encoding. Replay the two rules
   on `流星雨`, look up ID 5, and end beside the retained five-entry subset.

White canvas; Songti SC Chinese; naturally kerned Arial English and numbers;
semantic blue/purple/green; stable geometry; no dense tables or prose panels.
The narrator's human-experience contrast motivates the topic. The animation
does not claim that BPE learns semantics, that IDs have an inherent order of
meaning, or that tokenization determines a model's eventual semantic limits.
This is character-level BPE, not a byte-level tokenizer or a historical Qwen run.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src uv run --project visuals/animations \
  python -m dongxi_llms.meteor_bpe \
  --output visuals/animations/projects/meteor-bpe/trace.json

PYTHONPATH=src uv run --project visuals/animations \
  python -m unittest discover -s tests -p 'test_*bpe*.py' -v

uv run --project visuals/animations \
  python visuals/animations/projects/meteor-bpe/render.py --preview

uv run --project visuals/animations \
  python visuals/animations/projects/meteor-bpe/render.py
```

Acceptance: every displayed count and segmentation comes from `trace.json`;
the tie is disclosed on screen; IDs 1–5 are explicitly a teaching subset;
constituents remain in the vocabulary; encoding adds no vocabulary entries;
all transitions and text fit the canvas and read at 50% scale. Preserve a
1080p H.264 MP4, 960×540 GIF, PNG still, contact sheet, metadata, and local review
HTML. Source integration was approved on 2026-09-04; final visual review remains
pending. MP4 files stay local and are ignored by Git. GIF and still previews are
versioned; on a fresh clone, run the full render command above to enable MP4
playback in `review.html`, or use its GIF preview link without rendering.

## Measured result and visual review — 2026-09-03

- All seven predictions above matched the executable trace; eight encoding
  examples round-tripped exactly. All nine focused BPE tests passed.
- Main render: 40.23 seconds, 1920×1080, 30 fps, H.264/yuv420p;
  2,956,598 bytes (2.82 MiB). GIF: 960×540, 2,577,222 bytes.
- Open `review.html` for the current movie, original corpus, full trace, and
  precision boundaries. `rendered/preview.mp4` is an earlier draft, not the
  final candidate; `rendered/meteor-bpe.mp4` is authoritative.
- Reviewed all twelve storyboard checkpoints, both merge transitions, and
  full/half-size final stills. Geometry checks passed. Merge motion preserves
  the glyph shapes: characters translate together while token frames fuse.
- Source and media hashes, environment identity, and checkpoint timing are
  preserved in `metadata.json`. No existing animation was replaced or published.
- Next action: user review of pacing, counting clarity, and the transition from
  tokenizer training to fixed-vocabulary encoding before integration.
