# Chapter mechanisms in motion

Authorized on 2026-10-11: complete the reviewed 28 GIF placements and embed them
in their corresponding chapters. This is one bounded production package,
`ANIM-CHAPTER-GIFS-001`, extending existing candidates rather than replacing the
historical signature-animation packets. Rendering belongs on the Mac.

Completed locally on 2026-10-11: all 28 placements are embedded and verified.
The 23 new GIFs total 6,966,279 bytes (6.64 MiB); all 28 embedded assets total
9,771,830 bytes (9.32 MiB). See the [completion review](review-record.json).
All 464 chapter links, fifteen chapter browser previews, numerical/media checks,
1,871 existing formulas and 25 book regression tests pass. Existing prose/math,
notebooks, model code and historical evidence remain intact.

Base commit: `4e5a5ccda6327a7a8a7d88ebfc5b73c2280a778e`; branch: `main`.
Inputs are the approved placement plan, current chapters/notebook routes,
existing GIFs and the declared local fixtures. Allowed edits are
`visuals/chapters/`, the fifteen chapter inserts, current `PROGRESS.md` and
`LEARNING_MEMORY.md` entries, and an append-only `PROPOSALS.md` extension.
Outputs are the GIFs/PNGs, editable source, manifests, gallery and verification.
Gradient finite differences use a 1e-8 agreement tolerance; historical policy
replay uses 1e-12; chapter and retained-media preservation is byte-exact.

## Learning objective and scope

Use motion to expose token alignment, information and gradient paths,
probability transformations, state restoration, objective weighting and failure
mechanisms. Every placement has a prediction question, a runnable notebook route,
a descriptive caption and a PNG fallback.

| Chapter | Placements | Mechanisms |
|---|---:|---|
| 1 | 1 | Loss versus a capability claim |
| 2 | 2 | Existing BPE; contextual embedding gradients |
| 3 | 3 | Decoding; cross-entropy gradient; causal shift |
| 4 | 2 | Masked attention; one cached decode step |
| 5 | 3 | Residual path; rotary positions; grouped-query attention |
| 6 | 2 | Valid-target accumulation; checkpoint state |
| 7 | 1 | Raw response, parsing and grading |
| 8 | 2 | Assistant-owned targets; packed information boundaries |
| 9 | 2 | Teacher forcing; low-rank update |
| 10 | 1 | Pair scores and preference gradients |
| 11 | 2 | Reference-relative DPO; absolute-probability counterexample |
| 12 | 2 | Baseline covariance; sign-aware PPO clipping |
| 13 | 2 | Group advantages; response/token weighting |
| 14 | 1 | Finite-policy reward hacking |
| 15 | 2 | Candidate selection; soft-target distillation |

One BPE GIF remains in place, four existing GIFs are reused without changing
their bytes, and 23 new short loops are produced. The optional advanced topics
from the proposal remain outside this package.

## Source and reproduction

- [`catalog.py`](catalog.py): all 28 routes, learning questions, captions and
  existing candidate identities.
- [`fixtures.py`](fixtures.py): deterministic calculations and independent
  finite-difference checks; no model loading or GPU run.
- [`render.py`](render.py): 23 mechanism-specific scenes using the established
  [README renderer](../readme/render.py)'s canvas, semantic palette and geometry.
  Reuse the [approved style](../animations/STYLE_GUIDE.md), adapting the scene to
  the mechanism. Pillow draws these small code-native diagrams; the existing
  FFmpeg palette exporter produces GIFs directly from temporary PNGs.
- [`integrate.py`](integrate.py): idempotent insertion, decoded-media/link/hash
  verification and a responsive gallery with pause and reduced-motion support.
- [`manifest.json`](manifest.json): actual renderer/source/font hashes,
  environment, command, numerical checks and export metadata.
- [`verification.json`](verification.json): placement coverage, media hashes,
  links and exact chapter preservation against the declared base commit.
- [`review.html`](review.html): all chapter mechanisms and prediction prompts.

From the repository root, using the existing isolated animation environment:

```bash
uv run --project visuals/animations python visuals/chapters/fixtures.py
uv run --project visuals/animations python visuals/chapters/render.py --qa-dir /tmp/dongxi-chapter-qa
uv run --project visuals/animations python visuals/chapters/integrate.py --apply
uv run --project visuals/animations python visuals/chapters/integrate.py --verify
python3 scripts/check_book_math.py
python3 scripts/check_book_prose.py
```

The three pilots can be regenerated with `--only embedding-gradient
assistant-loss-mask ppo-clipping`. Each new GIF is 1120×560, 12 fps nominal,
12 seconds, infinitely looping and less than 3 MiB. Frame delays reflect GIF's
centisecond time resolution. The final white fade is an editorial reset.
No MP4 is generated; the existing repository video-ignore rule remains intact.

## Evidence boundaries

All vectors, score updates, conditionals and probability paths are constructed
teaching fixtures except the explicitly declared Chapter 14 finite-policy replay.
Animation time is not measured execution latency. Calculated distributions do
not establish semantic understanding or language-model capability.

The reward-hacking loop independently replays the exact 60-step float64 SGD
recurrence in
[`optimization_diagnostics_lab.py`](../../src/dongxi_llms/optimization_diagnostics_lab.py).
Verification compares every update with the existing
[recorded report](../../experiments/reports/2026-10-04-grpo-diagnostics-distillation.json).
The plot advances through integer updates; it never interpolates empirical
points. Proxy reward is divided by two only to share the displayed 0–1 axis with
strict accuracy, and that scaling is labeled.

The DPO counterexample interpolates constructed normalized vectors; it is not
an optimizer trajectory. The LoRA scene illustrates factorization and checked
first-step gradient asymmetry; later factor values are constructed. The packed
attention scene shows mask permission, not learned attention probabilities.
The distillation scene freezes teacher logits, changes a declared temperature,
then performs one independent student-logit descent step at the fixed final
temperature and shared prefix.

## Original acceptance criteria

1. Exactly 28 placements across all 15 chapters, with the agreed per-chapter
   counts and no duplicate BPE embed.
2. All 23 new GIFs have mechanism-specific editable source, calculated fixtures,
   source hashes, reproducible commands and matching PNG stills.
3. Finite-difference gradients, mask/shift rules, RoPE invariants, reductions,
   checkpoint RNG continuation, PPO branches and finite-policy replay pass.
4. Every decoded GIF loops correctly, has multiple distinct frames and positive
   delays, retains its dimensions and passes the declared size/duration checks.
5. Contact sheets and full/half-size stills are inspected; layout and gallery
   controls are reviewed at normal and narrow widths.
6. All chapter/media/notebook links resolve. Removing only the inserted blocks
   recovers the exact original chapter bytes at
   `4e5a5ccda6327a7a8a7d88ebfc5b73c2280a778e`; existing reused GIF bytes are unchanged.
7. Local book math/prose checks pass; notebooks, learner cells, experiment
   reports and model evidence remain untouched. Learner Day 9 is unchanged.
8. No video is added to Git, no model campaign is launched, and this package
   does not imply external publication or learner mastery.

The final completion and actual review evidence belong in the verification
receipt and the current progress/learning-memory entries, separately from
historical course verification results.
