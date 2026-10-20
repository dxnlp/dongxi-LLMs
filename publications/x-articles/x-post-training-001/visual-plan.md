# Visual plan — X-POST-001

Storyboard approved in discussion on 2026-10-20 ("sounds good"), with the
learner's direction that this first article keeps the three techniques as easy
as possible, with easy-to-understand animations. Rendered on 2026-10-20 at the
learner's request (`visuals/animations/projects/post-training-intro/`), adding a
static EN/ZH comparison figure and procedural covers. Task packet: `ANIM-POST-001` (below; indexed in
`LEARNING_MEMORY.md`, candidate `CAND-ANIM-031` in `visuals/animations/PROPOSALS.md`).

## Design rules for this article

- Beginner level. No equations on screen; at most one or two Greek-free labels
  per object. Depth belongs to the later per-technique articles.
- Dongxi Manim system (`visuals/animations/STYLE_GUIDE.md`): white canvas,
  semantic colors, Arial labels, motion grammar focus → approach → transform →
  pulse → rest. English labels only; both language versions share the GIFs.
- One model object (the same rounded "model" block) appears in every clip, so
  readers see one model being trained throughout the series.
- Consistent teacher colors across A2–A5: examples = blue `#2563EB`,
  comparisons = violet `#7C3AED`, checker = green `#047857`; leash/caveat =
  amber `#B45309`.
- Each loop 7–12 s, 960×540 GIF plus 1080p MP4, poster PNG, contact sheet.
  Reuse the exporter in `visuals/animations/projects/kv-article-loops/render.py`
  and character pieces from `rl-ppo-grpo-dpo-beginner/kit.py` where they fit.

## A1 — Same prompt, two models (`01-same-prompt-two-models`)

A request card ("How do I make my tea less bitter?" — a neutral stand-in for the
article's quoted example) enters two identical model blocks, labeled
`base model` and `post-trained`. Above each, a bar chart of next-word
candidates grows. Base: tallest bars are question-starters (`How`, `What`,
`Why`). Post-trained: tallest bars are answer-starters (`Use`, `Try`,
`Brew`). The tallest word of each is pulled out and a short continuation
types out. Caveat badge: `illustrative probabilities`.

Accuracy: probabilities are authored, not measured. The neutral prompt avoids
replaying harmful content; the article text carries the historical example.

## A2 — One model, two stages (`02-two-stages`)

Left: a wide, dense river of grey text fragments flows into the model block
(`pretraining · predict the next word`). The block fills with a texture
("knowledge"). Then the river stops and three thin, colored streams enter in
turn: blue cards (`examples`), violet card pairs (`comparisons`), green
checkmarked problems (`checked answers`). The block keeps its texture and gains
a small steering outline. Caveat: `volumes not to scale`.

Accuracy: same weights continue training; the three post-training stages are
shown in the common order but real pipelines vary.

## A3 — Instruction tuning (`03-instruction-tuning`)

A token row: instruction tokens (`Translate`, `to`, `English:`, `No`,
`tenía`, `lavandería`) then response tokens (`I`, `didn't`, `have`, `a`,
`laundry`). Instruction tokens dim to grey with label `context`. A highlight
walks the response left to right; at each step the model's guess bar is compared
with the true word and a small `graded` tick appears. Final frame: only the
response row carries ticks.

Accuracy: loss on response tokens only (SLP3 §8.1, Fig. 8.1); example adapted
from SLP3's Spanish→English figure.

## A4 — Preference alignment (`04-preference-alignment`)

A question card, then two answer cards side by side. A hand/check marks one
`preferred`. A balance tips toward it; the model's probability bars shift so
the preferred answer grows and the other shrinks. An amber leash from a faded
copy labeled `original` limits how far the bars move — they stop at the leash's
length. Small end label: `RLHF (judge) or DPO (no judge)`.

Accuracy: simplifies both routes into one picture (stated in caption). The
leash represents the KL penalty to the reference model (SLP3 eq. 8.6).

## A5 — RLVR (`05-rlvr`)

A problem card `17 × 3 = ?`. Four attempt cards fan out from the model
(`51`, `41`, `51`, `57`). A checker block reads each final answer and stamps
✓ or ✗. Bars for the attempts rise for ✓ and fall for ✗; next round, more
attempts land on `51`. Caveat badge: `checks final answer only`.

Accuracy: outcome-only verifier; no claim that steps are verified. Arithmetic
is checked in the fixture (17 × 3 = 51).

## Cover (5:2, 2000×800)

Built by `build_cover.py` (pycairo vector ribbons, Pillow text). Revised on
2026-10-20 after learner review: smoother engraved ribbons instead of random
stipple, color allowed. A broad river of raw-text strands narrows through three
gates colored like the animations (show = blue, compare = violet, check = green)
into one bright answer line with beads. Title "Why Post-Training?" in solid white
Avenir Next Heavy (gradient tried and dropped at learner review). Gold
`#FBBF24` labels "POST-TRAINING SERIES · 01" (top left) and "DONGXI LABS"
(top right, letter-spaced caps; chosen over "Dongxi Labs" at learner review); no other labels (the bottom-right X-POST-001 tag was
removed at learner review). One English cover is shared by both language
versions at the learner's request.

## ANIM-POST-001 task packet

- Base / branch / machine: current `main`, Mac Studio.
- Objective: five beginner loops (A1–A5) and a 5:2 cover for X-POST-001.
- Inputs: this plan; SLP3 Ch. 8; book Chapters 8, 10, 13; style guide.
- Allowed files: `visuals/animations/projects/post-training-intro/`,
  `publications/x-articles/x-post-training-001/`.
- Outputs: per clip MP4 (git-ignored), 960×540 GIF < 5 MB, poster PNG,
  contact sheet, timeline JSON, metadata with hashes and render command.
- Acceptance: contact-sheet and 50%-scale inspection; fixture checks (A3 mask
  covers response only; A5 arithmetic; A1/A2 caveat badges present); no
  production IDs in frames; learner visual approval before X transfer.
- Not included: X upload, publication, model runs or measured probabilities.
