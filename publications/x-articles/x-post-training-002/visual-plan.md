# Visual plan — X-POST-002

Approved by the learner 2026-10-22 (`ANIM-POST-002`). Follows the series conventions in
`../README.md` (animated GIFs for every figure, Dongxi Manim style, English labels shared
by both versions, 1280×720 20 fps GIF under 5 MB, 1080p MP4 kept local). Planned source:
`visuals/animations/projects/post-training-sft/` (reusing helpers from `post-training-intro/`).

Revised twice on 2026-10-22 after learner review: first for less text (one idea per GIF,
few large labels, motion carries the meaning), then for the big-picture rewrite (LIMA, FLAN
and recipe GIFs added; a static three-forms summary added; a whole-conversation loss overview
replaces the end-token GIF; no measured values from our own experiments). All figures are
English only and shared by both versions.

| # | Figure | Content | Evidence |
|---|---|---|---|
| 01 | 52,000 examples make an assistant (GIF) | AI teacher → example cards → LLaMA 7B becomes Alpaca 7B; three stats | Stanford's published numbers |
| 02 | LIMA (GIF) | 1,000 example squares become a thin "how to answer" layer on top of "knowledge from pretraining"; bar fills to 43% | Published result |
| 03 | FLAN (GIF) | 60 task tiles flow into a 137B model; 25 unseen tasks flip, 20 green | Published result; grey positions illustrative |
| 04 | The training recipe (GIF) | Collect → Format → Mask → Train → Evaluate light up with icons | Illustrative |
| 05 | One example, three forms (GIF) | chat bubbles → token line → only answer + end green | Illustrative, simplified markers |
| 06 | Three forms summary (static PNG) | Stored / What the model reads / What gets a loss, real Qwen markers | Illustrative |
| 07 | Which tokens get a loss (GIF) | Five-turn conversation; only assistant responses and their end markers turn green | Illustrative |
| 08 | Masked doesn't mean untouched (GIF) | Loss on the response flows back along attention into the prompt | Schematic |
| 09 | The one-position shift (GIF) | Arrows to the next token; forgotten shift loops back onto the same token | Schematic |
| 10 | Long answers weigh more (GIF) | 2 vs 500 squares → one loss bar, 250× | Illustrative counts |

Cover (`build_cover.py`, revised 2026-10-22 after learner review): series style (white
Avenir Next Heavy title "Instruction Tuning", gold "POST-TRAINING SERIES · 02" and
"DONGXI LABS"). The first version reused article 1's river-and-gates art and was rejected
as too similar. The new motif is one token chain: hollow grey "read" tokens dissolving
into text fragments, then glowing blue "graded" tokens, each reached by a fan of possible
next-token curves with one bright curve landing on the demonstrated token, ending at a
distinct "stop" token. Editorial metaphor, not real probabilities.
