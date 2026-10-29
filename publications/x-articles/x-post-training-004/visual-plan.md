# Visual plan — X-POST-004 RL terms for LLMs

Status: **plan for learner review** (2026-10-27). Not approved; nothing rendered. Running example
(learner choice, 2026-10-27): **Is OpenAI open?** (numbers in `plan.md`).

Planned source: `visuals/animations/projects/post-training-rl-terms/`. Candidate: `CAND-ANIM-038`
(revised by this plan). Production machine: Mac Studio.

## Learner requirements (2026-10-27)

The audience is LLM people with limited RL knowledge. The animations must be **simple, easy to
understand, nice and eye-catching, and visual > text**.

## Visual grammar

These rules apply to every GIF.

**One cast, the same in every figure.** A reader who learns an object once can read it everywhere.

| Object | Drawing | Meaning |
|---|---|---|
| User bubble | grey bubble with a person icon | the prompt (user turn) |
| Assistant bubble | solid blue bubble with a robot icon, left-aligned with the user bubble and the same width | the answer written so far |
| Candidate bubbles | light-blue bubbles in one column, same width; token on the first line, the rest of the answer below | possible continuations |
| Branches | blue curves from the end of the assistant bubble, thickness = probability | π(a\|s) |
| Next-token bars | three vertical bars, **open · closed · partly**, in blue | the policy π(·\|s) |
| Judge stamp | violet circle that lands at the end of the answer with a number | reward R |
| Value line | amber line floating above the chips, with height = V | value V(s) |
| Advantage arrows | green arrow up / red arrow down under a token, with length = \|A\| | advantage A |
| Reference bars | grey ghost bars behind the blue ones | π_ref (KL) / π_old (ratio) |

**Text budget per GIF**

- A title of at most 4 words.
- At most 6 short labels.
- At most one formula chip (pending decision 2 in `plan.md`).
- One amber caveat badge where numbers are illustrative.
- No sentences on screen.

**Motion carries the meaning**

- Each GIF is one continuous computation: sample, average, subtract, propagate or update. Objects
  keep their identity; nothing is replaced by a new slide.
- 8–12 s loops, with a 2 s hold on the final readable frame.
- It must be readable at phone size with the sound off. Tokens are drawn at 38+ and labels at 28+,
  and every frame is tested at 50% scale.

**Colour roles.** These are the Dongxi palette roles, plus one new role:

| Role | Colour |
|---|---|
| Policy / written tokens | blue `#2563EB` |
| Judge / compare | violet `#7C3AED` |
| Positive advantage / correct | green `#047857` |
| Negative advantage | **red `#B91C1C`** (new role, proposed for signed quantities only) |
| Value line and caveats | amber `#B45309` |
| Frozen / reference | grey |

**Eye-catching, kept honest**

- Large, saturated shapes on white, and a few satisfying motions: the stamp landing, bars racing,
  arrows snapping.
- No decorative motion.
- The R1-Zero numbers are published; everything else carries the badge "illustrative numbers".

## Figures

| # | Figure | Motion (storyboard) | Evidence |
|---|---|---|---|
| 01 | One score, thousands of tokens | A long, fast-scrolling strip of reasoning tokens. One ✓ stamp lands only at the very end. A bar climbs from 15.6% to 71.0% (AIME 2024 pass@1) while the strip grows longer. | DeepSeek-R1 paper's published numbers; strip length is schematic |
| 02 | The RL loop | Prompt card → LLM (agent, labeled **policy π_θ**) writes an answer → the judge (part of the **environment**) stamps a score → an update arrow goes back to the LLM. The loop runs twice, faster the second time. | Schematic |
| 03 | State and action | Prompt "Is OpenAI open?"; the answer "OpenAI is" grows chip by chip. The written prefix glows as **state**. The three bars rise as **π(·\|s)**: open 0.5, closed 0.3, partly 0.2. "open" is sampled and flies into the strip as the **action**, followed by "It's in the name." The finished strip is labeled **trajectory**. | Illustrative probabilities |
| 04 | Reward and return | A 0 drops under each chip, then the judge stamps 2 at the end (**sparse reward**). The 2 slides back under every token (**return**, γ = 1). | Illustrative |
| 05 | Value, Q, advantage | The state "OpenAI is" sits at the height of its value. Three branches (thickness = probability) lead to answer cards placed at the height of their Q. Ten samples land at their own scores: open 3, 1, 2, 1, 3; closed 5, 4, 6; partly 10, 8. All ten fall to one average, the amber line through the state: **V = 4.3**. Each branch's dots fall to their own average, written next to that branch: **Q = 2, 5, 9**. Arrows from the V line: open **−2.3** red, closed **+0.7** green, partly **+4.7** green. | Exact arithmetic on illustrative numbers (style pilot, revised 2026-10-28) |
| 06 | Baseline | The same frame as 05: the state, three branches, Q = 2, 5, 9 beside the cards and each branch's average on the axis. *Measured from 0* all three arrows point up (+2, +5, +9). The reference line rises to 4.3, the arrows are re-measured as it moves, and it lands on the state as the amber V line: −2.3, +0.7, +4.7. No sentences on screen; formula A = Q − V. Revised 2026-10-28. | Exact arithmetic |
| 07 | Credit assignment | 05's chat style: the grey question bubble and the blue assistant bubble holding "OpenAI is open. It's in the name.", with the judge's 2 at its end. An amber dashed V line above the words: V = 4.3, then V = 2 right after "open". The drop becomes a red arrow, "open" is marked red, and a pill reads **TD error −2.3**. Under the bubble, row A has one red arrow under "open" (λ = 0). The λ slider moves to 1 and the arrows spread to "OpenAI" and "is" (−2.3 each), then return. No zeros are drawn. Formula δ = r + V′ − V. Revised 2026-10-28. | Exact arithmetic; V idealized |
| 08 | Policy gradient | 05's frame in motion. Each update reshapes the branch thicknesses (50/30/20% → 15/36/49% → 4/11/85%, exact expected-gradient steps on the logits). The state and its V line rise with V (4.3 → 6.5 → 8.3), and the arrows from the V line to each Q are recomputed throughout. As V passes 5, "closed" turns red. Formula Δ log π ∝ A. Revised 2026-10-28. | Exact arithmetic; step size illustrative |
| 09 | Exploration | 05's frame with Q = ? on every card. *greedy*: three samples all run down the thickest branch (open: 3, 1, 2); only Q(open) = 2 is learned, V = 2, A = 0, and the branches only wiggle. *sampling*: 05's ten samples try every branch; Q = 2, 5, 9 appear, V = 4.3, the arrows −2.3/+0.7/+4.7 appear, and the branches move to 15/36/49%. Revised 2026-10-28. | Illustrative |
| 10 | Ratio and clipping | 05's frame with open and closed dimmed. A ρ meter on the right with the shaded window [1 − ε, 1 + ε]. The green A = +4.7 push moves π(partly) from 20% to 24% and the knob from ρ = 1.00 to 1.20; at 1 + ε the push vanishes. Formula ρ = π / π_old. Revised 2026-10-28. | Exact window (ε = 0.2) |
| 11 | KL penalty | The chat at "OpenAI is partly." branches to a plain answer (judge 9) and a flattering one (9.6). Grey ghost branches show π_ref (95/5%) and blue branches show π, with flattery drifting 5% → 90%. Right: one stacked bar, the judge's average with β·KL cut off its top; R − β·KL falls from 9.03 to 7.16 while the judge rises only to 9.54 (β = 1). Revised 2026-10-28. | Exact arithmetic on illustrative numbers |
| 12 | Term map | The whole loop on 05's frame, each core term placed on the picture: state (chat), policy π (branches), action (the token), reward (a sample's dot), V (the line), Q (beside each card), A (arrows), then a green update arrow back to the branches, which reshape. The other terms are listed in the article's summary. Revised 2026-10-28. | Exact arithmetic on illustrative numbers |
| 13 | Term cards | The quick-reference sheet that follows 12 (learner decision, 2026-10-28): five cards built in order with arrows and a loop back: sample (policy, state, action, trajectory, exploration) → score (environment, reward, return) → compare (V, Q, A, baseline) → assign credit (TD error, GAE λ) → update (policy gradient, ratio, clipping, KL). | Summary |

**Style pilot result (2026-10-27).** Learner feedback on GIF 05:

- Ten repeated token chips were cluttered and unclear. Show each candidate answer once, and let ten
  moving dots show the samples.
- Question and answer must look like a chat: color and icons, not text labels, mark user versus
  assistant, and the bubbles are aligned.
- No outer frame around the state. The word state sits under the two bubbles; the state is the
  prompt plus the answer so far.

The approved style lives in `visuals/animations/projects/post-training-rl-terms/scenes.py`
(`bubble`, `person_icon`, `bot_icon`, `stamp`).

**Production order:** render 05 first as the **style pilot** (cast, colours, text budget, pacing)
for learner review. Produce the rest only after the pilot is approved.

Output (series standard):

- 1920×1080 30 fps MP4, git-ignored.
- 1280×720 20 fps GIF in `assets/`, encoded from a lossless qtrle render with one palette and no
  dithering, under 5 MB.
- A poster PNG and a contact sheet per clip.

## Cover (series style, new motif)

White Avenir Next Heavy title **RL Terms for LLMs**, with gold `POST-TRAINING SERIES · 04` and
`DONGXI LABS` on a deep ink background with engraved vector lines and stipple dust. The motif,
different from covers 1–3:

- **A branching tree of token paths.** One glowing prompt node at the left fans out into many thin
  engraved trajectories running right.
- Each path ends in a small score dot: green for high, violet-red for low.
- One highlighted blue path runs through the tree, with a faint amber value line above it that
  steps down where the path takes a bad branch.
- No text in the art beyond the title and labels, or at most two tiny labels ("state", "reward").
