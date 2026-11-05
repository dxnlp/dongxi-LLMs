# Article plan — KL divergence itself (post-training series)

Status: **started 2026-11-04** at the learner's request during the S2 re-study of Part A: "I think we
need a lot of animations to explain KL itself, and maybe we need a separate X article about KL itself."
**Order decided by the learner (2026-11-04): this KL article comes first**, before the KL/DPO article
([`../x-post-training-kl-dpo/plan.md`](../x-post-training-kl-dpo/plan.md)), which then uses KL as a known
ruler and covers the objective, the optimum and DPO. Article number still to be assigned.

**Revised 2026-11-04 for the audience** (learner: "the audience might have limited math knowledge … the
article should explain KL in detail and in a easy to understand way"). English and Chinese drafts written;
twelve GIFs rendered on the learner's request (2026-11-04); title approved and cover built the same day. The sections below **Outline** and **Animations** are the current
version and replace the earlier nine-section outline.

## Audience and how the draft serves it

- The reader knows multiplication and percentages, and nothing else is assumed: no logarithms, no Σ, no
  probability notation.
- Ratio, log and weighted average each get a section, before the formula appears.
- The formula comes once, after the reader has done every step by hand, with each symbol mapped to a step.
- One running example throughout: the egg question. The three-answer toy of D57 is replaced by a sixth
  egg answer (from D79), so a reader follows one set of answers.
- New meaning section (not in the notes): KL as average evidence per reply for telling the two models
  apart, the original 1951 reading. It gives the number a size: 0.047 ≈ 64 replies to 20-to-1 odds.
- Calculus, Lagrange multipliers and the derivation of the optimum stay out; the optimum's values appear
  without derivation, and the next article derives them.

## Where to find everything (read before outlining or drafting)

The learner asked (2026-11-04) that this plan point to every learning note and animation plan behind it:
"we have discussed a lot". Every key point and outline section below cites these entries. To find an
entry, search its file for the heading, e.g. `### D68`.

### Learning notes

Files:
- **S2 note** = [`learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md`](../../../learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md)
- **Main note** = [`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`](../../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md)
- **Day 3 note** = [`learning_artifacts/day-03-probabilities-and-next-token-loss/nll-cross-entropy-and-perplexity.md`](../../../learning_artifacts/day-03-probabilities-and-next-token-loss/nll-cross-entropy-and-perplexity.md)

| Entry | File | Learner's question (short) | What it gives this article | Used in |
|---|---|---|---|---|
| D35 ★ | main note | KL to the reference, in detail (NB04 Part E) | per-token KL over the vocabulary (0.7/0.2/0.1 example, 0.0851; reverse 0.0920); sequence KL = sum over positions; k1/k3 estimators, why k1 can go negative; where PPO and GRPO put it | KP 5, §6, §7 |
| — | main note, **Key concepts** row "★ KL penalty to the reference"; plan rows **P6 ★**, **P13 ★**, **P15** | — | one-line summary; earlier animation and article requests | — |
| D57 ★ | S2 note | Why has KL been chosen? | compares distributions, not weights; estimable from samples, per token; same 0.1 move: KL 0.023 vs 0.374, total variation 0.10 both; direction (dropping 0.357 vs ∞); closed-form optimum | KP 5, §5, §6 |
| D59 | S2 note | How to check KL values; what is optimal? | true KL ≥ 0, a k1 estimate can be negative; 0 at step 0; no optimal value; choose by gold | KP 6, §8 |
| D60, D61 | S2 note | What is true quality / gold? three signals | reward and KL train, gold only evaluates | §8 |
| D67 ★ | S2 note | One story for Parts A–C with an everyday example | **the running example**: the egg question, five answers, π_ref and R; small move | all |
| D68 ★ | S2 note | Explain the KL formula in detail | symbol by symbol; term by term −0.086 + 0.133 = 0.047; why ≥ 0 (log x ≤ x − 1); zeros (dropping free, inventing ∞); as an average, from samples (10 → 0.096, 100 → 0.055, 10,000 → 0.049); per token; reverse 0.042 | KP 1, 2, 5; §2, §6, §7 |
| D69 | S2 note | Why "weights from the new model"? (correction) | the earlier intuition was wrong; correct reason: each term ≥ the probability gained | KP 2, §3 |
| D70 ★ | S2 note | Could the negative side win? Why the better direction? | lost = gained; extreme drop (−3.689 × 0.01 = −0.037, KL 0.701); spread (0.240); best going down still costs 0.076; direction comes from the reward (J 0.638 vs 0.209) | KP 2, 3; §3, §4 |
| D71 | S2 note | KL only a measure, not a control? | alone it measures; in the objective reward steers, β·KL brakes; best answer's π\* 1.00 → 0.71 → 0.40 → 0.24 → 0.16 → 0.12 as β grows | KP 3, §4 |
| D72 ★ | S2 note | Never negative because the model is either the same or different? | learner's intuition confirmed; other change measures go negative (−1.569, −0.126); not symmetric | KP 4, §5 |
| D79 | S2 note | When is training complete? | KL next to reward-model score and gold; computed toy with a "20 minutes, experts agree" answer: gold peaks at KL ≈ 0.74 | KP 6, §8 |
| D82 | S2 note | Explain Part C | invert π\*: R = β·log(π\*/π_ref) + β·log Z; implicit reward r̂ = β·log(π/π_ref) = R − 0.808 on the egg question; Z cancels in Bradley–Terry (0.731 both ways) | KP 7, §9 |
| D83 ★ | S2 note | Explain Part D | DPO loss −log σ(r̂_chosen − r̂_rejected); egg pair 0.693 (start) → 0.313 (π\*); only reference-relative changes count; (gradient and caveats: KL/DPO article) | KP 7, §9 |
| D84 | S2 note | Mention C and a small part of D in the KL article | origin of key point 7 and §9 | — |
| D80 | S2 note | Request for this article and its animations | origin of this plan and of `CAND-ANIM-058`–`064` | — |
| — | S2 note, **Key concepts** rows "★ The KL formula, symbol by symbol", "★ Why KL ≥ 0, and KL has no direction", "★ KL as 'how far from the reference'", "★ The DPO loss and its gradient" | — | one-paragraph summaries of D68–D72 and D83 ★ | — |
| — | Day 3 note | — | cross-entropy = entropy + KL: the pretraining loss already contains a KL | optional sidebar |

Background for the **next** article (KL/DPO), not this one: D56 ★ (why charge a KL cost), D58, D62 ★,
D64–D66, D73–D78 (the objective and its optimum); the full Part C derivation and Part D gradient
(D82, D83 ★) are covered there in depth, and only previewed here (§9).

### Animation plans

Storyboards: [`visuals/animations/PROPOSALS.md`](../../../visuals/animations/PROPOSALS.md). Search for the
heading, e.g. `### CAND-ANIM-058`.

| Candidate | PROPOSALS section | Status | Shows | Source entries |
|---|---|---|---|---|
| `CAND-ANIM-058` ★ | KL-itself set (058–064) | suggested | formula built term by term → 0.047 | D68 ★ |
| `CAND-ANIM-059` | KL-itself set | suggested | why a log of the ratio | D68 ★ |
| `CAND-ANIM-060` ★ | KL-itself set | suggested | never negative: lost = gained; extreme drop | D69, D70 ★ |
| `CAND-ANIM-061` | KL-itself set | suggested | ruler, not steering wheel; β knob | D70 ★, D71 |
| `CAND-ANIM-062` | KL-itself set | suggested | other change measures go negative; not symmetric | D72 ★ |
| `CAND-ANIM-063` | KL-itself set | suggested | KL from samples; one sample can be negative | D68 ★, D72 ★ |
| `CAND-ANIM-064` | KL-itself set | suggested | reading KL in a run, with reward and gold | D59, D79 |
| `CAND-ANIM-054` ★ | S2 set (052–056) | suggested | same 0.1 move, different price (0.023 vs 0.374) | D57 ★ |
| `CAND-ANIM-055` | S2 set | suggested | direction; dropping vs inventing an answer | D57 ★ |
| `CAND-ANIM-037` | Day 15 · 04 (P6 ★) | suggested | per-token KL; k1 / k3; where PPO and GRPO put it | D35 ★ |
| X-POST-004 GIF 11 | already produced: [`../x-post-training-004/assets/11-kl-penalty.gif`](../x-post-training-004/assets/11-kl-penalty.gif); scene `KlPenalty` in `visuals/animations/projects/post-training-rl-terms/scenes.py` | published (Chinese) | KL penalty in the RL-terms article (flattery judge) | X-POST-004 source map |

Production on the Mac Studio. The KL/DPO article keeps 052, 053, 056, 057.

### Other sources

- S2 notebook Part A: [`notebooks/study-path/s2_kl_optimum_and_dpo.ipynb`](../../../notebooks/study-path/s2_kl_optimum_and_dpo.ipynb)
- Day 15 · 04 Part E (origin of D35): [`notebooks/day-15/04_preference_alignment_end_to_end.ipynb`](../../../notebooks/day-15/04_preference_alignment_end_to_end.ipynb)
- Day 17 · 01: [`notebooks/day-17/01_kl_regularized_optimum.ipynb`](../../../notebooks/day-17/01_kl_regularized_optimum.ipynb); Day 20 · 02: [`notebooks/day-20/02_ppo_clipping_and_kl.ipynb`](../../../notebooks/day-20/02_ppo_clipping_and_kl.ipynb)
- Book: Chapter 11 §11.1–11.2 ([`book/chapters/11-direct-preference-optimization.md`](../../../book/chapters/11-direct-preference-optimization.md)); Chapter 12 §12.8 "KL estimates depend on direction and support"; Chapter 13 §13.4 "KL values and KL gradients"; Chapter 3 (cross-entropy and KL)
- SLP3 Ch. 8. Every claim maps to `source-map.md` when drafting.

## Running example (all numbers exact, from D67 ★ and D68–D72)

Prompt: *"How long should I boil an egg?"*

| # | Answer | π_ref | R |
|---|---|---|---|
| 0 | "Boil it until it's done." | 0.40 | 0 |
| 1 | "About 10 minutes." | 0.25 | 1 |
| 2 | a long paragraph that eventually says 10 minutes | 0.15 | 0.5 |
| 3 | "Soft-boiled 6–7 minutes, hard-boiled 10–12; then cold water so it peels easily." | 0.12 | 2 |
| 4 | "2 minutes gives a hard-boiled egg." (wrong) | 0.08 | −1 |

- Small move (0.10 from answer 0 to answer 3): π = (0.30, 0.25, 0.15, 0.22, 0.08); terms −0.086, +0.133;
  KL 0.047; reverse 0.042.
- Away from the best (0.10 from answer 3 to answer 0): KL 0.076.
- Extreme (answer 0 → 0.01, answer 3 → 0.51): KL 0.701. Spread (answer 0 −0.30; answers 1–3 +0.10 each): 0.240.
- Always answer 3: KL 2.12.
- Other measures on (small move, away): unweighted Σ log ratio (0.318, −1.569); π_ref-weighted
  (−0.042, −0.126).
- From samples of the small move: 10 → 0.096, 100 → 0.055, 10,000 → 0.049 (one seeded run).
- Parts C–D preview (D82, D83 ★): at π\* the implicit reward β·log(π\*/π_ref) = (−0.808, 0.192, −0.308,
  1.192, −1.808) = R − 0.808; P(best beats "about 10 minutes") = σ(1) = 0.731 from R or from r̂; DPO loss on
  that pair 0.693 at the start, 0.313 at π\*.
- Reading KL in a run (D79): a sixth answer "20 minutes, always, experts agree" (π_ref 0.005, reward
  model 3, gold −2); gold peaks at 1.10 at KL 0.74 while the reward-model score keeps rising.

## Key points (the reader must leave with these)

1. **KL measures how far a model moved from a starting model, on the answers it gives.**
   `KL(π‖π_ref) = Σ π(y)·log(π(y)/π_ref(y))`: ratio, log, weight by the new model, add. (D68 ★)
2. **It is never negative, and 0 only for no change.** Lost = gained; each term ≥ that answer's gain.
   A big drop has a big negative log-ratio but a tiny new probability. (D68 ★, D69, D70 ★)
3. **It has no direction of its own.** Toward or away from the best answer, both cost KL. In training the
   reward chooses the direction; β·KL limits the distance. (D70 ★, D71)
4. **The exact form matters.** Weighting by the new model makes it distance-like; other versions go
   negative. It is not symmetric. (D72 ★, D68 ★)
5. **It prices rare moves high and can be estimated from samples, token by token.** (D57 ★, D68 ★, D35 ★)
6. **There is no "good" KL value.** Read it next to reward and an independent gold signal. (D59, D79)
7. **The log-ratio inside KL is also a reward** (learner, 2026-11-04: mention Part C and a small part of
   D; D84). β·log(π/π_ref), the per-answer piece of KL, is the model's **implicit reward**: on the egg
   question it equals the true reward minus one constant (0.808), and the constant cancels when two
   answers are compared (0.731 either way). DPO trains the chat model on preference pairs with exactly
   this score: loss −log σ(r̂_chosen − r̂_rejected), 0.693 at the start, 0.313 at π\*. Kept short: one
   picture of the idea, the full derivation is in the KL/DPO article. (D82, D83 ★)

## Outline (current; 13 sections, as drafted)

| § | Section | Key point | Learning notes | GIF |
|---|---|---|---|---|
| 1 | **One question, five answers** | — | D67 ★ | 01 |
| 2 | **Step 1: the ratio** (why divide, not subtract) | 1 | D68 ★ | — |
| 3 | **Step 2: the log** (values, multiplication → addition, falls near 0) | 1 | D68 ★ | 02 |
| 4 | **Step 3: the average over the new model's replies**, then the formula | 1 | D68 ★ | 03 |
| 5 | **What the number means**: average evidence per reply | 1 | new (Kullback & Leibler 1951) | 04 |
| 6 | **Why KL is never negative**, and why the exact form matters | 2, 4 | D68 ★, D69, D70 ★, D72 ★ | 05 |
| 7 | **KL measures distance and ignores direction** (reward, β, score) | 3 | D70 ★, D71 | 06 |
| 8 | **The same move, a different price** (sixth answer) | 5 | D57 ★, D79 | 07 |
| 9 | **Which model goes on top** (not symmetric; dropping, inventing; pretraining) | 4 | D57 ★, D68 ★, Day 3 note | 08 |
| 10 | **Real models: samples and tokens** (PPO, GRPO) | 5 | D68 ★, D72 ★, D35 ★ | 09, 10 |
| 11 | **How to read KL during training** | 6 | D59, D60, D61, D79 | 11 |
| 12 | **The log-ratio is also a reward** (implicit reward, DPO in one paragraph) | 7 | D82, D83 ★, D84 | 12 |
| 13 | **Summary**; next: the KL/DPO article | — | — | — |

Number corrections made while drafting (`check_numbers.py`; `source-map.md` has the details): the toy's gold
peak is at KL 0.81, not 0.74; the reference's gold is 0.48, not 0.49; sampled KL
uses one new seeded draw, 0.096 / 0.061 / 0.048. D79 and the storyboards keep their older values until
production.

## Animations (12 GIFs, rendered 2026-11-04)

| GIF | Candidate | Shows |
|---|---|---|
| 01 two-splits | `058` ★, first half | the five shares and the 0.10 move |
| 02 why-log | `059` (now core) | ratio dial, log curve, doubling and halving |
| 03 kl-term-by-term | `058` ★, second half | columns → 0.047, formula assembles (see decision 2) |
| 04 evidence | `065` (new) | replies add evidence; 64 replies to 20-to-1 |
| 05 never-negative | `060` ★ | gains and terms; drop to 0.01 |
| 06 no-direction | `061` | 0.047 and 0.076; scores; β |
| 07 rare-moves | `054` ★, egg version | 0.031 vs 0.233 |
| 08 which-way | `055`, with `062`'s symmetry tail | 0.047 vs 0.042; dropping and inventing |
| 09 from-samples | `063` | running average; negative sample |
| 10 tokens | `037`, reduced | product → sum; where PPO and GRPO put it |
| 11 reading-a-run | `064` | reward, gold and KL, updated numbers |
| 12 implicit-reward | `066` (new) | log-ratio = reward − 0.808; 0.731 both ways |

`062`'s other measures are covered in prose. Rendered from `visuals/animations/projects/post-training-kl/`.

## Open decisions for the learner

1. **Title.** Approved 2026-11-04: *KL Divergence: One Number for How Far a Model Has Moved* / *KL：用一个数衡量模型变了多少*.
2. **Equation on screen.** Draft shows the formula assembling in GIF 03, an exception to the "no equations on
   screen" rule, because the article is about this formula.
3. **Series number.** Draft says the sixth article; RLVR's ID `X-POST-006` would shift.
4. **Section 5** (evidence reading) is new material beyond the notes.
5. **Cover motif.** Built 2026-11-04: the reference as an engraved blue distribution, the new model as a
   violet outline, and the part where the new model is higher lit in green.

## Conventions

Follow `../README.md` (post-training series conventions).
Claims marked "from memory" in the learning notes (D35's estimator details, D59, D79's practice list)
must be checked against primary sources and recorded in `source-map.md`.
