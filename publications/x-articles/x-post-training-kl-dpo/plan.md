# Article plan — the KL-regularized objective and DPO (post-training series, slot to be decided)

Status: **started 2026-11-03 from study-path S2**, at the learner's request for animations that explain
"why RLHF maximizes reward while charging a cost for moving away from a frozen reference model, and
why and how KL works". Article number, slot and final scope are **for the learner to decide**: this may
become the series' policy-optimization article (PPO, GRPO, DPO; see `../README.md`) or its first half.
The outline grows as the learner works through S2. No drafting or rendering yet.

**Split decided (learner, 2026-11-04): the KL article comes first**:
[`../x-post-training-kl/plan.md`](../x-post-training-kl/plan.md). KL-itself material (key point 2;
sections 5–6; animations 054, 055, 037) moves there; this article opens from KL as a known ruler and
covers the objective, the optimum and DPO. The KL article previews Part C (implicit reward) and the
DPO loss in one section (its key point 7, §9; D84); this article gives the full derivation and gradient.

## Source: read the learning notes first

Before outlining or drafting, read the S2 learning note
[`learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md`](../../../learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md)
and the main preference note
[`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`](../../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md)
(RL glossary, D21, D32, D35). They record the learner's own questions in order.

| Topic | Learning-note entries |
|---|---|
| ★ Why charge a cost for moving away from a frozen reference | S2 note D56 ★ |
| ★ Why KL and not another distance; direction; estimation from samples | S2 note D57 ★ |
| How to read KL values; what true quality (gold) is; ★ the three components of a run and what labs publish | S2 note D59, D60, D61, D62 ★ |
| ★ Math vocabulary in plain language (policy, reference, E[R], KL, coefficient β, objective, optimum, `e^{R/β}`, ∝, partition function Z); the optimum in three steps | S2 note D64, D65 ★ and its **Math glossary** section |
| ★ Parts A–C as one story with an everyday example (egg question) and the math | S2 note D66, D67 ★ |
| KL penalty: definition, per-token estimators k1/k3, where PPO and GRPO put it | main note D35, Key concepts ★ "KL penalty to the reference" |
| DPO as the RL-free loss; online vs offline | main note D21, D32–D34 |
| The reward model the objective uses | X-POST-005 (published) |

Also: the S2 notebook (`notebooks/study-path/s2_kl_optimum_and_dpo.ipynb`), Chapter 11
(`book/chapters/11-direct-preference-optimization.md`) and SLP3 Ch. 8. Every claim maps to `source-map.md`.

## Key points (the reader must leave with these)

1. ★ **Why the KL cost** (D56 ★). RLHF maximizes `E[R] − β·KL(π‖π_ref)`. The reward model is reliable
   only near the answers it was trained on; reward alone drives the policy onto what it overrates and
   collapses it onto one answer. The cost keeps the policy a re-weighting of the reference:
   `π* ∝ π_ref · e^{R/β}`. Toy: an exploit with π_ref 0.001 and an overrated score gets probability 1
   with reward only, 0.009 at β = 1. The reference is frozen so the anchor cannot drift.
2. ★ **Why KL** (D57 ★). It compares answers, not weights; it is an average of log-ratios, so it can be
   estimated from the policy's own samples and split over tokens; it charges most where the reference
   almost never goes (same 0.1 move: KL 0.023 toward a common answer, 0.374 toward a rare one, total
   variation 0.10 for both); its direction lets the policy drop answers; and it gives the closed-form
   optimum from which DPO follows.

3. ★ **The three components of a run** (D61, D62 ★). Reward (frozen reward model or verifier) and KL
   (frozen reference) enter the update; the gold signal (held-out human comparisons, benchmarks,
   verifier on held-out problems, reading samples) only evaluates and is never optimized. Published lab
   setups follow this pattern; current β values are not public.
4. ★ **The math in plain language** (learner, 2026-11-03: "the above is important"; D64, D65 ★). Every
   symbol of `J(π) = E[R] − β·KL(π‖π_ref)` and `π* = π_ref·e^{R/β} / Z` defined in words with the toy
   (π_ref 0.7 / 0.299 / 0.001, R 2 / 1 / 4): E[R] as a probability-weighted average (1.703 for the
   reference), β as the price per unit of KL that you choose, the objective as one score per policy, the
   optimum as the policy with the highest score (J = 1.798, compared with five other policies), and the
   optimum built in three steps: **multiply** by `e^{R/β}` (5.172, 0.813, 0.055), **add** (Z = 6.040),
   **divide** (0.856, 0.135, 0.009). Z is just the total to divide by; for an LLM it is a sum over every
   possible answer, which is why DPO's cancellation of Z matters.
5. ★ **One story from KL to DPO, on an everyday example** (learner, 2026-11-03: "the above is important,
   mark it in note and article plan"; S2 note D67 ★). Each step told as **example → math → numbers**
   on the prompt *"How long should I boil an egg?"* with five answers: vague "until it's done"
   (π_ref 0.40, R 0), "about 10 minutes" (0.25, 1), a long paragraph (0.15, 0.5), the best answer
   "soft-boiled 6–7 minutes, hard-boiled 10–12, then cold water" (0.12, 2), wrong "2 minutes" (0.08, −1);
   β = 1. The learner rejected a technical prompt ("What is a hash table?") as the example: use an
   everyday question.
   - **A, a ruler for change:** move 0.10 from the vague to the best answer → KL 0.047 (terms −0.086 and
     +0.133); always the best answer → KL 2.12.
   - **B, the best compromise:** J = E[R] − β·KL. Don't change 0.485, small move 0.638, always the best
     answer −0.12, π\* 0.808. Multiply → add (Z = 2.243) → divide: best answer 12% → 40%, vague 40% →
     18%, wrong 8% → 1%. β dial 0.5 / 1 / 2 / 5 (best answer 0.71 / 0.40 / 0.24 / 0.16).
   - **C, read the reward back:** from the before and after models alone, r̂ = β·log(π\*/π_ref) =
     (−0.808, 0.192, −0.308, 1.192, −1.808), the true reward minus the same 0.808. Bradley–Terry uses
     differences, so P(best beats "10 minutes") = 0.731 either way and Z is never needed.
   - **D, so DPO:** train on "this answer beat that one" directly; the pair's loss goes 0.693 (before)
     → 0.313 (π\*).

**Running example.** The egg question (key point 5) is the article's main running example for the
optimum, the implicit reward and DPO (sections 4 and 8). The three-answer exploit toy (π_ref 0.7 /
0.299 / 0.001) stays in section 3, where an overrated rare answer is the point; the learner may later
choose to move sections 3–4 onto the egg question as well (e.g. a "fake confident" answer as the exploit).

## Outline (draft, grows with S2)

1. **Opening.** A policy trained on reward alone finds an answer the reward model overrates.
2. **The objective and the workflow.** `E[R] − β·KL(π‖π_ref)` in words: improve the reward, pay for moving
   away from the reference. Define π, π_ref (frozen SFT model), β. The three components, with the gold
   check outside the loop; a short "what labs publish" paragraph (verify every claim). Animation
   `CAND-ANIM-057` ★. (D56 ★, D61, D62 ★)
3. **Why the cost.** Two reasons: the reward model is unreliable far from its training data; reward
   alone collapses onto one answer. The toy table. Animation `CAND-ANIM-052`. (D56 ★)
4. ★ **The optimum is a re-weighting** (key point 4). First the words: objective, coefficient, optimum
   (D64's table of six policies and their scores). ★ Then the egg question told as A → B (key point 5:
   KL readings, the score table, multiply → add → divide). Then `π* ∝ π_ref · e^{R/β}` built in the three steps
   multiply → add (Z) → divide, with a short plain-language box for each symbol (S2 note, **Math
   glossary**); then the β dial (β = 2: Z = 2.403, π* = 0.792 / 0.205 / 0.003). No equation is used
   before its words are defined. Animation `CAND-ANIM-053`. (D64, D65 ★, S2 Part B)
5. **Why KL.** Same move, different price; log-ratios from samples; per token. Animation
   `CAND-ANIM-054`; per-token estimation `CAND-ANIM-037`. (D57 ★, D35)
6. **Direction.** KL(π‖π_ref) vs KL(π_ref‖π). Animation `CAND-ANIM-055`. (D57 ★, S2 Part A)
7. **Why frozen.** A moving anchor hides drift. Animation `CAND-ANIM-056`. (D56 ★)
8. ★ **From the optimum to DPO** (key point 5, parts C–D). On the egg question: read the reward back
   from the before and after models (r̂ = R − 0.808), the constant cancels in Bradley–Terry (0.731 both
   ways), so train on preference pairs directly; DPO loss 0.693 → 0.313. Rest written as the learner
   reaches S2 Parts D–G. (D66, D67 ★)

## Animations

Storyboards in `visuals/animations/PROPOSALS.md` (S2 set, 2026-11-03). Production on the Mac Studio.

| Section | Candidate | Shows | Source | Core |
|---|---|---|---|---|
| 2 | `CAND-ANIM-057` ★ | the workflow: reward and KL feed the update, gold only evaluates checkpoints; over-optimization | D61, D62 ★ | **core** |
| 3 | `CAND-ANIM-052` ★ | reward alone flows to the overrated exploit; reward − β·KL improves on the reference without leaving it | D56 ★ | **core** |
| 4 | `CAND-ANIM-053` | `π_ref × e^{R/β}`, renormalize; β dial from "keep the reference" to "argmax". Show the three steps with labels **multiply / add (= Z, the partition function) / divide** and the D65 numbers | D56 ★, D65 ★, S2 Part B | **core** |
| 5 | `CAND-ANIM-054` ★ | same 0.1 move: total variation 0.10 twice, KL 0.023 vs 0.374; `log(π/π_ref)` blows up as π_ref → 0; estimate from samples | D57 ★ | **core** |
| 5 | `CAND-ANIM-037` | KL at one token and over a sequence; k1/k3 from one sampled token; where PPO and GRPO put it | D35 | **core** (existing) |
| 6 | `CAND-ANIM-055` | direction: dropping an answer costs 0.357 vs ∞; adding a rare one 0.374 vs 0.103 | D57 ★, S2 Part A | optional |
| 7 | `CAND-ANIM-056` | moving anchor: every step small, total drift large; frozen anchor sees it | D56 ★ | optional |

Possible merges: 052 + 053 (the β dial as 052's last frame), 054 + 055.

## Conventions

Follow `../README.md` (post-training series conventions): English at the package root and an
independently written Chinese version in `zh/`; core technical terms in English (policy, reward, KL,
reference model, …); GIFs by default; no links in the public body; every claim in `source-map.md`.
