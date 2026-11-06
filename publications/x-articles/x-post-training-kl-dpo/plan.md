# Article plan — DPO, from the KL-regularized objective (post-training series, slot to be decided)

Status: **started 2026-11-03 from study-path S2**, at the learner's request for animations that explain
"why RLHF maximizes reward while charging a cost for moving away from a frozen reference model, and
why and how KL works". Article number and slot are **for the learner to decide**: this may become the
series' policy-optimization article (PPO, GRPO, DPO; see `../README.md`) or its first half. The outline
grows as the learner works through S2.

**Drafted 2026-11-06** (learner: "start work on the DPO article"): English and Chinese drafts, `check_numbers.py`,
`source-map.md`, captions and `build.py`, with paper facts checked in the primary PDFs. Working title only. No
GIFs rendered yet (image 05 reuses KL GIF 12), no cover, nothing committed or on X. Decisions made while drafting,
for the learner to review: eleven figures (the core plus `070` and `077`, since KL GIF 10 ends on PPO and GRPO);
§12 uses the shape of the notebook's Qwen3 run and none of its numbers; `CAND-ANIM-078` should do the same;
§12's real-model section was written before the learner studied Parts F–G and gradient piece 4.

**GIFs rendered 2026-11-06** (learner: "start producing the GIFs"): ten clips in
`visuals/animations/projects/post-training-dpo/`, copied to `assets/`, with KL GIF 12 reused as image 05. The
learner did not answer the figure-set and `078` questions, so the recommended options were used.

**Cover built 2026-11-06** (learner: "build the cover"): title DPO (first built as Direct Preference Optimization, changed at the learner's request); motif: a frozen
reference line, with the chosen answer rising and the rejected answer falling away from it along a sigmoid. It
says 07, which depends on the series number.

**Shortcut GIF added 2026-11-06** (learner: "we need a gif to inlusrate this 捷径"): `CAND-ANIM-079`, split at the learner's
request into two GIFs for readability (image 1 after the shortcut paragraph, image 2 after the implicit reward
paragraph). Thirteen figures in all.

**Badge removed 2026-11-06** (learner: "去掉 illustrative numbers 所有 gif"): eight clips re-rendered, including
KL GIF 12 as this set's clip 05; the KL package is unchanged. Later the same day ("toy example 之类的gif里面的标签一概
不要，只保留 DONGXILABS") the last three badges went too.

**Split decided (learner, 2026-11-04): the KL article comes first**:
[`../x-post-training-kl/plan.md`](../x-post-training-kl/plan.md). KL-itself material (key point 2;
the former sections "Why KL" and "Direction"; animations 054, 055, 037) moved there; this article opens
from KL as a known ruler and covers the objective, the optimum and DPO. The KL article previews Part C
(implicit reward) and the DPO loss in one section (its key point 7, §12 as drafted; D84); this article
gives the full derivation and gradient. The KL article was drafted on the Mac on 2026-11-04 (EN/ZH X
drafts, 12 GIFs, not published), so this article can treat KL as known.

**Revised 2026-11-06 for the audience** (learner: "similar as the KL article, the audience of this DPO
article is for LLM beginners or to whom not having a strong math knowledge, we need simple yet well
explained animations and examples and language (as the KL article) for the DPO article. make that in
the article plan, make sure the article plan can find the corresponding learning notes and animation
plan"; S2 note D99). The **Audience**, **Where to find everything**, **Outline** and **Animations**
sections below are the current version.

## Audience and how the draft serves it

Same reader as the KL article: an LLM beginner who knows multiplication and percentages and has read
(or can skim) the KL article. Nothing else is assumed: no calculus, no Σ, no probability notation, no RL.

- **Words, then numbers, then (at most) a symbol.** Every step is told as *everyday example → what
  happens → the numbers*, as in the KL article (D67 ★ pattern). A symbol appears only after the idea has
  been shown, and each symbol is mapped to the step it names.
- **One running example throughout: the egg question** (below), the same five answers and colors as the
  KL article. The three-answer exploit toy (π_ref 0.7 / 0.299 / 0.001) is **replaced** by the KL
  article's sixth egg answer, "20 minutes, always, experts agree" (π_ref 0.005, reward-model score 3,
  gold −2; D79), so the reader follows one set of answers. Exception allowed: §12 uses a real chat
  model.
- **Recap only what the KL article taught**: ratio, log, KL as "how far the model moved" in one
  paragraph (§2), with the egg numbers (KL 0.047 for the small move).
- **No calculus.** The optimum is found by the "no move helps" argument (D75, D76 ★), not Lagrange
  multipliers. Gradients are explained by the **nudge test**: change one number by 0.01 and watch the loss
  (D91, D94). "Gradient", "learning rate", "hyperparameter" get plain-language boxes (D94, D95).
- **σ (sigmoid) is introduced as Bradley–Terry from the reward-model article**: it turns a score gap into
  "the chance the first answer wins" (σ(1) = 0.731). X-POST-005 GIFs 02 and 08 can be reused.
- **Short sentences, plain words, precise statements, no empty metaphors** (series conventions,
  `../README.md`; learner's preference for technical over metaphor).
- **Animations are simple:** one idea per GIF, five answer cards, numbers a reader can check, illustrative
  values badged (the DPO set `CAND-ANIM-067`–`078` was written to these rules).
- **What stays out:** the Lagrange derivation, the equivalence-class theorem of the DPO paper, IPO and
  other variants (at most one sentence), code. The notebook's Qwen3 run is told as "what to expect and
  watch", not as an experiment report (series convention).

## Where to find everything (read before outlining or drafting)

The learner asked (2026-11-06, D99) that this plan point to every learning note and animation plan
behind it, as the KL article's plan does. Every key point and outline section below cites these entries.
To find an entry, search its file for the heading, e.g. `### D83`.

### Learning notes

Files:
- **S2 note** = [`learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md`](../../../learning_artifacts/day-18-dpo-experiment/study-path-s2-kl-optimum-and-dpo.md) (Key concepts, **Math glossary**, study log, Discussions D56 onward)
- **Main note** = [`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`](../../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md) (RL glossary D39, S1, plan table P13–P16)
- **KL article** (prerequisite) = [`../x-post-training-kl/plan.md`](../x-post-training-kl/plan.md) and its drafts [`../x-post-training-kl/x-editor-clean-body.md`](../x-post-training-kl/x-editor-clean-body.md)

**The objective and its optimum (S2 intro, Parts A–B), before the DPO discussions:**

| Entry | File | Learner's question (short) | What it gives this article | Used in |
|---|---|---|---|---|
| D56 ★ | S2 note | Why charge a cost for moving away from a frozen reference? | two reasons for the KL cost; reward alone collapses onto an overrated answer; frozen anchor | KP 1; §3, §4 |
| D58 | S2 note | Why does S2 pair KL with DPO? | DPO follows from the KL-regularized optimum; PPO/GRPO later | §2 |
| D61, D62 ★ | S2 note | The three signals; what labs publish | reward and KL train, gold only evaluates; published setups (verify) | KP 3; §2 |
| D63 | S2 note | In DPO, is true quality defined? | no gold inside DPO training either | §12 |
| D64, D65 ★ | S2 note (+ **Math glossary**) | "Optimum" and every symbol in plain language | glossary boxes; multiply → add (Z) → divide | KP 4; §5 |
| D66, D67 ★ | S2 note | Parts A–C as one story, everyday example | **the running example** (egg question); example → math → numbers | KP 5; all |
| D71 | S2 note | KL only a measure? | reward steers, β·KL limits the distance; best answer's π\* by β | §2, §5 |
| D73 | S2 note | Part B formulas in detail | J, net value, π\*, J = β log Z − β·KL(π‖π\*) | §5 |
| D74, D75 | S2 note | What the star means; J one piece at a time; "net value", "all equal" | J table (0.485, 0.638, −0.12, 0.808); net values (0.288, 1, 0.5, 1.394, −1) → all 0.808 | KP 4; §2, §5 |
| D76 ★ | S2 note | What is Z? From "no move helps" to π\* | Z = 2.243 = the total to divide by; π\* = (0.178, 0.303, 0.110, 0.395, 0.013) | KP 4; §5 |
| D77, D78 | S2 note | One prompt? Many prompts? | π\* is the target on paper; training moves shared weights for all prompts | §5 (one sentence) |
| D79 | S2 note | When is training done? | gold peaks while the reward-model score keeps rising; sixth egg answer | §3, §12 |

**DPO (S2 Parts C–G):** every entry D63, D82–D99 is indexed in **DPO discussions** below (with ★ for
D83, D87, D98).

**Key concepts rows (S2 note), one-paragraph summaries:** "★ Why the KL cost to a frozen reference",
"★ Three components of an RLHF run", "★ Math vocabulary of the objective", "★ Parts A–C as one story",
"★ From 'no move helps' to π\* = π_ref·e^{R/β}/Z, and what Z is", "★ The DPO loss and its gradient",
"★ Implicit reward → DPO: the link that opens the KL/DPO article", "★ The D → G story".

**Main note:** D21, D32–D34 (DPO as the RL-free loss; online vs offline), D35 ★ (KL penalty; k1), D39
(RL glossary), D47 (Bradley–Terry as a softmax over two answers); plan rows **P13 ★** (this article),
**P14 ★** (workflow animation), **P16** (beginner audience and the DPO animation set, D99).

### Animation plans

Storyboards: [`visuals/animations/PROPOSALS.md`](../../../visuals/animations/PROPOSALS.md). Search for the
heading, e.g. `### CAND-ANIM-067`. Sections: "S2 set" (052–057), "KL-itself set" (058–066, produced for
the KL article), "**DPO set**" (067–078, for this article).

| Candidate | PROPOSALS section | Status | Shows | Source entries |
|---|---|---|---|---|
| `CAND-ANIM-067` ★ | DPO set | suggested | your language model is secretly a reward model | D87 ★, D90 |
| `CAND-ANIM-057` ★ | S2 set | suggested | the workflow: reward and KL train, gold only evaluates | D61, D62 ★ |
| `CAND-ANIM-068` | DPO set | suggested | one score per model: reward minus distance | D73, D75 |
| `CAND-ANIM-052` ★ | S2 set | suggested (move to the egg question with the sixth answer) | reward alone vs reward − β·KL | D56 ★, D79 |
| `CAND-ANIM-056` | S2 set | suggested | why the reference is frozen | D56 ★ |
| `CAND-ANIM-069` ★ | DPO set | suggested | "no move helps": net values level at 0.808 | D75, D76 ★ |
| `CAND-ANIM-053` | S2 set | suggested (move to the egg question: Z = 2.243) | multiply / add / divide; β dial | D65 ★, D76 ★ |
| `CAND-ANIM-066` | KL-itself set | **produced** as KL GIF 12 [`../x-post-training-kl/assets/12-implicit-reward.gif`](../x-post-training-kl/assets/12-implicit-reward.gif) | log-ratio = reward − 0.808; 0.731 both ways | D82 |
| `CAND-ANIM-070` | DPO set | suggested | the DPO loss on one pair | D83 ★, D85 |
| `CAND-ANIM-071` ★ | DPO set | suggested | the push σ(−m): wrong pairs hard, won pairs barely | D91, D92 |
| `CAND-ANIM-072` | DPO set | suggested | learning rate is speed, β is how far | D93, D95 |
| `CAND-ANIM-073` ★ | DPO set | suggested | where the probability comes from | D96 |
| `CAND-ANIM-074` ★ | DPO set | suggested | the ideal test: π\* from "who won" alone | D97, D98 ★ |
| `CAND-ANIM-075` | DPO set | suggested | too few pairs: the push never stops | D98 ★ |
| `CAND-ANIM-076` | DPO set | suggested | β sets the distance, not the gap | D98 ★ |
| `CAND-ANIM-077` | DPO set | suggested | from answers to tokens | D98 ★ |
| `CAND-ANIM-078` ★ | DPO set | suggested | what to watch on a real model | D98 ★ |
| X-POST-005 GIFs 02, 03, 08 | produced: [`../x-post-training-005/assets/`](../x-post-training-005/assets/) | published (Chinese) | Bradley–Terry; gradient push on a reward model; pair softmax | X-POST-005 source map |
| KL GIF 10 | produced: [`../x-post-training-kl/assets/10-tokens.gif`](../x-post-training-kl/assets/10-tokens.gif) | drafted | product → sum over tokens | D68 ★ |

Production on the Mac Studio, Dongxi visual system, same five answer cards and colors as the KL GIFs.

### Other sources

- S2 notebook Parts B–G: [`notebooks/study-path/s2_kl_optimum_and_dpo.ipynb`](../../../notebooks/study-path/s2_kl_optimum_and_dpo.ipynb) (toy constants `TOY_REF`, `TOY_REWARD`, `TOY_BETA` in `src/dongxi_llms/study_path/s2_kl_optimum_and_dpo.py` = the egg question)
- DPO paper: Rafailov et al. 2023, arXiv 2305.18290v3 (§3 Eq. 3, §4 Eq. 4–7, Appendix B hyperparameters; quotes checked in D89, D95)
- Book: Chapter 11 ([`book/chapters/11-direct-preference-optimization.md`](../../../book/chapters/11-direct-preference-optimization.md), §11.4 likelihood displacement); SLP3 Ch. 8
- X-POST-005 (reward models) for Bradley–Terry and the score head. Every claim maps to `source-map.md`.

## DPO discussions (recorded as they happen)

Recording rule (learner, 2026-11-05, D86): every DPO discussion from S2 Parts C–G is written into the
S2 note as it happens **and** gets a row here, so the article can find all of them. Items the learner
marks important (★) also become key points below.

| D | Part | Question | What it gives the article |
|---|---|---|---|
| D63 | intro | In DPO, is true quality defined? | No: DPO, like RLHF, has no gold signal inside training; gold only evaluates |
| D82 | C | Explain Part C | Implicit reward r̂ = β log(π\*/π_ref) = R − β log Z; Z cancels in pairs (egg: R − 0.808, 0.731 both ways) |
| D83 ★ | D | Explain Part D | Margin, loss −log σ(m) (egg 0.693 → 0.435 → 0.313); notebook's three pairs (mean 0.863); gradient weight σ(−m) (0.953 / 0.5 / 0.269 / 0.047); caveats: no finite minimum, chosen log-prob can fall, reference detached |
| D84 | D | Mention C and part of D in the KL article | KL article §9 previews this; this article gives the full version |
| D85 | D–G | Orientation to the DPO parts | Map D loss → E tabular check → F sequence log-prob → G real model; the `[B, 2]` input layout on the egg pair; why `logsigmoid` (σ(−200) rounds to 0 in float32) |
| D86 | D–G | Record DPO discussions in note and plan | This table |
| D87 ★ | C–D | Connect the implicit reward to DPO at the beginning of the article | Key point 0 ★ and §1: open with r̂ = β·log(π_θ/π_ref) as DPO's score, then explain why |
| D88 | B–D | Is DPO derived from KL or invented independently? | Derived from E[R] − β·KL plus Bradley–Terry; a table of where each loss piece comes from (β, π_ref, log-ratio from the KL; σ and pairs from Bradley–Terry); β → 0 leaves nothing to invert; no KL computed in DPO training; history: closed form cited in the DPO paper to Peters & Schaal 2007, Peng 2019, Korbak 2022, Go 2023; f-DPO still to verify |
| D89 | B–D | In the original paper, is DPO derived from RLHF? | Yes, checked in arXiv 2305.18290v3: §3 RLHF pipeline and Eq. 3 objective; §4 "We start with the same RL objective as prior work" → Eq. 4 optimum (prior work) → Eq. 5 reward from policy → Eq. 6 Z cancels → Eq. 7 loss; abstract "solve the standard RLHF problem with only a simple classification loss". Usable quotes for the opening (key point 0) |
| D90 | C–D | Use "Your Language Model is Secretly a Reward Model" in the article | Opening line and hook of §1 (quote block, attributed to the DPO paper); title candidates below |
| D91 | D | The gradient fully, plain language | Piece 1: gradient = loss change per small nudge; L(m) slope −σ(−m) table (−0.953 / −0.5 / −0.269 / −0.047); descent steps from m = 0 (0.693 → 0.219 in 4 steps, steps shrink). Pieces 2–5 to come |
| D92 | D | Gradient piece 2 | Chain rule: ∂L/∂log π(chosen) = −β·σ(−m), ∂L/∂log π(rejected) = +β·σ(−m); reference frozen; egg nudge table and one step (m 0.606 → 1.312, loss 0.435 → 0.238); β enters twice |
| D93 | D | Do β and learning rate together affect the gradient? | β is in the gradient, lr only in the step; equal lr·β gives the same first step, then larger β stops sooner (d after 100 steps 4.589 / 2.651 / 1.503 for β 0.5 / 1 / 2); "learning rate = speed, β = how far the pair must move"; Adam cancels β's speed effect |
| D94 | D | Is a derivative a gradient? | Derivative = one number; gradient = list of partial derivatives, one per input (m: 1 number; (a, b): 2; Qwen3-0.6B: ~0.6 billion). Plain-language box for the article |
| D95 | D | β and learning rate are hyperparameters | Hyperparameter (chosen, never trained) vs parameter (weights, changed by the gradient); β belongs to the objective, lr to the optimizer; DPO paper Appendix B: β = 0.1 (0.5 for TL;DR), RMSprop, lr 1e-6, 150-step warmup, batch 64 (checked) |
| D96 | D | Gradient piece 3: where the probability comes from | Softmax over 5 logits; only the pair's two logits get a gradient (0, +0.5, 0, −0.5, 0); one step: best 0.12 → 0.202, 10 min 0.25 → 0.155, answers outside the pair rise slightly (wrong 0.080 → 0.082); same margin as π\* but not π\*: DPO needs pairs covering all answers. Animation candidate (not yet proposed) |
| D97 | E | What does tabular DPO mean? | Policy = a table of one logit per answer (no network), so it can represent any distribution, the condition for DPO → π\*; the notebook toy is the egg question; pairs drawn uniformly, winner by σ(R_i − R_j); DPO sees only winner/loser indices; 1,000 pairs → TV 0.05 from π\* (from 0.33) |
| D98 ★ | D–G | The coherent story from Part D to G | Rule (D) → ideal test (E: reaches π\*, 1/√N, one-sided pairs overshoot, β sets distance not margin) → adaptor (F: sequence log-prob) → real test (G: 50% → 67.2% held-out, 97.3% train, chosen/rejected both rose then split, KL 1.61 vs 0.82 at β = 0.5). Key point 6 ★ and outline §8–§12 (learner, 2026-11-06: "the above is important") |
| D99 | all | Beginner audience, simple animations, examples and language, as in the KL article; plan must find all notes and animation plans | **Audience** section, **Where to find everything**, the DPO animation set `CAND-ANIM-067`–`078`, outline rewritten for beginners (learner, 2026-11-06) |

## Key points (the reader must leave with these)

0. ★ **The implicit reward is what DPO trains; say it first** (learner, 2026-11-05: "I think it is
   important to connect the implicit reward to dpo at the beginning of the article"; S2 note D87 ★).
   The article opens where the KL article ends (its §9, "the log-ratio is also a reward"): DPO scores an
   answer by the model's own change from the reference, r̂ = β·(log π_θ − log π_ref), and applies the
   reward-model Bradley–Terry loss to that score (the DPO paper's subtitle says it: **"Your Language Model
   is Secretly a Reward Model"**, learner 2026-11-05, D90), m = r̂(chosen) − r̂(rejected), loss −log σ(m). No
   separate reward model, no sampling loop. Egg pair: 0.693 at the start, 0.313 at π\*, where r̂ = R − 0.808
   and P(best wins) = 0.731 as with the true rewards. Every later section answers "why does this work?".
1. ★ **Why the KL cost** (D56 ★). RLHF maximizes `E[R] − β·KL(π‖π_ref)`. The reward model is reliable
   only near the answers it was trained on; reward alone drives the policy onto what it overrates and
   collapses it onto one answer. The cost keeps the policy a re-weighting of the reference:
   `π* ∝ π_ref · e^{R/β}`. Toy: an exploit with π_ref 0.001 and an overrated score gets probability 1
   with reward only, 0.009 at β = 1. The reference is frozen so the anchor cannot drift.
2. ★ **Why KL** (D57 ★). *Now covered in the KL article; here only a one-paragraph recap in §2.* It compares answers, not weights; it is an average of log-ratios, so it can be
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
   possible answer, which is why DPO's cancellation of Z matters. *For this audience the article tells it
   on the egg question, not the three-answer toy:* multiply by e^{R} (0.40, 0.680, 0.247, 0.887, 0.029),
   add (Z = 2.243), divide (0.178, 0.303, 0.110, 0.395, 0.013); found by "no move helps" (D75, D76 ★).
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
     → 0.313 (π\*). Each pair is pushed with weight σ(−m): hard for pairs in the wrong order, little for
     pairs already right (D83 ★).

6. ★ **From the rule to a real model: D → G as one story** (learner, 2026-11-06: "the above is
   important, mark it in note and in article plan"; S2 note D98 ★). Each step answers the question the
   previous one leaves open.
   - **The rule (D):** r̂ = β·log(π_θ/π_ref), margin, loss −log σ(m), push σ(−m); egg pair 0.693 → 0.313.
     *Does following it reach π\*?*
   - **The ideal test (E):** a table policy on the egg question, pairs judged by Bradley–Terry on the true
     reward, DPO sees only who won → it reaches π\* (1,000 pairs: 0.05 away; reference 0.33; ≈ 1/√N).
     Warnings: few pairs → a pair type seen one way is pushed without limit; β sets how far the model
     moves (KL 0.02 at β = 4, 2.12 at β = 0.1), not the margin it ends at (1.065). *A real model has no list
     of answers: what is log π of a reply?*
   - **The adaptor (F):** sum of the reply's token log-probs (answer tokens and the end-of-turn token,
     never the prompt); tens of nats, mostly length, so compare with the reference, not raw. *Does the
     rule survive shared weights?*
   - **The real test (G):** Qwen3-0.6B, 512 HH-RLHF pairs, β = 0.1, 128 updates: starts exactly at loss
     log 2 and 50%; held-out 67.2% ± 2.9%, training 97.3% (overfits small data, as E warned); chosen and
     rejected rose together, then split; with a 5× learning rate both fell (likelihood displacement);
     the KL is not in the loss, so measure it from samples (1.61 nats; 0.82 at β = 0.5).
   - **One line:** D writes the rule, E shows it reaches π\* when everything is ideal, F connects it to
     real text, G shows what holds up on a real model.
   Notebook reference-run numbers; the Qwen3 results are one seed each (the notebook states its evidence
   limits) and are illustrations, not general measurements: no detailed reports of our own experiments in
   the public body (series convention), so G is told as "what to expect and watch", citing the notebook only
   in `source-map.md`.

## Running example (all numbers exact unless marked; from D67 ★, D73–D98)

Prompt: *"How long should I boil an egg?"* β = 1.

| # | Answer | π_ref | R |
|---|---|---|---|
| 0 | "Boil it until it's done." | 0.40 | 0 |
| 1 | "About 10 minutes." | 0.25 | 1 |
| 2 | a long paragraph that eventually says 10 minutes | 0.15 | 0.5 |
| 3 | "Soft-boiled 6–7 minutes, hard-boiled 10–12; then cold water so it peels easily." | 0.12 | 2 |
| 4 | "2 minutes gives a hard-boiled egg." (wrong) | 0.08 | −1 |
| (5) | "20 minutes, always, experts agree" (§3 only; KL article's sixth answer) | 0.005 | reward model 3, gold −2 |

- **Recap (KL article):** small move (0.30, 0.25, 0.15, 0.22, 0.08), KL 0.047.
- **Scores J = average reward − β·KL:** reference 0.485; small move 0.685 − 0.047 = 0.638; always answer 3:
  2 − 2.12 = −0.12; π\* 0.808 (D73, D75).
- **No move helps:** small-move net values (0.288, 1, 0.5, 1.394, −1); at π\* all 0.808 (D75).
- **Multiply / add / divide:** weights (0.40, 0.680, 0.247, 0.887, 0.029), Z = 2.243, π\* = (0.178,
  0.303, 0.110, 0.395, 0.013); β dial, best answer 0.71 / 0.40 / 0.24 / 0.16 at β = 0.5 / 1 / 2 / 5 (D76 ★, D71).
- **Read the reward back:** β·log(π\*/π_ref) = (−0.808, 0.192, −0.308, 1.192, −1.808) = R − 0.808; pair best
  vs "10 minutes": σ(1) = 0.731 either way (D82).
- **DPO loss on that pair:** 0.693 at the start, 0.435 at the small move (gap 0.606), 0.313 at π\* (D83 ★, D85).
- **Push σ(−m):** 0.953 / 0.5 / 0.269 / 0.047 at gap −3 / 0 / 1 / 3; one pair trained with learning rate 1:
  loss 0.693 → 0.474 → 0.348 → 0.270 → 0.219 (D91, D92).
- **Learning rate vs β** (same first step): distance after 100 steps 4.589 / 2.651 / 1.503 at β 0.5 / 1 / 2 (D93).
- **Where the probability comes from**, one step from the reference: (0.408, 0.155, 0.153, 0.202, 0.082);
  wrong answer 0.080 → 0.082 (D96).
- **Ideal test** (notebook Part E, one seed): 1,000 pairs → distance 0.05 from π\* (reference 0.33);
  0.38 at 10 pairs, 0.012 at 10,000; gap 1.065 for every β; KL 0.02 at β = 4 to 2.12 at β = 0.1 (D97, D98 ★).
- **Real model** (notebook Part G, one run each, illustrative): held-out 50% → 67.2%, training 97.3%;
  chosen/rejected rose, then split; larger learning rate: both fell; KL from samples 1.61 (β = 0.1) vs
  0.82 (β = 0.5) (D98 ★).

## Title candidates (for the learner to choose; none decided)

Built on the DPO paper's subtitle (D90): "Your Language Model Is Secretly a Reward Model: How DPO Works";
"DPO: The Reward Model Hidden in Your Language Model"; "From KL to DPO: Your Language Model Is Secretly a
Reward Model". If the subtitle is used in the title, attribute it to Rafailov et al. (2023) in the body.

## Outline (revised 2026-11-06 for beginners; 13 sections)

Each section: everyday example → what happens → numbers; a symbol only after its idea. "KP" = key point.

| § | Section | KP | Learning notes | Animation |
|---|---|---|---|---|
| 1 | ★ **Your language model is secretly a reward model** (opening) | 0 ★ | D87 ★, D89, D90 | `067` ★ |
| 2 | **Recap and the goal**: KL in one paragraph; one score per model, reward minus distance; the workflow, gold outside the loop | 3, 4 | D58, D61, D62 ★, D71, D73, D75; KL article | `068`, `057` ★ |
| 3 | **Why pay for distance**: the sixth egg answer the reward model overrates | 1 | D56 ★, D79 | `052` ★ (egg version) |
| 4 | **Why the reference is frozen** (short) | 1 | D56 ★ | `056` (optional) |
| 5 | ★ **The best model: "no move helps"**, then multiply → add (Z) → divide; the β dial | 4, 5 | D64, D65 ★, D74–D76 ★, D77, D78 | `069` ★, `053` |
| 6 | ★ **Read the reward back**: the log-ratio is the reward minus 0.808; comparisons cancel it; how the DPO paper derives it (Eq. 3 → 7); where each piece of the loss comes from | 0 ★, 5 | D82, D88, D89 | `066` (KL GIF 12, reuse) |
| 7 | **The DPO loss on one pair**: the gap, σ as "chance the chosen wins" (Bradley–Terry recap), loss 0.693 → 0.313 | 5, 6 ★ | D83 ★, D85 | `070`; X-POST-005 GIF 02/08 |
| 8 | ★ **The push**: nudge test, σ(−m), steps that shrink; boxes: gradient, learning rate, hyperparameter; learning rate = speed, β = how far | 6 ★ | D91–D95 | `071` ★, `072` (optional) |
| 9 | ★ **Where the probability comes from**: only the pair's answers are pushed | 6 ★ | D96 | `073` ★ |
| 10 | ★ **The ideal test**: DPO finds π\* from "who won" alone; too few pairs; β sets distance | 6 ★ | D97, D98 ★ | `074` ★, `075`, `076` (optional) |
| 11 | ★ **From answers to tokens** | 6 ★ | D98 ★ | `077`; KL GIF 10 |
| 12 | ★ **What to watch on a real model**: held-out vs training, chosen and rejected, KL measured separately | 6 ★ | D63, D79, D98 ★ | `078` ★ |
| 13 | **Summary**; next article | — | — | — |

Section notes (kept from the learner's earlier decisions):
- **§1** (D87 ★, D90): first line the DPO paper's subtitle as a quote block, "Your Language Model is
  Secretly a Reward Model" (Rafailov et al. 2023). Explain "secretly": X-POST-005's reward model had a
  separate `1024 → 1` score head; here the chat model's own log-ratio to the reference is the score. Pick
  up the KL article's last line; the egg pair (score 0 → loss 0.693; score R − 0.808 → loss 0.313, 0.731
  both ways). End with the question the rest answers: why is this log-ratio a reward at all?
- **§2**: "what labs publish" in one or two sentences, every claim verified (D62 ★ is from memory).
- **§5** (KP 4 ★): no equation before its words; glossary boxes from the S2 note's **Math glossary**.
- **§6**: the table "where each piece of the DPO loss comes from" (D88): β, π_ref, log-ratio from the
  KL; pairs and σ from Bradley–Terry; no Z from the pair difference. Paper quotes: "We start with the same
  RL objective as prior work"; "solve the standard RLHF problem with only a simple classification loss".
- **§8–§12** follow KP 6 ★ (D98 ★): each section ends with the question the next one answers.
- **§12**: gradient piece 4 (shared weights) still to be discussed with the learner; the Qwen3 numbers are
  one notebook run, told as what to expect.

Rest grows as the learner works through S2 Parts E–G (see **DPO discussions**).

## Animations (DPO set + reuse; recommended core of 9)

| § | Candidate | Core | Note |
|---|---|---|---|
| 1 | `067` ★ | **core** | new |
| 2 | `068` | optional | can merge into `057` ★'s first frame |
| 2 | `057` ★ | **core** | S2 set; learner marked ★ (P14) |
| 3 | `052` ★ | **core** | retell on the egg question with the sixth answer |
| 4 | `056` | optional | |
| 5 | `069` ★ + `053` | **core** (one GIF: level the bars, then multiply / add / divide) | `053` moves to the egg numbers |
| 6 | `066` | **core** (reuse KL GIF 12) | already produced |
| 7 | `070` | optional | X-POST-005 GIF 02 can stand in |
| 8 | `071` ★ | **core** | `072` optional |
| 9 | `073` ★ | **core** | |
| 10 | `074` ★ | **core** | `075`, `076` optional |
| 11 | `077` | optional | KL GIF 10 can stand in |
| 12 | `078` ★ | **core** | badge "one run, illustrative" |

Core = 067, 057, 052, 069+053, 066 (reuse), 071, 073, 074, 078: eight new GIFs and one reused. The
learner decides the final number (the KL article used 12).

## Conventions

Follow `../README.md` (post-training series conventions): English at the package root and an
independently written Chinese version in `zh/`; core technical terms in English (policy, reward, KL,
reference model, …); GIFs by default; no links in the public body; every claim in `source-map.md`.
