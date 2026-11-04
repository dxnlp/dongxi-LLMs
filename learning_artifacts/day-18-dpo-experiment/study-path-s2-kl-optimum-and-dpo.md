# S2 study notes: The KL-regularized optimum → DPO

- Notebook: [`notebooks/study-path/s2_kl_optimum_and_dpo.ipynb`](../../notebooks/study-path/s2_kl_optimum_and_dpo.ipynb) (learner copy in
  `learner_work/notebooks/study-path/`)
- Study path stage 2 of 7 · Chapter 11 · Day 18
- Status: in progress (opened 2026-11-03)
- Main preference note (Key concepts D1–D55, RL glossary, S1 log, plan table P1–P12):
  [learning-from-preferences.md](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md); RL terms:
  [glossary](../day-15-preference-data-and-bradley-terry/learning-from-preferences.md#-rl-terms-in-the-llm-setting-glossary-d39)

One learning note per study-path notebook from S2 on (learner, 2026-11-03). Questions raised while
working through **S2** are recorded here as they happen, tagged `[S2 · Part X]`. **Discussion numbers
continue one sequence across all notes** (D1–D55 in the main note, D56 onward in the stage notes): the
next number is one more than the highest in any of them, so a D-number points to exactly one place.
Animation and article requests also go into the main note's plan table and
`visuals/animations/PROPOSALS.md`. ★ = marked important by the learner.

## Key concepts

| Concept | Description | Status |
|---|---|---|
| ★ Why the KL cost to a frozen reference | RLHF maximizes `E[R] − β·KL(π‖π_ref)`. The cost (1) keeps the policy where the reward model was trained and is reliable, (2) keeps what the reference does well that the reward does not measure, (3) stops the optimum collapsing onto one answer. Optimum `π* ∝ π_ref·e^{R/β}`: a re-weighting of the reference. Toy: an exploit with π_ref 0.001 and an overrated R gets probability 1 with reward only, 0.009 at β = 1. Frozen, so the anchor cannot drift | **marked important by the learner** (D56 ★); learner: understands why a frozen reference is needed (2026-11-03, after D56–D58) |
| ★ Why KL (and not another distance) | compares output distributions, not weights; `E_{y~π}[log π − log π_ref]` estimated from samples and split over tokens; `log(π/π_ref)` grows without limit as π_ref → 0 (same 0.1 move: KL 0.023 toward a common answer, 0.374 toward a rare one; total variation 0.10 both); KL(π‖π_ref) lets the policy drop answers (0.357 vs ∞ in the other direction); closed-form optimum `π* ∝ π_ref·e^{R/β}` → DPO | **marked important by the learner** (D57 ★) |
| ★ Three components of an RLHF run, and what labs publish | (1) **reward** from a frozen reward model, or a verifier (RLVR), often both, plus principle/rubric-guided AI judges; (2) **KL** to the frozen SFT reference, standard in RLHF, reduced or dropped in some reasoning-RL recipes; (3) **gold signal**: held-out human comparisons, benchmarks, red-teaming, reading samples, never optimized. Published: Stiennon 2020, InstructGPT, Gao 2022, Anthropic HH and Constitutional AI, Llama 2, DeepSeek-R1; current β values and stopping rules not public (from memory, verify) | **marked important by the learner** (D61, D62 ★); animation `CAND-ANIM-057` ★ |
| ★ Math vocabulary of the objective | policy π, reference π_ref, reward R, expectation E[R] (probability-weighted average), KL, coefficient β (price per unit KL, chosen by you), objective J (one score per policy), optimum π\* (highest J), `e^{R/β}` (reward weight), ∝ (right up to one common factor), partition function Z (the total to divide by; a sum over all answers, impractical for an LLM, cancels in DPO). See **Math glossary** | **marked important by the learner** (D64, D65 ★); key point 4 and section 4 of the KL/DPO article plan |
| ★ Parts A–C as one story, example + math (egg question) | Prompt "How long should I boil an egg?", five answers (vague "until it's done" 0.40/R 0, "about 10 minutes" 0.25/1, long paragraph 0.15/0.5, best "soft 6–7, hard 10–12, then cold water" 0.12/2, wrong "2 minutes" 0.08/−1), β = 1. **A** ruler: small move (0.10 from vague to best) KL 0.047, always-best 2.12. **B** best compromise: J = E[R] − β·KL; don't change 0.485, small move 0.638, always-best −0.12, π\* 0.808; multiply → add (Z = 2.243) → divide gives best answer 12% → 40%, wrong 8% → 1%. **C** read reward back: r̂ = β·log(π\*/π_ref) = R − 0.808 for every answer; the shift cancels in Bradley–Terry (0.731 either way), so Z is never needed. **D** follows: train on preference pairs directly (DPO; loss 0.693 → 0.313) | **marked important by the learner** (D66, D67 ★); key point 5 and the running example of the KL/DPO article plan |
| ★ The KL formula, symbol by symbol | `KL(π‖π_ref) = Σ_y π(y)·log(π(y)/π_ref(y))`: for each answer y, ratio new/reference (>1 up, 1 unchanged, <1 down), log makes it signed (0 = no change), weight by how often the **new** model says it, add. Egg small move: terms −0.086 (vague answer, down) and +0.133 (best answer, up), others 0 → **0.047**. Never negative overall (log x ≤ x − 1 ⇒ −KL ≤ Σπ_ref − Σπ = 0); 0 only if nothing changed. π(y) = 0 → term 0 (dropping is free); π_ref(y) = 0 < π(y) → ∞. As an average `E_{y~π}[log π − log π_ref]`: estimated from samples (10 → 0.096, 100 → 0.055, 10,000 → 0.049), split over tokens for an LLM. Reverse direction 0.042 | **marked important by the learner** (D68 ★) |
| ★ Why KL ≥ 0, and KL has no direction | Probabilities add to 1, so **lost = gained**; each term π·log(π/π_ref) ≥ π − π_ref (the answer's gain), so lowered answers subtract at most what they lost, raised answers add at least what they gained → KL ≥ 0. The log applies to each answer's ratio and is multiplied by its **new** probability: extreme drop 0.40 → 0.01 has log-ratio −3.689 but term only −0.037 (KL 0.701). KL measures *any* change, good or bad: moving 0.10 away from the best answer still costs 0.076. The direction comes from the reward in J = E[R] − β·KL: toward best J 0.638, away from best J 0.209 (less reward and a KL cost) | **marked important by the learner** (D69, D70 ★) |
| ★ KL as "how far from the reference" | Learner's intuition (confirmed): the new model is either the same as the reference (KL = 0) or different (KL > 0); nothing is closer than identical, so no negative values. KL earns this through its weighting by the new model: other change measures go negative (egg question, moving 0.10 away from the best answer: unweighted Σ log(π/π_ref) = −1.569; π_ref-weighted −0.126; KL 0.076). Distance-like, not a true distance: not symmetric (0.047 vs 0.042); exact KL ≥ 0, a one-sample estimate can be negative | **marked important by the learner** (D72 ★) |
| ★ From "no move helps" to π\* = π_ref·e^{R/β}/Z, and what Z is | **Net value** of an answer = R(y) − β·log(π(y)/π_ref(y)): how much J rises if it gets a little more probability (reward gained minus the KL price it already pays). Small move: (0.288, 1, 0.5, 1.394, −1); moving 0.01 vague → best gains ≈ 0.01 × (1.394 − 0.288) = 0.011. **At the top no move helps ⇔ all net values equal** (= c). Solve: log(π/π_ref) = (R − c)/β → π = π_ref·e^{R/β}·(one shared number). **Z = sum of the weights** π_ref·e^{R/β} (0.40 + 0.68 + 0.247 + 0.887 + 0.029 = 2.243); the shared number is 1/Z so probabilities add to 1 (counts → percentages); c = β·log Z = 0.808. Check 0.12 × e² / 2.243 = 0.395 | **marked important by the learner** (D75, D76 ★) |
| ★ The DPO loss and its gradient | Score answers with the implicit reward r̂ = β·(log π_θ − log π_ref) (Part C; for an LLM a sum of token log-probs). Pair (chosen y_w, rejected y_l): margin m = r̂(y_w) − r̂(y_l); loss −log σ(m), the Bradley–Terry loss of S1 with the chat model as scorer. Egg pair (best over "about 10 minutes"): 0.693 at the start (m = 0), 0.313 at π\* (m = 1). Notebook pairs (β = 0.5): m = +1.0, −1.5, +0.25 → loss 0.313, 1.701, 0.576; only changes vs the reference count. Gradient raises chosen, lowers rejected, weight σ(−m): 0.953 (m = −3), 0.5, 0.269, 0.047 (m = 3). Caveats: no finite minimum; β scales the gradient; shared weights → the chosen log-prob can fall too; reference detached | **marked important by the learner** (D83 ★) |

## Animations and article

The learner (2026-11-03): "we need good animation candidates in notes and article plan for (S2) to
explain why RLHF maximizes reward while charging a cost for moving away from a frozen reference model,
and why and how KL works." Storyboards in `visuals/animations/PROPOSALS.md` (S2 set); article plan
[`publications/x-articles/x-post-training-kl-dpo/plan.md`](../../publications/x-articles/x-post-training-kl-dpo/plan.md)
(slot to be decided by the learner). Main-note plan row P13.

| Candidate | Explains | Source |
|---|---|---|
| `CAND-ANIM-052` ★ | reward alone vs reward − β·KL on the three-answer toy: the policy flows to the overrated exploit, or improves on the reference without leaving it | D56 ★ |
| `CAND-ANIM-053` | the optimum `π_ref × e^{R/β}`, renormalized; a β dial from 4 to 0 | D56 ★, Part B |
| `CAND-ANIM-054` ★ | why KL: the same 0.1 move priced by total variation (0.10, 0.10) and by KL (0.023, 0.374); `log(π/π_ref)` near π_ref → 0; estimate from samples | D57 ★ |
| `CAND-ANIM-055` | direction: KL(π‖π_ref) vs KL(π_ref‖π) when dropping or adding answers | D57 ★, Part A |
| `CAND-ANIM-056` | why frozen: a moving anchor makes every step look small while total drift is large | D56 ★ |
| `CAND-ANIM-057` ★ | the RLHF workflow with its three components: reward (frozen reward model or verifier) and KL (frozen reference) feed the update; the gold signal only evaluates checkpoints, behind a "no gradients" wall; over-optimization frame | D61, D62 ★ |
| `CAND-ANIM-037` (existing) | how KL is estimated from one sampled token (k1, k3) and where PPO and GRPO put it | main note D35 |

**KL-itself set (2026-11-04, D80).** The learner: "we need a lot of animations to explain KL itself, and
maybe a separate X article about KL itself." Plan
[`publications/x-articles/x-post-training-kl/plan.md`](../../publications/x-articles/x-post-training-kl/plan.md)
(**comes first, before the KL/DPO article**, learner 2026-11-04; the plan has a "Where to find everything" section listing every entry below and every animation); main-note plan row P15. Running example: the egg question (D67 ★).

| Candidate | Explains | Source |
|---|---|---|
| `CAND-ANIM-058` ★ | the formula built term by term: ratio → log → weight → sum = 0.047 | D68 ★ |
| `CAND-ANIM-059` | why a log of the ratio: signed, 0 = no change | D68 ★ |
| `CAND-ANIM-060` ★ | never negative: lost = gained, each term ≥ the gain; extreme drop −3.689 × 0.01 | D69, D70 ★ |
| `CAND-ANIM-061` | a ruler, not a steering wheel: toward and away from the best answer both cost; β knob | D70 ★, D71 |
| `CAND-ANIM-062` | other change measures go negative; KL not symmetric | D72 ★ |
| `CAND-ANIM-063` | KL from samples: running average → 0.047; one sample can be negative | D68 ★, D72 ★ |
| `CAND-ANIM-064` | reading KL in a run, next to reward-model score and gold; keep the gold-peak checkpoint | D59, D79 |

## ★ Math glossary: the KL-regularized objective, symbol by symbol (D65 ★)

> ★ **Marked important by the learner (2026-11-03).** The optimum in three steps (β = 1): **multiply** each
> reference probability by its reward weight `e^{R/β}` (0.7·7.389 = 5.172, 0.299·2.718 = 0.813,
> 0.001·54.6 = 0.055); **add** them: Z = 6.040; **divide** each by Z: π\* = (0.856, 0.135, 0.009).

One running toy: one prompt, three answers A, B, C; reference π_ref = (0.7, 0.299, 0.001); reward
R = (2, 1, 4).

    J(π) = E_{y~π}[R(y)] − β · KL(π ‖ π_ref)          π*(y) = π_ref(y) · e^{R(y)/β} / Z

| Symbol | Plain meaning | Toy example |
|---|---|---|
| **policy π** | the model being trained, seen as a probability split over the possible answers; the split sums to 1 | (0.856, 0.135, 0.009) |
| **reference π_ref** | a frozen copy of the starting model (usually the SFT model); never changes | (0.7, 0.299, 0.001) |
| **reward R(y)** | a number for one answer, from the reward model (or a verifier) | A 2, B 1, C 4 |
| **E_{y~π}[R]** (expectation) | the average reward if answers are drawn from π: each answer's reward × its probability, summed | reference: 0.7·2 + 0.299·1 + 0.001·4 = 1.703 |
| **log** | natural logarithm; turns ratios into differences: log(a/b) = log a − log b; log 1 = 0 | log(1/0.7) = 0.357 |
| **KL(π‖π_ref)** | how far π has moved from π_ref: Σ π(y)·log(π(y)/π_ref(y)); 0 if identical, never negative | all on A: 1·log(1/0.7) = 0.357 |
| **coefficient β** | a number you choose that sets the price of KL: J subtracts β × KL. Large β = moving is expensive (stay near the reference); small β = cheap (chase reward) | β = 1: KL 0.357 costs 0.357; β = 0.1: it costs 0.036 |
| **objective J(π)** | the one score every possible policy gets: reward gained minus price paid | reference 1.703; all on A 1.643 (D64) |
| **optimum π\*** | the policy with the highest J among all policies | (0.856, 0.135, 0.009), J = 1.798 (D64) |
| **e^{x}** (exponential) | e ≈ 2.718 raised to x; always positive; turns differences into ratios: e^{a−b} = e^a / e^b | e^2 = 7.389, e^1 = 2.718, e^4 = 54.6 |
| **e^{R/β}** (the reward weight) | how much the optimum boosts an answer: higher reward → larger factor; dividing by β makes the boost stronger when β is small | β = 1: 7.389, 2.718, 54.6; β = 2: 2.718, 1.649, 7.389 |
| **∝** (proportional to) | "equal up to one common factor": the bars have the right *relative* sizes but don't sum to 1 yet | π_ref·e^{R}: 5.172, 0.813, 0.055 (sum 6.040) |
| **partition function Z** | the sum of those unnormalized bars over **all** answers; dividing by Z makes the probabilities sum to 1. The name comes from physics; it is just the normalizer | β = 1: Z = 6.040 → 5.172/6.040 = 0.856, 0.135, 0.009; β = 2: Z = 2.403 |
| **log Z** | the best achievable J divided by β: β·log Z = J(π\*) | β = 1: log 6.040 = 1.798 = J(π\*) |

**Why Z matters in S2.** In the toy, Z is a sum of three numbers. For a language model it is a sum over
every possible answer to the prompt, far too many to compute. Any method that needed Z would be
impractical; DPO works because Z cancels when two answers to the same prompt are compared (Parts C–D).

## Knowledge gaps

- Math vocabulary of the KL-regularized objective (optimum, coefficient β, ∝, partition function Z): asked for plain explanations (D64, D65); open until the learner explains the three-step construction of π\* back.

## S2 notebook study log

| Part | Question | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|---|
| A · KL divergence | What does KL(π‖π_ref) measure, and why does its direction matter? | in progress | — | D56 ★ (intro: why charge a cost for moving away from the frozen reference), D57 ★ (why KL and not another distance); animations + article plan, main-note plan P13 → `CAND-ANIM-052`–`056`; D58 (why S2 pairs KL with DPO, before PPO/GRPO), D59 (how to read KL values: sign, size, no single optimum), D60 (what "true quality" / gold reward is, and how it is measured), D61 (three signals: reward model and KL train, gold only evaluates), D62 ★ (what frontier labs have published about this setup); workflow animation `CAND-ANIM-057` ★ (plan P14); D63 (DPO does not define true quality either), D64 ("optimum" in plain words, with the objective computed for six policies), D65 ★ (plain-language glossary of every symbol in the KL-regularized objective; three-step construction of π\*); D66 (Parts A–C as one story); D67 ★ (Parts A–C with an everyday example, "How long should I boil an egg?", and the math; key point 5 of the KL/DPO article plan); re-study of A–C begins (2026-11-04): D68 ★ (the KL formula symbol by symbol); D69 (why KL ≥ 0: corrected explanation, each term ≥ the probability gained); D70 ★ (lost = gained so the negative side cannot win; KL measures any change, direction comes from the reward); D71 (KL alone measures; in the objective, reward sets the direction and β·KL the distance); D72 ★ (learner: KL ≥ 0 because the new model is either the same or different, confirmed; other change measures can go negative); D80 (request: KL animations 058–064 and a separate KL article); D81 (KL article first; its plan indexes every note entry and animation) |
| B · The optimum | Which policy maximizes E[R] − β·KL, and what does β do to it? | in progress (re-study 2026-11-04) | — | D73 (the Part B formulas in detail: objective, equal net value → π\*, softmax form, J = β log Z − β·KL(π‖π\*), reward shift, β limits); D74 (what the star in π\* means; learner lost in D73, re-explain one piece at a time); D75 (piece 1: J; "net value" = J gained by a tiny extra probability; equal = no move helps); D76 ★ (piece 2: from "no move helps" to π\* = π_ref·e^{R/β}/Z; what Z is); D77 (one prompt solved on paper = the target; training moves shared weights toward it for all prompts); D78 (many prompts: total J = fixed − β·average KL to each prompt's π\*; trained jointly through shared weights); D79 (when training is done: keep the checkpoint where gold peaks; proxy keeps rising; warning signs) |
| C · Inverting the optimum | How does a policy define a reward, and why does the partition function drop out? | in progress (re-study 2026-11-04) | — | D82 (Part C in three steps: invert π\*, implicit reward, Z cancels in Bradley–Terry; why it leads to DPO) |
| D · The DPO loss | What does the loss push, and how hard for each pair? | in progress (re-study 2026-11-04) | — | D83 ★ (Part D in four steps: implicit-reward margin, loss −log σ(m), notebook's three pairs, gradient weight σ(−m); caveats); D84 (C and part of D previewed in the KL article) |
| E · Tabular DPO | Does DPO on sampled pairs recover π*? What do the number of pairs and β change? | not started | — | — |
| F · Sequence log-probs (Qwen3) | Which tokens make up log π(y \| x) for a real answer? | not started | — | — |
| G · DPO on Qwen3 | After a few hundred real pairs: held-out accuracy, chosen vs rejected likelihood, KL to the reference | not started | — | — |

## Discussions

### D56 · ★ [S2 · intro] "I understand maximizing the reward, but why charge a cost for moving away from a frozen reference model?" (2026-11-03)

> ★ **Marked important by the learner (2026-11-03).** The KL cost keeps the policy where the reward model
> is reliable and keeps what the reference already does well; without it the optimum puts all
> probability on the single highest-scoring answer, including an overrated one.

**Setup.** RLHF maximizes `E[R] − β·KL(π‖π_ref)`. `π` is the policy being trained; `π_ref` is a frozen copy
of the starting model (usually the SFT model); `KL(π‖π_ref) = Σ_y π(y)·log(π(y)/π_ref(y))` is 0 when the
two are equal and grows when `π` puts probability where `π_ref` put little. `β` sets the price.

**Reason 1: the reward model is only reliable near the reference.** It was trained on answers like the
ones the reference writes. On answers the reference almost never writes, its scores are untested and can
be wrong (length bias, D50 ★ in the main note; reward hacking, P2). Maximizing reward alone moves the policy toward
exactly those errors, because an overrated answer has the highest score.

**Reason 2: without the cost the optimum collapses to one answer.** `max E[R]` alone puts probability 1
on the single highest-scoring answer: no diversity, and everything the reference did well that the reward
model doesn't measure (fluency, knowledge, format) is unprotected.

**Toy example** (one prompt, three answers): A good (`π_ref` 0.7, R = 2), B fine (0.299, R = 1), C an
exploit the reference almost never writes (0.001), overrated by the reward model (R = 4). The optimum of
the KL-regularized objective is `π*(y) ∝ π_ref(y)·e^{R(y)/β}` (S2 Part B):

| β | π*(A, B, C) | E[R] | KL |
|---|---|---|---|
| none (reward only) | (0, 0, 1) | 4.00 | 6.91 (= log 1000) |
| 0.25 | (0.19, 0.00, 0.81) | 3.62 | 5.16 |
| 1 | (0.86, 0.14, 0.01) | 1.88 | 0.09 |
| 4 | (0.75, 0.25, 0.00) | 1.75 | 0.01 |

Reference alone: E[R] = 1.70. With β = 1 the policy shifts toward A (the best answer the reference already
writes) and gives C 0.9%; reward alone puts everything on C. The KL cost turns "replace the model" into
"re-weight the reference's answers by their reward".

**Why frozen.** The cost needs a fixed anchor. If `π_ref` moved with `π`, KL would stay near 0 at every
step and the policy could drift step by step without limit.

### D57 · ★ [S2 · intro] "Why has KL been chosen for this task?" (2026-11-03)

> ★ **Marked important by the learner (2026-11-03).** KL compares answers, can be estimated from the
> policy's own samples (per token), charges most where the reference almost never goes, lets the policy
> drop answers, and gives the closed-form optimum behind DPO. Key point 2 of the KL/DPO article plan.

KL is one of several possible distances between π and π_ref; it is chosen because it has five properties
the task needs at once.

1. **It compares behaviour, not weights.** It is a distance between the output distributions. Two weight
   settings can differ a lot and answer the same way; what has to be controlled is the answers.
2. **It can be estimated from the policy's own samples.** `KL(π‖π_ref) = E_{y~π}[log π(y) − log π_ref(y)]`:
   sample an answer, compute two log-probabilities, subtract. RL already samples from π and computes
   `log π`; the reference adds one forward pass. It also splits over tokens (chain rule):
   `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, so the sequence KL is a sum of per-token log-ratios, and PPO can
   charge it token by token (D35). Total variation `½ Σ_y |π(y) − π_ref(y)|` or Jensen–Shannon need sums
   over all possible answers, which cannot be enumerated or easily estimated from samples.
3. **It charges most where the reward model is least reliable.** The per-answer term is `log(π/π_ref)`,
   which grows without bound as π_ref → 0. Toy (π_ref = 0.7, 0.299, 0.001 for A, B, C), two policies
   that each move 0.1 of probability:

   | Policy | moves 0.1 from A to | total variation | KL(π‖π_ref) |
   |---|---|---|---|
   | X | B (common) | 0.10 | 0.023 |
   | Y | C (rare, π_ref 0.001) | 0.10 | 0.374 |

   Total variation sees no difference; KL charges 16× more for moving toward the answer the reference
   almost never writes, which is where an overrated reward is most likely (D56 ★).
4. **The direction KL(π‖π_ref) allows dropping answers.** A policy that keeps only A (π = 1, 0, 0) pays
   0.357; the other direction, KL(π_ref‖π), is infinite for it, because it requires π to keep every
   answer the reference gives, including bad ones (S2 Part A).
5. **It gives a closed-form optimum.** `max E[R] − β·KL(π‖π_ref)` is solved by `π* ∝ π_ref·e^{R/β}`
   (S2 Part B). Inverting it gives the reward in terms of π (Part C), and the normalizer cancels in a
   Bradley–Terry difference: that is DPO (Part D). Other divergences give other optima and other losses
   (f-divergence variants of DPO exist), not this clean form.

Also: the optimum can be read as Bayesian updating, prior π_ref times likelihood `e^{R/β}` (Korbak et
al. 2022; cited from memory, verify before reuse), and log-ratios are the same quantities as next-token
cross-entropy, so the same code computes them. KL is a choice, not a law: entropy bonuses, PPO's clipping
(a limit relative to the previous policy, not the reference) and other divergences are also used.

### D58 · [S2 · intro] "Why does the S series start with KL + DPO, not KL + others? Are KL and DPO more related?" (2026-11-03)

**KL is not specific to DPO.** PPO and GRPO maximize the same `E[R] − β·KL(π‖π_ref)` (D35). S2 studies the
objective first because every later stage uses it, and DPO comes with it for two reasons.

**1. DPO is derived from the objective; PPO and GRPO only include it.** The objective has an exact
solution `π* ∝ π_ref·e^{R/β}` (Part B). Solve it for the reward: `R = β·log(π*/π_ref) + β·log Z` (Part C).
Put that into S1's Bradley–Terry loss: `Z` cancels in the difference of two answers, leaving
`−log σ(β·[log π/π_ref (chosen) − log π/π_ref (rejected)])`, the DPO loss (Part D). Without the KL term
there is no such solution (the optimum is the argmax of R) and no DPO. In PPO and GRPO the KL is one
term added to a reward or a loss, and the method works the same way with β = 0. In that sense KL and DPO
are more tightly related: the KL term *is* DPO's β and its reference log-ratio.

**2. Dependencies: DPO needs only S1 and KL.** DPO is supervised: fixed labeled pairs, a loss, backprop,
no sampling during training. PPO and GRPO estimate the gradient of expected reward from sampled answers,
which needs the policy-gradient machinery of S3 (REINFORCE, baselines, advantages, value model, GAE,
clipping); GRPO (S4) is PPO without the value model. So the order is S1 (Bradley–Terry) → S2 (objective,
its exact optimum, DPO) → S3 (policy gradient → PPO) → S4 (GRPO).

**3. The exact optimum is a reference point for later stages.** On a toy where `π*` can be computed, later
stages can check what a sampling method converges to (S7 Part A computes proxy and gold reward at the
exact optimum as β falls).

Recorded in the main note's study-path table (D40): "DPO's derivation needs only the KL objective (D35),
not PPO"; DPO was moved before PPO for that reason.

**Learner's response (2026-11-03, after D56–D58):** "now I understand better why we need a reference
policy (frozen policy) for KL to compare." Stated understanding, not yet tested; the S2 Part A exercises
and predictions are the check.

### D59 · [S2 · intro] "How should we check KL values? Positive, negative, bigger or smaller? What is the optimal value?" (2026-11-03)

**Sign.** The exact `KL(π‖π_ref) ≥ 0`, and `= 0` only when the two policies are identical. A *sample estimate*
can be negative: the per-token k1 `= log π − log π_ref` is negative whenever the sampled token is more likely
under the reference (D35). Averaged over a batch it should be ≥ 0 up to noise; k3 `= e^d − d − 1` (with
`d = log π_ref − log π`) is ≥ 0 for every sample. A batch mean that stays clearly negative points to a bug:
answers sampled with different settings than the log-probs assume (temperature, top-p), wrong masks
(prompt or padding tokens counted), or the two models seeing different tokens.

**Sanity check at step 0.** Policy = reference, so KL = 0 exactly (Day 15 · 04 asserts this: same
weights, same arithmetic).

**Size: there is no single optimal value.** KL is the *budget spent*, not a target. Too small: the policy
barely changed (β too high, or the learning rate too low). Too large: the policy is far from where the
reward model was trained, and the reward it reports may be wrong. What matters is the **true quality
gained per unit of KL**, which must be measured with something other than the training reward.

**Toy** (D56's three answers; true quality A 2, B 1, C −1, the reward model says C 4):

| β | π*(A, B, C) | reward-model score | true quality | KL |
|---|---|---|---|---|
| 100 | (0.70, 0.30, 0.00) | 1.71 | 1.70 | 0.00 |
| 1 | (0.86, 0.14, 0.01) | 1.88 | **1.84** | 0.09 |
| 0.5 | (0.88, 0.05, 0.07) | 2.09 | 1.74 | 0.40 |
| 0.35 | (0.69, 0.02, 0.30) | 2.58 | 1.09 | 1.63 |
| 0.25 | (0.19, 0.00, 0.81) | 3.62 | −0.43 | 5.16 |

The reward-model score rises with KL all the way; true quality peaks at a moderate KL (0.09 here) and
then falls: **over-optimization** (S7 Part A computes the same curve; Gao et al. 2022 measured it for real
reward models).

**How to check in practice.**
1. Sign and step-0 checks above.
2. Units: per token or summed per answer (sequence KL grows with answer length); compare runs only in the
   same units on the same prompts.
3. Trend: KL should rise gradually; a sudden jump means an unstable update (learning rate, clipping).
4. Plot training reward against KL, and evaluate checkpoints at several KL levels with an independent
   signal (held-out human labels, a separate reward model, a verifier, reading samples). Pick the KL where
   that independent quality peaks. Some implementations set a target KL and adjust β to hold it (adaptive
   KL controller, Ziegler et al. 2019; cited from memory).
5. Our runs, as scale examples, not targets: Day 15 · 04 PPO ≈ 8 nats per answer (last 5 iterations),
   GRPO ≈ 0.03.

### D60 · [S2 · intro] "True quality? What is that? Is there a concrete calculation?" (2026-11-03)

**Definition.** True quality (in papers: the **gold reward**) is how good an answer really is by the
standard we want to train toward, usually the judgment of the people the model serves. The reward model
is a **proxy**: a network trained on a limited set of labels to *predict* that judgment (S1). The two agree
near the training data and can disagree elsewhere.

**In D59's toy it is authored.** The numbers A 2, B 1, C −1 were chosen to show the effect; in real
training there is no formula for true quality, which is why a proxy is trained at all.

**Concrete ways to measure it (each an approximation):**

| Gold signal | Calculation | Example |
|---|---|---|
| Fresh human comparison (win rate) | show people the new policy's answer and the reference's answer to the same held-out prompt; win rate = wins / prompts, standard error `√(p(1−p)/n)` | 118 of 200 preferred → 59% ± 3.5% (> 50% by about 2.6 SE) |
| Verifier (checkable tasks) | 1 if the answer is correct (math answer matches, code passes tests), else 0; pass rate on held-out problems | 310 of 500 correct → 62% ± 2.2% (S5) |
| Independent reward model | score with a reward model not used in training, often larger or trained on other labels | Gao et al. 2022: a 6B reward model plays "gold", a smaller one trained on its labels is the proxy |
| Formula (simulation) | define gold and proxy directly | S7 Part A: gold `g = h − 0.5(z² − 1)`, proxy `r = h + b·z` (h helpfulness, z length); `E[g] = t − a·b²·t²`, `E[r] = t(1 + b²)` with `t = 1/β`: proxy rises linearly, gold is a downward parabola |

Rule: the gold signal must not be the reward used for training; otherwise it cannot show
over-optimization. Every real gold signal is itself imperfect (human noise, verifier gaps), so it is
"more trusted", not perfect.

### D61 · [S2 · intro] "So: a reward signal from the trained reward model, a KL divergence, and a true reward (gold signal) model?" (2026-11-03)

**Learner's model:** three signals. **Correction:** yes, three, but in two different roles; and the gold
signal is often not a model.

| Signal | Comes from | Role | Used in the update? |
|---|---|---|---|
| Reward `R(x, y)` | the frozen reward model (proxy), or a verifier in RLVR | training: what to increase | yes: the `E[R]` term |
| KL(π‖π_ref) | log π and log π_ref of the sampled tokens (frozen reference, one extra forward pass) | training: the price of moving | yes: the `−β·KL` term |
| Gold signal | held-out human comparisons, a verifier, an independent reward model, or (simulation) a formula | evaluation only: did real quality improve? | **no**: used to compare checkpoints, choose β, stop training |

Training uses only the first two: maximize `E[R] − β·KL`. The gold signal is kept out of training on
purpose: if it were optimized it would become a proxy too, and nothing would be left to detect
over-optimization. In RLVR the verifier is the training reward; the gold check is then held-out problems
(and reading answers, since a verifier can also be gamed: S5, S7).

### D62 · ★ [S2 · intro] "What is the setting in practice at frontier labs such as Anthropic and OpenAI?" (2026-11-03)

> ★ **Marked important by the learner (2026-11-03),** with D61: an RLHF run has three components,
> reward model (or verifier) and KL in the update, gold signal for evaluation only. Workflow animation
> `CAND-ANIM-057` ★.

**Limit first:** current production recipes are not public. What is known comes from papers and model
reports, mostly 2020–2025. Written from memory in this session, not re-checked against the papers:
**verify each line before reuse in an article.**

**Published RLHF setups (the three signals of D61):**

| Source | Reward (proxy) | KL to reference | Gold / evaluation |
|---|---|---|---|
| OpenAI, summarization (Stiennon et al. 2020) | reward model from human comparisons | per-token KL penalty to the SFT policy | fresh human comparisons; showed that pushing the reward model further eventually lowered human-judged quality |
| OpenAI, InstructGPT (Ouyang et al. 2022) | 6B reward model, rankings of 4–9 answers | per-token KL penalty to the SFT model; plus a pretraining-loss mix (PPO-ptx) | labeler win rate against the SFT baseline on held-out customer prompts; public benchmarks (truthfulness, toxicity) |
| OpenAI, over-optimization (Gao et al. 2022) | small proxy reward models trained on labels from a 6B "gold" reward model | measured as the distance optimized | the 6B gold model: gold reward rises then falls with KL, for both RL and best-of-n |
| Anthropic, HH assistant (Bai et al. 2022a) | preference model (with a pretraining stage) | KL penalty in PPO | crowdworker comparisons summarized as Elo scores; reported reward rising roughly linearly in √KL; iterated "online" rounds with new preference data |
| Anthropic, Constitutional AI (Bai et al. 2022b) | preference model trained on **AI** comparisons guided by written principles (RLAIF) for harmlessness, human labels for helpfulness | KL penalty in PPO | human comparisons of helpfulness and harmlessness |
| Meta, Llama 2 (2023) | two reward models (helpfulness, safety), margin loss | KL penalty in PPO, after rounds of rejection sampling | human raters; reward-model scores as a secondary check |
| DeepSeek-R1 (2025), RLVR | rule-based: answer correctness and format (no learned reward model, to avoid reward hacking) | GRPO with a KL term | held-out reasoning benchmarks |

**Patterns in the published work:** the reward is a learned reward model for open-ended quality and a
verifier (tests, exact answers) where answers can be checked, often both, plus rubric- or
principle-guided AI judges (Constitutional AI; OpenAI's rule-based rewards). The KL penalty to the SFT
reference is standard in RLHF; some later reasoning-RL recipes reduce it or drop it (e.g. DAPO, 2025) and
control drift in other ways. The gold signal is several independent checks, not one number: held-out
human preference tests, benchmark suites, red-teaming and safety evaluations, and reading samples. Exact
β values, reward-model sizes and stopping rules for current models are not published.

### D63 · [S2 · intro] "In DPO, is true quality defined?" (2026-11-03)

**No.** DPO changes components 1 and 2 of D61; component 3 (gold) stays outside the method, chosen by
whoever evaluates.

| Component | RLHF (PPO / GRPO) | DPO |
|---|---|---|
| 1 · reward | frozen reward model scores sampled answers | **none at training time**: the labeled pairs are used directly; the model's own **implicit reward** `β·log(π(y|x)/π_ref(y|x))` plays the role of the score inside a Bradley–Terry loss (Parts C–D) |
| 2 · KL | `−β·KL` term, estimated from samples | **built into the loss**: the same β and the reference log-ratios; no separate KL estimate during training |
| 3 · gold | external evaluation | external evaluation, the same options: held-out human comparisons, an independent judge or reward model, verifiers, benchmarks, reading samples |

**A trap specific to DPO:** the metric it reports most naturally, implicit-reward accuracy on held-out
pairs (S2 Part G: 50% → 67% ± 3% on 512 pairs), is *not* gold. It measures whether the model ranks two
**fixed, human-written** answers the way the labeler did. It says nothing directly about the answers the
model **generates**. A larger chosen-vs-rejected margin can even come with the chosen answer getting less
likely (both log-probabilities fall, the rejected one faster), and a pair can be won without the model
producing the preferred answer: Day 18 lesson
[margin versus the generated answer](margin-versus-generated-answer.md). Gold for DPO therefore means
evaluating **generations**: sample from the trained policy on held-out prompts and compare against the
reference with humans, a judge, or a verifier.

DPO can also over-optimize: the implicit reward is a proxy fitted to a finite set of pairs, and
training longer or with small β can keep raising held-out margins while generation quality stops
improving (reported for direct alignment methods in 2024; from memory, verify before reuse).

### D64 · [S2 · intro] "What does optimum mean? Plain language and easy examples" (2026-11-03)

**Optimum** = the best choice among all allowed choices, judged by one stated score (the **objective**).
Three parts are always needed: what may be chosen, the score, and "highest score wins". Everyday example:
hours of study tonight, score = what you learn − how tired you are tomorrow; the number of hours with the
highest score is the optimum, and it is neither 0 nor 10.

**In S2:** the choice is the policy, i.e. how to split 100% of probability over the possible answers;
the score is `J(π) = E[R] − β·KL(π‖π_ref)`; the optimum `π*` is the split with the highest J.
Toy (D56: π_ref 0.7 / 0.299 / 0.001, R 2 / 1 / 4), β = 1:

| Policy π (A, B, C) | E[R] | KL | J = E[R] − KL |
|---|---|---|---|
| keep the reference (0.70, 0.30, 0.00) | 1.703 | 0.000 | 1.703 |
| all on A (1, 0, 0) | 2.000 | 0.357 | 1.643 |
| (0.90, 0.10, 0) | 1.900 | 0.117 | 1.783 |
| (0.80, 0.15, 0.05) | 1.950 | 0.199 | 1.751 |
| all on C (0, 0, 1) | 4.000 | 6.908 | −2.908 |
| **π* = (0.856, 0.135, 0.009)** | 1.884 | 0.085 | **1.798** (highest) |

No other split scores higher; Part B shows why (`π* ∝ π_ref·e^{R/β}`, and the best value equals
`β·log Z` = 1.798). Two cautions: "optimal" is always optimal **for this objective** (by true quality
the best β may differ, D59); and training methods (PPO, GRPO, DPO) only approach `π*` from samples, while
the toy computes it exactly.

### D65 · ★ [S2 · intro] "I need to understand the different math concepts in the KL-regularized objective, like optimum, coefficient, partition function, in plain language" (2026-11-03)

**Learner's request:** a plain-language explanation of each mathematical term. **Answer:** the symbol-by-symbol
glossary above ("Math glossary"), all on the same three-answer toy, with the optimum built in three
steps: multiply each reference probability by its reward weight `e^{R/β}`, add the results (= Z),
divide each by Z. Recorded as a knowledge gap until the learner explains it back.

### D66 · [S2 · Parts A–C] "Make a coherent explanation and story of Parts A, B, C so I understand the full picture; plain language" (2026-11-03)

**Setting (the notebook's toy, β = 1).** One prompt, five possible answers 0–4. The starting model's
habits: π_ref = (0.40, 0.25, 0.15, 0.12, 0.08). The reward model's opinion: R = (0, 1, 0.5, 2, −1);
answer 3 is best but the model rarely says it. Question of the stage: how should the model change?

**Part A: a ruler for change.** KL(π‖π_ref) = Σ π·log(π/π_ref): for each answer, how much more (or less)
likely it became, averaged over what the new model says. 0 = no change; large = moved far, especially
toward answers the reference almost never gives. Direction matters: weights come from the new model.
(D35's example: 0.0851 one way, 0.0920 the other.)

**Part B: the best compromise.** Score each possible policy: J = average reward − β × KL. The best one is
built in three steps: multiply each π_ref by e^{R/β} (0.40, 0.68, 0.247, 0.887, 0.029), add (Z = 2.243),
divide → π* = (0.178, 0.303, 0.110, 0.395, 0.013). Answer 3 rises 0.12 → 0.395 without taking everything.
Scores: keep the reference J = 0.485, always answer 3 J = −0.12 (KL cost 2.12), π* J = 0.808 = β·log Z;
no random policy beats it (every other policy loses exactly β·KL(π‖π*)). The β knob: large β → stay at
π_ref; small β → all on answer 3; answer 1 rises then falls (peak 0.31 near β ≈ 1.3).

**Part C: run the arrow backwards.** Part B went reward → policy. Part C reads the reward back out of the
policy: how much the policy raised each answer relative to the reference, in log units, times β, is the
**implicit reward** r̂ = β·log(π/π_ref). For π*: r̂ = (−0.808, 0.192, −0.308, 1.192, −1.808) = R − 0.808 for
every answer. The shared constant 0.808 is β·log Z. Bradley–Terry compares differences only, so it
cancels: P(3 beats 1) = σ(2 − 1) = σ(1.192 − 0.192) = 0.731 either way. Z, impossible to compute for a
real LLM, is never needed.

**The arc:** A gives the cost, B gives the best policy for a reward, C shows a policy contains its
reward up to a constant that comparisons ignore. So instead of training a reward model and then a
policy, train the policy directly so its implicit reward fits human preference pairs: Part D, the DPO
loss.


### D67 · ★ [S2 · Parts A–C] "Pair an easy example with the math so A, B, C form one complete story" (2026-11-03)

> ★ **Marked important by the learner** (2026-11-03: "the above is important, mark it in note and article
> plan"). The egg-question story is key point 5 and the running example of the KL/DPO article plan
> ([`plan.md`](../../publications/x-articles/x-post-training-kl-dpo/plan.md)); main-note plan row P13.

Same numbers as D66 (β = 1), with the five answers made concrete and each part told as **example → math
→ numbers**. Numbers checked in Python.

**The example** (revised at the learner's request: "What is a hash table?" was not a good example; use an
everyday one). Prompt: *"How long should I boil an egg?"*

| # | Answer (shortened) | π_ref | R |
|---|---|---|---|
| 0 | "Boil it until it's done." (vague; the model's habit) | 0.40 | 0 |
| 1 | "About 10 minutes." | 0.25 | 1 |
| 2 | a long paragraph about eggs that eventually says 10 minutes | 0.15 | 0.5 |
| 3 | "Soft-boiled 6–7 minutes, hard-boiled 10–12; then cold water so it peels easily." | 0.12 | 2 |
| 4 | "2 minutes gives a hard-boiled egg." (wrong) | 0.08 | −1 |

**Part A (ruler).** Example: a "small move" policy shifts 0.10 from answer 0 to answer 3:
π = (0.30, 0.25, 0.15, 0.22, 0.08). Math: KL(π‖π_ref) = Σ π·log(π/π_ref). Per answer: answer 0 term
0.30·log(0.30/0.40) = −0.086 (negative: made less likely), answer 3 term 0.22·log(0.22/0.12) = +0.133,
unchanged answers 0; total **0.047** ≥ 0. Other readings: reference 0, π* 0.328, "always answer 3"
log(1/0.12) = **2.12**. Direction: KL(π_ref‖π) for the small move = 0.042, not 0.047.

**Part B (best compromise).** Score J = E[R] − β·KL for each policy:

| Policy | E[R] | KL | J |
|---|---|---|---|
| reference | 0.485 | 0 | 0.485 |
| small move | 0.685 | 0.047 | 0.638 |
| always answer 3 | 2.000 | 2.12 | −0.12 |
| π* | 1.136 | 0.328 | **0.808** |

π* by multiply (×e^{R/β}: 0.40, 0.68, 0.247, 0.887, 0.029) → add (Z = 2.243) → divide
(0.178, 0.303, 0.110, 0.395, 0.013). Check of "every other policy loses exactly β·KL(π‖π*)": small move
0.808 − 0.638 = 0.170 = KL(small‖π*). β dial: β = 0.5 → (0.043, 0.200, 0.044, 0.711, 0.001); β = 2 →
(0.290, 0.299, 0.140, 0.236, 0.035); β = 5 → (0.359, 0.274, 0.149, 0.160, 0.059).

**Part C (read the reward back).** Example: you are handed only the trained model π* and π_ref, not R.
Math: r̂ = β·log(π/π_ref) = (−0.808, 0.192, −0.308, 1.192, −1.808) = R − 0.808 for every answer. A person
compares answer 3 with answer 1: σ(2 − 1) = σ(1.192 − 0.192) = 0.731 with either reward. Bridge to DPO:
at the reference, r̂ = 0 for every answer, P(3 beats 1) = σ(0) = 0.5, loss −log 0.5 = 0.693; at π*, 0.731
and loss 0.313. The DPO loss trains the policy to move from the first to the second using only the pair.

### D68 · ★ [S2 · Part A] "I want to re-study A, B, C. Explain the KL formula KL(π‖π_ref) = Σ_y π(y) log(π(y)/π_ref(y)) in detail" (2026-11-04)

> ★ **Marked important by the learner** (2026-11-04: "the above is important, mark it in note").

Start of the learner's re-study of Parts A–C. Answer, on the egg question of D67 (small move: 0.10 from
the vague answer to the best answer). Numbers checked in Python.

- **Read aloud:** for every possible answer y, take how many times more likely the new model makes it
  (π(y)/π_ref(y)), take the log, weight by how often the new model gives it (π(y)), add up.
- **Symbols:** y = one complete answer; π(y) = new model's probability; π_ref(y) = frozen reference's
  probability; ratio > 1 up, = 1 unchanged, < 1 down; log turns ratio into a signed amount (log 1 = 0,
  up → positive, down → negative); π(y) weights each answer by how often the new model says it; Σ_y
  over all answers.
- **Term by term:** ratios 0.75, 1, 1, 1.833, 1 → logs −0.288, 0, 0, 0.606, 0 → weighted −0.086, 0, 0,
  +0.133, 0 → **KL = 0.047** (nats).
- **Why never negative overall:** probability taken from one answer must go to another; using
  log x ≤ x − 1: −KL = Σ π log(π_ref/π) ≤ Σ π (π_ref/π − 1) = Σ π_ref − Σ π = 1 − 1 = 0 (Gibbs' inequality).
  KL = 0 only when π = π_ref.
- **Edge cases:** π(y) = 0 → the term is 0 (dropping an answer costs nothing directly); π(y) > 0 where
  π_ref(y) = 0 → ∞ (inventing an answer the reference never gives is forbidden). Always-best: only one
  term, 1·log(1/0.12) = 2.12.
- **As an average:** KL = E_{y~π}[log π(y) − log π_ref(y)]: sample answers from the new model, average
  their log-ratios. Samples: 10 → 0.096, 100 → 0.055, 10,000 → 0.049 (true 0.047). For an LLM, log π(y)
  is the sum of token log-probabilities, so the log-ratio splits over tokens (D35).
- **Direction:** swap the roles → weights from π_ref: 0.40·log(0.40/0.30) + 0.12·log(0.12/0.22) =
  0.115 − 0.073 = 0.042, not 0.047.

### D69 · [S2 · Part A] "Why 'because the weights come from the new model, which says the raised answers more often'?" (2026-11-04)

**Correction.** That sentence (from the D68 chat answer, not in this note) was wrong. In the egg small
move the raised answer has weight 0.22 and the lowered one 0.30: the lowered answer carries *more*
weight, yet KL = 0.047 > 0. Non-negativity also holds in the other direction (weights from π_ref), so it
is not caused by the choice of weights.

**Correct reason, term by term.** Each term is at least the probability that answer gained:
π(y)·log(π(y)/π_ref(y)) ≥ π(y) − π_ref(y) (from log x ≥ 1 − 1/x). The gains add to 0 (Σπ − Σπ_ref = 1 − 1),
so the terms add to ≥ 0.

| Move | Answer that went down: lost / term | Answer that went up: gained / term | KL |
|---|---|---|---|
| 0.10 from vague (0.40) to best (0.12) | −0.10 / −0.086 | +0.10 / +0.133 | 0.047 |
| 0.10 from best (0.12) to vague (0.40) | −0.10 / −0.036 | +0.10 / +0.112 | 0.076 |

A term for a lowered answer is always less negative than the amount lost; a term for a raised answer
is always more positive than the amount gained. Gains and losses are equal, so the positive side wins.
Equality only when nothing moved.

### D70 · ★ [S2 · Part A] "What does 'each term ≥ probability gained, gains add to zero' mean? Could the negative be much larger than the positive, so KL is negative? Why does it always go in the better direction; can the best answer go down?" (2026-11-04)

> ★ **Marked important by the learner** (2026-11-04: "the above is important, mark it in note").

**(1) Meaning.** Probabilities add to 1 before and after, so the total lost by the answers that went
down equals the total gained by the answers that went up. "Gained" for a lowered answer is negative.
Each KL term is at least that answer's gain; so the lowered answers subtract *at most* what they lost,
the raised answers add *at least* what they gained, and the sum is ≥ 0.

**(2) Can the negative side win?** No. A big loss cannot meet a small gain: lost = gained always. The log
is applied to each answer's ratio, not to the total, and a large negative log-ratio is multiplied by
the answer's *new*, now small, probability. Examples (egg question):

| Move | gains | log-ratios of moved answers | terms | KL |
|---|---|---|---|---|
| small: vague 0.40 → 0.30, best 0.12 → 0.22 | −0.10, +0.10 | −0.288, +0.606 | −0.086, +0.133 | 0.047 |
| extreme: vague 0.40 → 0.01, best 0.12 → 0.51 | −0.39, +0.39 | −3.689, +1.447 | −0.037 (= 0.01 × −3.689), +0.738 | 0.701 |
| spread: vague 0.40 → 0.10; 10-min, long, best each +0.10 | −0.30, +0.10 ×3 | −1.386; +0.336, +0.511, +0.606 | −0.139; +0.118, +0.128, +0.133 | 0.240 |

**(3) Direction.** KL does not know which answers are good; it measures any change. The best answer can
go down: 0.10 from best to vague gives KL 0.076 > 0. The *direction* comes from the reward term of
J = E[R] − β·KL, not from KL. Scores: don't change 0.485; small move toward best 0.638; extreme 0.564;
spread 0.595; best goes down E[R] 0.285, J = 0.285 − 0.076 = **0.209** (worse on both: less reward and a
KL cost), so the optimizer never chooses it. In the D67–D69 examples the move toward the best answer
was picked by hand to show the ruler.

### D71 · [S2 · Part A] "So KL is only a measure of two distributions, not a control? The control happens in the reward part?" (2026-11-04)

**Half right.** KL by itself is only a measurement (D59: it is also logged during training as a
diagnostic). Once it enters the objective J = E[R] − β·KL, it does control something: the reward
controls **which way** the policy moves, β·KL controls **how far**. Without it the optimum goes all-in;
the KL term pulls every answer back toward its reference probability. Egg question, best answer's
probability in π\* (reference 0.12): β → 0 (no KL cost): 1.00; β = 0.5: 0.71; β = 1: 0.40; β = 2: 0.24;
β = 5: 0.16; β → ∞: 0.12. In the formula π\* = π_ref·e^{R/β}/Z: e^{R/β} (reward) chooses which answers
are boosted; π_ref and β (the KL part) set the starting point and the size of the boost.

### D72 · ★ [S2 · Part A] "So 'how far to move' explains why KL is never negative: the new model is either the same as the reference or different; it cannot be 'inside' the reference" (2026-11-04)

> ★ **Marked important by the learner** (2026-11-04: "the above is important, mark it in note").

**Learner's intuition, confirmed with two refinements.** Yes: KL behaves like a distance, the
smallest possible value is "no change" (0), and there is nothing below "the same". Refinements:
(1) The intuition says what a measure of change *should* do; the D69–D70 math is why KL actually does
it. A badly built change measure can go negative. Egg question: unweighted Σ log(π/π_ref) gives 0.318
for the small move but **−1.569** for moving 0.10 away from the best answer; weighting by π_ref instead
of π gives −0.042 and −0.126. KL (weights π) gives 0.047 and 0.076. (2) KL is not a true distance: it is
not symmetric (0.047 vs 0.042 for the small move, D68). Also, the *exact* KL is ≥ 0, but a one-sample
estimate (k1) can be negative (D59).

### D73 · [S2 · Part B] "Explain the Part B formula in detail" (2026-11-04)

Re-study of Part B on the egg question (β = 1). Numbers checked in Python.

1. **Objective** J(π) = Σ π(y)R(y) − β Σ π(y) log(π(y)/π_ref(y)): average reward minus β × the Part A ruler.
   β = price in reward units of one nat of KL. Small move: 0.685 − 0.047 = 0.638.
2. **Where π\* comes from (equal net value).** Net value of an answer = R(y) − β·log(π(y)/π_ref(y)) (the
   exact derivative of J with respect to π(y) is this minus β, the same for every answer, so it can be
   ignored). If one answer's net value is higher than another's, shifting probability toward it raises
   J; at the best policy no shift helps, so all net values are equal. Small move: (0.288, 1, 0.5, 1.394,
   −1) → move toward the best answer, away from "2 minutes". π\*: 0.808 for all five. Equal net values ⇒
   log(π/π_ref) = R/β − const ⇒ π\* = π_ref·e^{R/β}/Z, Z = Σ π_ref·e^{R/β} = 2.243 (multiply, add, divide).
   This is the plain form of Chapter 11's Lagrange-multiplier derivation.
3. **Softmax form**: π\* = softmax(log π_ref + R/β). log π_ref acts as the reference's logits, R/β adds a
   bonus: logits (−0.916, −0.386, −1.397, −0.120, −3.526) → (0.178, 0.303, 0.110, 0.395, 0.013). Same numbers;
   never overflows.
4. **Proof that nothing beats π\***: J(π) = β log Z − β·KL(π‖π\*). β log Z = 0.808 is fixed; KL ≥ 0 (D69–D72)
   and = 0 only at π = π\*. Small move: 0.808 − 0.170 = 0.638. Uses the Part A result directly.
5. **Adding a constant to every reward changes nothing**: +1 → weights × e (1.087, 1.847, 0.672, 2.410,
   0.080), Z = 6.097, π\* unchanged. Only reward differences within a prompt matter.
6. **β limits**: β = 0.1 → all on the best answer; β = 100 → (0.398, 0.251, 0.150, 0.122, 0.079) ≈ π_ref.
   Answer 1 rises then falls: π\*(3)/π\*(1) = (0.12/0.25)·e^{1/β} = 1.305 at β = 1, 3.547 at 0.5, 26.2 at 0.25.

### D74 · [S2 · Part B] "I got lost. What does the star on π mean?" (2026-11-04)

The learner got lost in D73 (six pieces at once was too much). Answer: the star is math notation for
**the best one** (optimal). Three πs: π_ref = the frozen starting model (0.40, 0.25, 0.15, 0.12, 0.08);
π = any model we are considering or training (e.g. the small move 0.30, 0.25, 0.15, 0.22, 0.08); π\* =
the one model with the highest score J for this reward and β (0.178, 0.303, 0.110, 0.395, 0.013,
J = 0.808). π\* is a target computed on paper, not something training starts from. Next: go through D73
one piece at a time.

### D75 · [S2 · Part B] "Go one piece at a time, start with J. And I got lost after 'net value' and 'all answers equal': what does it mean?" (2026-11-04)

**Piece 1, the score J**, for the small move (β = 1): reward part Σ π·R = 0 + 0.25 + 0.075 + 0.44 − 0.08
= 0.685; KL part 0.047 (Part A); J = 0.685 − 1 × 0.047 = 0.638. Starting model: 0.485 − 0 = 0.485. β is
the price of one unit of KL: small move J = 0.685 at β = 0, 0.638 at β = 1, 0.591 at β = 2 (starting
model always 0.485, since it has KL 0). J is a comparison tool: higher J = better trade.

**"Net value" in plain words** = how much J goes up if you give that answer a tiny bit more
probability. Test by moving 0.01 from the vague answer to the best answer: from the small move,
J 0.638 → 0.649 (+0.011): a gain, so keep moving. From π\*: J 0.8078 → 0.8074 (−0.0004), and moving the
other way also −0.0004: no move helps. "All net values equal" = "no small move of probability can raise
J any more" = you are at the top = π\*. (Piece 2 to continue from here.)

### D76 · ★ [S2 · Part B] "What is Z? Explain from 'no move helps' to the formula of π\*" (2026-11-04)

> ★ **Marked important by the learner** (2026-11-04: "the above is important, mark it in note").

**Piece 2**, egg question, β = 1.

1. **Net value, from a tiny move.** Moving 0.01 from vague to best (starting at the small move): reward
   +0.01 × 2 = +0.020; KL 0.0470 → 0.0564 (+0.009: about +0.01·(0.606 + 1) for best, −0.01·(−0.288 + 1) for
   vague); J +0.011. Per unit of extra probability, an answer adds R(y) − β·(log(π/π_ref) + 1); the "+1" is
   the same for every answer and cancels when probability moves between answers. Net value
   = R(y) − β·log(π(y)/π_ref(y)); small move: (0.288, 1, 0.5, 1.394, −1).
2. **No move helps ⇔ all net values equal** (to some number c). If one were higher, moving probability
   to it would raise J.
3. **Solve** R − β·log(π/π_ref) = c → log(π/π_ref) = (R − c)/β → (undo the log with e) π = π_ref · e^{R/β} · e^{−c/β}.
4. **Z.** e^{−c/β} is one number shared by all answers, unknown at first. The weights π_ref·e^{R/β} =
   (0.40, 0.68, 0.247, 0.887, 0.029) add to 2.243, not 1; probabilities must add to 1, so the shared factor
   must be 1/2.243. **Z = 2.243 = the sum of the weights**; dividing by it turns weights into
   probabilities (like turning counts into percentages). Then c = β·log Z = 0.808, the equal net value
   of D73. Check: 0.12 × e² / 2.243 = 0.395. "Partition function" is a name from physics. For an LLM, Z
   sums over every possible reply, so it cannot be computed; Part C shows DPO never needs it.

### D77 · [S2 · Part B] "So this whole process explained the optimization on one training example, right?" (2026-11-04)

**Close, three corrections.** (1) **One prompt, not one training example**: the egg question with all
five answers and their rewards; in DPO a training example is a pair (prompt, chosen, rejected), and
here we have a reward for every answer, not a pair. (2) **Solved on paper, not trained**: with five
answers we can list everything and compute π\* exactly (multiply, add, divide); no gradient steps.
(3) **Real setting**: the objective is averaged over many prompts, and the optimum is per prompt,
π\*(y|x) = π_ref(y|x)·e^{R(x,y)/β}/Z(x), with a separate Z(x) for each prompt. A real LLM cannot list every
reply or set probabilities directly; its shared weights serve every prompt at once. So Part B gives
the **target** for each prompt; training (PPO/GRPO by sampling replies, or DPO from preference pairs)
is the approximate way to move the model's weights toward it for all prompts together.

### D78 · [S2 · Part B] "If we have a lot of π\*, how do we optimize them overall?" (2026-11-04)

**Overall score** = average of the per-prompt scores: J_total = E_x[ E_{y~π(·|x)}[R(x,y)] − β·KL(π(·|x)‖π_ref(·|x)) ].
Each prompt's score is J_x = β·log Z(x) − β·KL(π(·|x)‖π\*(·|x)) (D73 piece 4), so

  J_total = (average of β·log Z(x), fixed) − β · (average over prompts of KL(π(·|x)‖π\*(·|x))).

Maximizing the total = making the model as close as possible, **on average over prompts**, to every
prompt's own target. No prompt's target has to be given up for another's: if the model could set each
prompt's probabilities separately, the best is simply π\* everywhere.

**Two-prompt example (β = 1).** Egg question: Z = 2.243, ideal 0.808, π\* = (0.178, 0.303, 0.110, 0.395,
0.013). Second prompt "How do I politely say no to a meeting?": answers "No." / "I can't make it." /
"Thanks for the invite; I can't join, but could you send me the notes?" with π_ref (0.5, 0.3, 0.2),
R (−1, 0.5, 2): Z = 2.156, ideal 0.768, π\* = (0.085, 0.229, 0.685). Average ideal 0.788.

| Model | egg J | email J | average J | average KL to targets |
|---|---|---|---|---|
| both at π\* | 0.808 | 0.768 | 0.788 | 0 |
| egg small move, email at π\* | 0.638 | 0.768 | 0.703 | 0.085 |
| egg small move, email (0.4, 0.3, 0.3) | 0.638 | 0.318 | 0.478 | 0.31 |

Average J = 0.788 − average KL to targets, exactly.

**Why it is still hard.** A real LLM's probabilities for all prompts come from the same weights θ, so a
weight change that helps one prompt also moves others (sometimes helping similar prompts, sometimes
hurting). Training = gradient descent on θ (D52), each step on a minibatch of prompts with the gradient
averaged. π\* and Z(x) are never computed: PPO/GRPO sample replies for a batch of prompts, score them
with the reward model, subtract the KL, and step to raise the average J; DPO steps on a batch of
preference pairs from many prompts, and Z(x) cancels inside each pair. The trained model ends near, not
exactly at, every target (limited capacity, shared weights, finite data).

### D79 · [S2 · Part B] "In practice, when can training be regarded as complete; which signal?" (2026-11-04)

**No single "done" signal.** Training loss and proxy reward keep improving long after the model stops
getting better (D59–D61, D63). In practice: train for a planned budget, save checkpoints, evaluate each
on signals that are **not optimized**, keep the best checkpoint (often not the last), and stop early on
warning signs.

**Toy (computed).** Egg question plus a sixth answer the reward model overrates: "Boil it 20 minutes,
always, for food safety. Experts agree." π_ref 0.005, proxy R = 3, gold = −2 (other answers as before,
"2 minutes" 0.075; gold = proxy for them). Each row is π\* at a smaller β, standing in for "trained
further" (KL grows as training goes on):

| KL | proxy reward | gold | P(best) | P(trick) |
|---|---|---|---|---|
| 0.00 | 0.51 | 0.49 | 0.12 | 0.005 |
| 0.16 | 0.97 | 0.85 | 0.28 | 0.023 |
| 0.37 | 1.22 | 1.00 | 0.38 | 0.043 |
| **0.74** | 1.53 | **1.10** (peak) | 0.50 | 0.086 |
| 1.35 | 1.89 | 0.99 | 0.58 | 0.18 |
| 2.34 | 2.30 | 0.36 | 0.53 | 0.39 |
| 3.61 | 2.67 | −0.76 | 0.30 | 0.69 |

Proxy reward rises the whole way; gold peaks at KL ≈ 0.74 and then falls (the over-optimization
pattern of Gao et al. 2022). Pick the checkpoint at the gold peak.

**Signals used (from memory, verify before reuse in an article):**
1. **Main: gold evaluation of checkpoints** — held-out win rate against the SFT model (human or
   strong AI judge), benchmarks, verifier accuracy on held-out problems, safety evals. Stop / select when
   it plateaus or turns down.
2. **Proxy reward**: flattening is normal; still rising while gold falls = reward hacking.
3. **KL**: watched as a budget; a fast rise is a warning; adaptive controllers aim at a target KL
   (Ziegler et al. 2019).
4. **Sample health**: reply length blowing up, entropy / diversity collapsing, repetition, a phrase the
   reward model likes appearing everywhere; read samples.
5. **Capability regressions**: general benchmarks (math, code, knowledge) dropping ("alignment tax").
6. **DPO**: training loss and held-out implicit-reward accuracy are not gold (D63); watch that the
   chosen answers' log-probability does not keep falling; typically 1–3 epochs, checkpoint chosen by
   held-out win rate.

### D80 · [S2 · Part A] "We need a lot of animations to explain KL itself, and maybe a separate X article about KL itself" (2026-11-04)

Request recorded as main-note plan row P15. Proposed: animation set `CAND-ANIM-058`–`064` on the egg
question (storyboards in `visuals/animations/PROPOSALS.md`; table under **Animations and article** above),
reusing 054 ★, 055, 037; and a new article plan
[`publications/x-articles/x-post-training-kl/plan.md`](../../publications/x-articles/x-post-training-kl/plan.md)
with six key points (ruler term by term; never negative; no direction; the exact form matters; rare
moves priced high and estimated from samples; no good KL value) and a 9-section outline. Proposed to
come before the KL/DPO article, which would then use KL as a known ruler. Slot and split for the
learner to decide.

### D81 · [S2 · Part A] "Yes, KL article first. In the KL article plan make sure it can find the corresponding learning note and animation plan; we have discussed a lot" (2026-11-04)

Decided: the KL article comes before the KL/DPO article. The plan
[`publications/x-articles/x-post-training-kl/plan.md`](../../publications/x-articles/x-post-training-kl/plan.md)
now opens with **Where to find everything**: a table of every learning-note entry it uses (D35 ★ in the
main note; D57 ★, D59–D61, D67 ★, D68 ★, D69, D70 ★, D71, D72 ★, D79, D80 here; the Key concepts rows; the
Day 3 cross-entropy note) with what each gives and which key point / section uses it; a table of every
animation (058–064 new, 054 ★, 055, 037, and the produced X-POST-004 GIF 11) with its PROPOSALS
section and status; other sources (S2, Day 15 · 04, Day 17 · 01, Day 20 · 02 notebooks; Chapters 3, 11,
12 §12.8, 13 §13.4); the running example's exact numbers; and an outline table mapping each section to
its notes and animations.

### D82 · [S2 · Part C] "Explain Part C" (2026-11-04)

Re-study of Part C on the egg question (β = 1), in three steps. Numbers checked in Python.

1. **Run the arrow backwards.** Part B: reward → π\*. Take the log of π\* = π_ref·e^{R/β}/Z:
   log π\* = log π_ref + R/β − log Z, solve for R: **R = β·log(π\*/π_ref) + β·log Z.** Ratios π\*/π_ref
   (0.446, 1.212, 0.735, 3.294, 0.164) → logs (−0.808, 0.192, −0.308, 1.192, −1.808) → + 0.808 =
   (0, 1, 0.5, 2, −1) = R exactly.
2. **Implicit reward** r̂(y) = β·log(π(y)/π_ref(y)): defined for any policy, how much it raised or lowered
   each answer, in reward units. For π\*: r̂ = R − 0.808 (one constant for all answers). For the small move:
   r̂ = (−0.288, 0, 0, 0.606, 0): the reward for which that policy would be optimal (up to a constant).
3. **Z cancels in comparisons.** Bradley–Terry P(i ≻ j) = σ(s_i − s_j) uses differences; the shared
   β·log Z drops out. Pairs, from R and from r̂: best vs 10 min 0.731 / 0.731; 10 min vs vague 0.731 /
   0.731; best vs wrong 0.953 / 0.953; vague vs long 0.378 / 0.378 (all ten pairs agree to ~1e-16).

**Why it matters.** For an LLM, Z(x) sums over every possible reply and cannot be computed, but r̂ needs
only two log-probabilities (policy and reference) per answer. So a policy *is* a reward model, up to
a per-prompt constant: train π so that σ(r̂(chosen) − r̂(rejected)) fits human choices, which is the
Bradley–Terry likelihood with π as the scorer. That is the DPO loss (Part D).

### D83 · ★ [S2 · Part D] "Interesting! Continue to explain Part D" (2026-11-04)

> ★ **Marked important by the learner** (2026-11-04: "the above is important, mark it in note"). Parts C
> and a small part of D are previewed in the KL article (learner, same message; D84).

Re-study of Part D (the DPO loss and its gradient) in four steps. Numbers checked in Python.

1. **Score with the implicit reward** (Part C), now for the trainable model π_θ: r̂(y) =
   β·(log π_θ(y) − log π_ref(y)). For an LLM, log π(y) is the sum of the reply's token log-probabilities.
2. **Margin and loss** for one recorded pair (chosen y_w, rejected y_l): m = r̂(y_w) − r̂(y_l);
   loss = −log σ(m) = softplus(−m), the Bradley–Terry loss of S1 with π_θ as the scorer; averaged over a
   batch. Egg pair (best chosen over "about 10 minutes", β = 1): start π_θ = π_ref → m = 0 → loss
   log 2 = 0.693; small move m = 0.606 → 0.435; π\* m = 1.0 → 0.313.
3. **Notebook's three pairs** (β = 0.5, sequence log-probs): pair 0 chosen +1 nat, rejected −1 → m = +1.0,
   loss 0.313; pair 1 chosen −2, rejected +1 → m = −1.5, loss 1.701; pair 2 only rejected −0.5 → m = +0.25,
   loss 0.576; mean 0.863. Only changes relative to the reference enter, not raw log-probabilities.
4. **Gradient** ∂L/∂log π_θ(y_w) = −β·σ(−m)/B, ∂L/∂log π_θ(y_l) = +β·σ(−m)/B: gradient descent raises the
   chosen answer and lowers the rejected one, weighted by σ(−m): 0.953 at m = −3 (wrong order, big push),
   0.5 at m = 0, 0.269 at m = 1, 0.047 at m = 3 (already right, small push). Three pairs: weights 0.269,
   0.818, 0.438.

**Caveats (notebook).** No finite margin minimizes the loss for a hard label, but the push fades. β
multiplies the gradient (Adam largely removes the factor). The derivatives are per answer; in a network
both answers share weights, so where the probability goes is not controlled: the chosen answer's
log-probability can fall too (D79 warning signal). The reference's log-probs are detached: no gradient
reaches the frozen model.

### D84 · [S2 · Part D] "In the KL article, it is good to mention C and a small part of D as well" (2026-11-04)

Added to the KL article plan
([`publications/x-articles/x-post-training-kl/plan.md`](../../publications/x-articles/x-post-training-kl/plan.md)):
key point 7 and outline §9 "The log-ratio is also a reward": the KL's per-answer log-ratio, times β,
is the implicit reward (Part C, D82; egg: r̂ = R − 0.808, Z cancels in comparisons, 0.731 both ways), and
DPO trains on it directly (a small part of D, D83 ★: margin, −log σ(m), 0.693 → 0.313). The full
derivation and the gradient stay in the KL/DPO article. D82 and D83 ★ added to the plan's "Where to
find everything" table.
