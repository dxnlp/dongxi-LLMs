# Learning from preferences

A running record of the learner's study of preference learning, opened 2026-10-27 while
reading [SLP3 Chapter 8 §8.3 "Learning from Preferences"](https://web.stanford.edu/~jurafsky/slp3/8.pdf).
Everything on this topic is kept in this one file: important concepts, knowledge gaps and
discussions. It's meant for reuse in later course material and articles.

**Sources:** SLP3 §8.3 (8.3.1 preference data, 8.3.2 modeling preferences, 8.3.3 learning to
score preferences) and §8.4 (preference-based alignment, DPO). Course:
[Chapter 10](../../book/chapters/10-preferences-and-reward-models.md) (Days 15–16) and
[Chapter 11](../../book/chapters/11-direct-preference-optimization.md) (DPO, Days 17–18).
Previous topic: [learning instruction tuning](../day-11-instruction-data-engineering/learning-instruction-tuning.md).

## Plan: deeper dives

Topics the learner flagged for further discussion or future course material.

| # | Topic | Why | Existing course material | Status |
|---|---|---|---|---|
| P1 | **Bradley–Terry model in depth** | the learner sees it as the key technical idea of preference learning (2026-10-27) | Chapter 10 §10.2–10.5; notebook Day 15 · 01 (margin), Day 15 · 02 (disagreement, calibration) | started (D4) |
| P3 | **Animation: why the sigmoid** (Bradley–Terry derivation) | the learner asked for animations to make SLP3 §8.3.2's derivation easy (2026-10-27) | Chapter 10 loop `preference-margin.gif` (gap → probability → gradient) | proposed as `CAND-ANIM-033`; learner: produce **later on the Mac Studio, after this learning topic** (2026-10-27) |
| P4 | **Animations for Predict 5:** gap → probability → loss → gradient | the learner asked for animation candidates while working on NB04 Predict 5 (2026-10-27) | Day 15 · 04 Part B, figure 03; `preference-margin.gif` | proposed as `CAND-ANIM-034` (four linked pairs, +10 shift) and `CAND-ANIM-035` (the gradient goes where the model is still wrong); state *suggested*, produce on the Mac Studio with 033 |
| P5 ★ | **Animation/visual: PPO's four networks in one iteration** | the learner, at NB04 Part E (2026-10-27): "deserves a very well designed animation or visual" | Day 15 · 04 Part E; Chapter 12 `ppo-clipping` loop; D21–D25 | proposed as `CAND-ANIM-036` (signature candidate): cast table, rollout → no-grad scoring → reward vector → GAE → two updates → repeat, plus a GRPO/DPO contrast; state *suggested*, **high priority**, produce on the Mac Studio |
| P6 ★ | **KL to the reference: fine description + animation** | the learner, at NB04 Part E (2026-10-27): "we need to address KL understanding by fine description and animation as well" | NB04 `ppo_rewards` (k1 in reward), `grpo_iteration` (k3 in loss), Predict 12/15; D35 | description written (D35); animation proposed as `CAND-ANIM-037` (companion to 036); state *suggested*, produce on the Mac Studio |
| P7 ★ | **Article: RL terms for LLMs** (many animations, easy examples) | the learner, after the D39 glossary (2026-10-27) | glossary section above; D21–D25, D35–D39; NB04 Part E code | X article `X-POST-004`, slot approved 2026-10-27: **before preference alignment** (now `X-POST-005`; RLVR → `X-POST-006`) + GIF set `CAND-ANIM-038` (10 loops on the garlic example); write and produce on the Mac Studio |
| P2 | **Reward hacking** | flagged as "very interesting, worth exploring" after D3 (2026-10-27); candidate for discussion and new course material or an article | Chapter 10 §10.8–10.9 (shortcut experiment, optimization changes the population); Chapter 14 §14.1 "A real reward-hacking microscope", §14.5 length bias; notebook Day 24 · 01 `reward_hacking.ipynb`; Gao et al. 2022 overoptimization | planned: discuss after §8.3–8.4, then decide on material |

## Notebook built from these discussions

[Day 15 · 04 preference alignment end to end](../../notebooks/day-15/04_preference_alignment_end_to_end.ipynb)
(`spark-gpu`, built 2026-10-27 at the learner's request after D25): real Qwen3-0.6B, HH-RLHF
`helpful-base` and UltraFeedback `test_prefs`. Parts: A data formats and noise, agent-trajectory
masks (D1, D3, D19) · B Bradley–Terry (D4–D10) · C reward model built and trained (D8, D11, D15–D16)
· D phase-2 prompts (D12, D19–D20) · E PPO's four networks, per-token rewards, GAE, clipping
(D21–D25) · F GRPO (D21, D23) · G DPO on the same pairs (D14, D19, D21). Nineteen predictions.

### Notebook study log (started 2026-10-27)

The learner works through the notebook alongside SLP3 §8.4. Questions raised while working through it go
into **Discussions** as D26, D27, … with a `[NB04 · Part X]` tag. The predictions are recorded
below as the learner answers them.

| Part | Status | Predictions answered | Discussions |
|---|---|---|---|
| A · data formats, noise, agent masks | in progress | P4 | D26 |
| B · Bradley–Terry | in progress | — (P5 open) | animation candidates 034–035 (plan P4) |
| C · reward model | in progress | — (P7 asked about, not predicted) | D27, D28, D29, D30 |
| D · phase-2 prompts | in progress | — | D31 (learner synthesis), D32, D33, D34 (learner synthesis) |
| E · PPO | in progress (four networks) | — | animation CAND-ANIM-036 (plan P5 ★); KL D35 + CAND-ANIM-037 (plan P6 ★); D36, D37, D38, D39 (RL glossary) |
| F · GRPO | not started | — | — |
| G · DPO | not started | — | — |

Prediction record (learner's answer → reference → what it showed):

- **P4 · Which tokens of a trajectory are the model's?** (2026-10-27) The learner ran the reference
  cell first, gave no written prediction, then asked whether the output was correct. *Reference:*
  graded = the tool-call turn and the final answer, each through its `<|im_end|>`. Context = the
  system/tools header, the user question, the `<tool_response>` (Qwen3 wraps it in a *user* turn) and
  the `<|im_start|>assistant\n` headers. See D26.

## ★ RL terms in the LLM setting (glossary, D39)

> ★ **Marked important by the learner (2026-10-27).** Planned as an article with many animations and
> easy-to-follow examples: plan P7, animation set `CAND-ANIM-038`.

Built at the learner's request (2026-10-27): "we need to first explain all the related RL terms in
the context of LLMs". Ordered so each term uses only earlier ones. Running example: prompt "Can
dogs eat raw garlic?", answer "No" · ", raw garlic is" · "toxic" · "." · `<end>` (D24).

| # | Term | In the LLM setting | Garlic example / NB04 code |
|---|---|---|---|
| 1 | **Agent / policy π_θ** | the chat model; π_θ(a \| s) = probability of next token a given the text so far | `policy` |
| 2 | **Environment** | whatever produces states and rewards: the prompt set + the reward model (or verifier, or tools) | `rl_prompts`, `rm` |
| 3 | **State s_t** | prompt + all tokens generated so far | s_2 = prompt + "No, raw garlic is" |
| 4 | **Action a_t** | the next token (one choice out of 151,936) | a_2 = "toxic" |
| 5 | **Trajectory / episode / rollout** | one full sampled answer: s_0, a_0, s_1, a_1, … until `<|im_end|>` or the length limit | `rollout(policy, prompts)` |
| 6 | **Reward r_t** | number received after action t. RLHF: 0 (or the KL charge) on every token, R from the RM on the last | r = [0, 0, 0, 0, −2.0] |
| 7 | **Sparse / delayed reward** | almost all r_t are 0; the informative signal comes once, at the end | — |
| 8 | **Return G_t** | sum of rewards from t to the end: G_t = Σ_k γ^k r_{t+k}; with γ = 1 and only R at the end, G_t = R for every t | G = [−2, −2, −2, −2, −2] |
| 9 | **Discount γ** | weight for later rewards; LLM RLHF usually uses γ = 1 (answers are short, only the end matters) | `gamma=1.0` |
| 10 | **Value V(s)** | expected return from state s if the *current* policy keeps writing: E[G \| s] | V(s_2) = 0.0: from "No, raw garlic is", answers average 0 |
| 11 | **Action-value Q(s, a)** | expected return if you take token a in s, then continue with the policy | Q(s_2, "toxic") ≈ V(s_3) = −1.8 |
| 12 | **Advantage A(s, a)** | Q(s, a) − V(s): how much better this token is than the policy's *average* choice in that state | A = −1.8 − 0.0 = **−1.8** |
| 13 | **Baseline** | a number subtracted from the return to reduce noise without changing the expected gradient; V(s) is the standard one, the group mean is GRPO's | E[∇log π]·b = 0 |
| 14 | **TD error δ_t** | one-step advantage estimate: r_t + γV(s_{t+1}) − V(s_t) | δ_2 = 0 + (−1.8) − 0.0 = −1.8 |
| 15 | **GAE (λ)** | advantage = Σ_k (γλ)^k δ_{t+k}; λ = 0 → only δ_t (trust V), λ = 1 → G_t − V(s_t) (trust the actual R) | `gae(rewards, V, mask, lam=0.95)` |
| 16 | **Credit assignment** | turning one outcome R into per-action A_t (D38) | — |
| 17 | **Policy gradient** | ∇J ≈ E[Σ_t A_t ∇log π_θ(a_t \| s_t)]: raise the log-prob of tokens with A > 0, lower those with A < 0 | `loss = −ρ·A` |
| 18 | **On-policy / off-policy** | on-policy: trained on samples from the current policy; off-policy: from another (older) policy or a dataset | PPO/GRPO on-policy; DPO offline |
| 19 | **Importance ratio ρ_t** | π_θ(a_t\|s_t) / π_old(a_t\|s_t): corrects for reusing samples after the policy has changed a little | `ratio = exp(logp − old_logp)` |
| 20 | **Clipping (ε)** | stop the gain once ρ leaves [1−ε, 1+ε] in the helpful direction: no big jump from one batch | `EPS = 0.2` |
| 21 | **KL penalty (β)** | cost for moving away from π_ref (D35) | `BETA = 0.05` |
| 22 | **Exploration** | trying tokens other than the most likely one, so new behaviours can be found and rewarded; in LLMs = sampling at temperature 1 | `temperature=1.0` |

## Key concepts

Filled in as each concept is discussed and understood. Status reflects the conversation, not
assessed mastery.

| Concept | One-line meaning | Status |
|---|---|---|
| Preference judgment | a *comparison*: which of several responses to the same prompt is better, under some criterion | discussed (D1) |
| Preference data sources | human annotators, implicit signals (votes), LLM judges, rated aspects | read (SLP3 §8.3.1); noise discussed (D3) |
| Bradley–Terry model | `P(a ≻ b) = σ(z_a − z_b)`: a score difference becomes a probability | discussing (D4, plan P1) |
| Reward model | a network scoring any (prompt, response) pair, trained on comparisons | discussed (D8); learner explained, refined (D15); separate copy from the chat model (D16) |
| Pairwise loss | `−log σ(r(x, o_w) − r(x, o_l))`, binary cross-entropy on the score gap | discussed (D9) |
| PPO | RL recipe: sample, score with the reward model, compare with a value model's expectation, take small clipped steps | introduced (D21); learner explained, refined (D34) |
| GRPO | RL recipe: sample a group per prompt, compare each answer with the group average; no value model | introduced (D21); learner explained, refined (D34) |
| DPO | offline recipe: raise chosen vs. rejected probability relative to a frozen reference; no reward model, no sampling | introduced (D21); learner explained, refined (D34) |
| ★ Loss mask on agent trajectories | the template decides which tokens are graded: only tokens the policy *chose* (tool-call turns, final answer, each closing `<|im_end|>`) get loss. System/user text, `<tool_response>` (a *user* turn in Qwen3), `<|im_start|>assistant\n` headers and template-inserted text (empty `<think>` when thinking is off, trailing `\n`) are context. Grading tool outputs would train the policy to invent results | **marked important by the learner** (D19, D26; NB04 Predict 4) |
| ★ `rm_scores`: how a reward model turns text into one number | render (chat template, ends at `<|im_end|>`) → right-pad + truncate → forward Qwen3 → head gives a number at *every* position → keep the one at the last non-pad token (HF: rightmost index with `id != pad_token_id`). Works because pad (`<|endoftext|>` 151643) ≠ `<|im_end|>` (151645) and rendering at train and score time match | **marked important by the learner** (D27, D28) |
| ★ Reward-model training process (NB04 Part C) | (1) build: Qwen3 + 1024→1 head = `rm`; (2) measure untrained: ≈50% held-out pair accuracy (Predict 7); (3) train: `AdamW(rm.parameters())`, 375 steps × 8 pairs of `−log σ(r_c − r_r)`, all weights, 1 epoch; (4) measure again: ≈62–66%; (5) freeze: `rm.requires_grad_(False)`, it only grades from then on. The chat model is trained later (Parts E–G) | **marked important by the learner**; learner stated it (D27, D29) |
| ★ Head swap: `AutoModelForSequenceClassification` | loading Qwen3 with `num_labels=1` keeps the transformer body but **discards `lm_head`** (1024 → 151,936 token logits; load report `lm_head.weight UNEXPECTED`) and **adds `score`** (1024 → 1, random init; `score.weight MISSING`). Result: one float per conversation, no token output, no `generate()`, no instruction prompt | **marked important by the learner** (D30) |
| ★ Choosing PPO / GRPO / DPO / RLVR | three separate axes: **data → scope** (any method can be general or domain-specific); **online vs offline → cost and learning from its own outputs** (PPO/GRPO generate + grade new answers; DPO doesn't); **reward source → verifiable or not** (RLVR = verifier; learned `rm` otherwise). GRPO is an update rule, used with `rm` or a verifier | **marked important by the learner**; learner synthesis refined (D33, D34) |
| ★ KL penalty to the reference | KL(π‖π_ref) = Σ_y π(y)·log π(y)/π_ref(y) per position, summed over the answer; ≥ 0, = 0 only if identical, asymmetric. RLHF maximizes E[R] − β·KL so the policy stays where `rm` is reliable. Estimated from sampled tokens: k1 = log π − log π_ref (can be < 0 per sample), k3 = e^d − d − 1 (≥ 0). PPO: −β·k1 in each token's reward; GRPO: β·k3 in the loss | **marked important by the learner**; described (D35), animation planned (P6) |
| Credit assignment | RL term: deciding how much each action (token) contributed to a single outcome (R). The quantity is the advantage A_t; the update is ∝ A_t·∇log π(a_t\|s_t). Options: uniform per answer (REINFORCE/GRPO), learned value + GAE (PPO), per-step graders (process reward models); SFT has no credit problem (every token has a target) | introduced (D37–D38); term new to the learner |

## Knowledge gaps

Questions the learner raised that aren't fully settled, and points to revisit.

- ★ **Important: grading and masking in the agentic template** (from D26, flagged by the learner).
  When a trajectory is trained with SFT, DPO, PPO or GRPO, the loss is summed *only* over masked-in
  tokens, so getting the mask right decides what the model learns. Check items:
  (1) the tool response is context, not graded;
  (2) Qwen3 wraps tool responses in a `user` turn, so a mask built from role headers, not from the
      word "tool", is what excludes them;
  (3) `<|im_end|>` is graded, because stopping is a decision;
  (4) text the template inserts (role headers, an empty `<think>` with thinking off) is template text,
      and a strict mask excludes it;
  (5) in DPO and RM scoring on trajectories, the log-prob/score sums must apply the same masking rule to both
      chosen and rejected (each gets its own mask, built the same way).
  Revisit when reading about agentic RL (Chapter 11+). Check that the learner can write the mask for
  a new template.

- **Whose values?** (from D2) Preference learning aligns a model to *the judges and their
  rubric*, not to "good" in general. Who writes the rubric, and how disagreement between groups
  is handled, is an open question to return to (links: Chapter 10 §10.4 on disagreement).

- **Where does a probability come from if each pair has one label?** (from D6) The ice cream
  example's "100 tasters per pair" didn't match real datasets, where each pair usually has a single
  label. Clarified in D6. Revisit to confirm the learner can explain it in their own words.

- **"Generated vs. ground truth" in preference learning** (from D10): the learner's model of
  cross-entropy is "prediction vs. ground truth", but it wasn't clear what plays each role here,
  partly because the *answers* were generated by an LLM earlier. Clarified in D10. Check that the
  learner can name both roles in their own words.
- **Root confusion found in D11: two different models.** The learner assumed the model being
  trained in §8.3.3 *generates* the chosen answer. It doesn't: the reward model is a separate
  *grader* that only reads and scores. Generation belongs to the chat model, trained later (§8.4)
  or directly by DPO. Check after §8.4 that the learner can separate grader and generator.
- **"No labels in RLHF, so how is anything graded?"** (from D20): the learner read D19's "prompts
  only" as "no grading at all". The labels moved: humans label pairs once (phase 1); in phase 2 the
  reward model *is* the label-maker, scoring each new answer. Also open: one score per answer vs.
  per-token credit (value model, GRPO group advantage). Check the learner can trace one prompt
  through phase 2.
  *Update (D31, NB04 Part D):* learner stated it in their own words: "no label is needed, only
  prompts, because we already have the reward model." Main part settled. Still to check: who writes
  the answers in phase 2 (the policy), and the per-token credit part (Part E).
  *Update (D37):* confusion resurfaced: "rm is answer-level, so how is grading token-level?" Answer:
  grading stays answer-level; token-level *credit* comes from V + GAE. Check the learner can say who
  produces each per-token number.

## Discussions

Numbered, in the order they happen.

### D1 · What does "preference" mean in "preference judgment"? (2026-10-27)

**Learner's question:** what does "preference" mean here?

**Explanation given:** a preference judgment is a **comparison**. Given one prompt and two or
more candidate responses, a judge (a person or a model) says which one is *better*. It's
relative ("A is better than B"), not absolute ("A scores 7/10" or "A is correct").

- **Key contrast with SFT:** SFT needs someone to *write* the ideal answer. A preference only
  needs someone to *recognize* the better one. SLP3: "preference judgments do not require people
  to know how to do something. We simply have to have an opinion about the end result."
  (Chapter 10's epigraph: you can judge a table without being a carpenter.)
- **Forms it takes (SLP3 §8.3):** pick one of a pair; pick one from N; rank N options; or accept /
  reject a single option.
- **Written** `(o_i ≻ o_j | x)`: "for prompt x, output o_i is preferred over o_j".
- **"Better" by what standard?** Not just personal taste. Judges usually follow a criterion such
  as helpfulness, honesty, harmlessness or correctness (Chapter 10 §10.1: a judgment needs a
  rubric). Without a stated criterion, judges may reward the wrong thing, e.g. length or
  confident tone.
- **Example (SLP3, from HH-RLHF):** prompt "Does garlic help with colds?": chosen "It can be
  helpful against colds, but may make you stink."; rejected "It might be one of the best natural
  antibiotics out there…". The chosen one is more careful and honest. Neither is a perfect
  answer, but one is clearly better.

### D2 · Can preferences be bad, or even evil? (2026-10-27)

**Learner's question:** so in theory, preferences can be bad or evil, right?

**Answer: yes.** Preference learning is value-neutral: the model learns *whatever the judges
prefer*. The method has no built-in notion of "good".

*Accidentally bad (the common case):*
- **Sycophancy:** people tend to prefer answers that agree with them or flatter them, so
  preference-trained models learn to tell users what they want to hear. Anthropic research (2023)
  documented this across models trained with human feedback.
- **Shortcuts:** judges may favor longer, more confident or nicely formatted answers over correct
  ones. A reward model then learns "long = good" (Chapter 10 §10.8, the shortcut experiment).
- **Narrow or biased judges:** a small or homogeneous annotator pool bakes its own opinions and
  blind spots into the model.

*Deliberately bad (rarer, but possible):* someone who controls the data could prefer harmful
answers. Research on **preference-data poisoning** shows that a small fraction of manipulated
comparisons can plant hidden behaviors in the trained model.

*Safeguards in practice:* written rubrics and annotator guidelines, annotator training and
quality audits, written principles for AI judges (e.g. Anthropic's Constitutional AI),
separate safety data, red-teaming, evaluations independent of the preference data, and the KL
"leash" that limits how far training can pull the model (§8.4).

**Takeaway:** preference learning aligns a model to *the judges and their rubric*. That makes
"who judges, and by what standard?" a central question, not a detail.

### D3 · Preference data is never 100% accurate. Does that hurt training? (2026-10-27)

**Context:** the learner finished SLP3 §8.3.1 "LLM Preference Data" (sources: trained human
annotators, implicit judgments from votes on Reddit/StackExchange, LLM judges such as GPT-4 for
UltraFeedback, and per-aspect Likert ratings).

**Learner's question:** whatever the source, the labels aren't 100% accurate. Does that affect
preference training, given the data isn't high quality to begin with?

**Answer: yes, it's noisy, and training is designed to tolerate *random* noise much better than
*systematic* bias.**

*How noisy:* even trained annotators often disagree. InstructGPT reported roughly 73% agreement
between its labelers. Votes reflect popularity, not correctness, and LLM judges have their own
biases (position, length, self-preference).

*Why it still works:*
1. **Random noise averages out.** Across thousands of pairs, wrong labels point in no consistent
   direction, while the true preference keeps pulling the same way.
2. **The model is probabilistic by design.** Bradley–Terry predicts `P(a ≻ b)`, not a certainty,
   so occasional disagreement is expected. With 7 votes for A and 3 for B, the best fit is a
   *finite* margin (σ(Δ) = 0.7), not infinite confidence (Chapter 10 §10.4, "disagreement carries
   information").
3. **Mostly-right direction is enough.** The reward only has to rank better answers higher more
   often than not.

*What it does cost:*
- **A ceiling on reward-model accuracy:** if humans agree only ~70–75% of the time, a reward model
  can't reliably exceed that on held-out human labels.
- **Systematic bias doesn't average out:** if judges consistently prefer longer or flattering
  answers, the bias is learned (D2; Chapter 10 §10.8).
- **Exploitation (reward hacking):** when a policy is optimized hard against an imperfect reward,
  it finds the reward's errors. Gao et al. (2022) measured this "reward-model overoptimization":
  the proxy reward keeps rising while true quality peaks and falls (Goodhart's law, Chapter 14).

*Common mitigations:* several judges per pair and majority or soft labels; dropping low-agreement
pairs; annotator quality control; swapping answer order for LLM judges; label-noise-aware losses
(e.g. "conservative DPO" assumes some labels are flipped); reward-model ensembles; and the KL
leash (§8.4), which stops the policy from drifting into regions where the reward is unreliable.


### D4 · Bradley–Terry, part 1: from "A beats B" to a probability (2026-10-27)

**Learner's request:** Bradley–Terry looks like the key technical explanation; discuss it
further (plan P1). Also flagged reward hacking for later (plan P2).

**Explanation given (part 1):**
- **The problem:** the data holds only facts like "A ≻ B". Training needs numbers.
- **The idea:** give every response a hidden **score** `z` (its quality), and let the probability
  that A beats B depend only on the **score gap**:
  `P(A ≻ B) = σ(z_A − z_B) = 1 / (1 + e^{−(z_A − z_B)})`.
- **Numbers:** gap 0 → 50%; gap 1 → 73%; gap 2 → 88%; gap 3 → 95%; gap −1 → 27%.
- **Analogy:** chess ratings. Players carry one number each, and the rating gap predicts how often
  the stronger one wins. Elo uses the same logistic idea, with a different scale.
- **Why a difference:** only *relative* quality matters. Adding the same constant to every score
  changes no probability (the "gauge", Chapter 10 §10.5).
- **Why the sigmoid:** it turns any gap into a probability in (0, 1), with gap 0 → ½ and
  `σ(d) + σ(−d) = 1`, so P(A ≻ B) + P(B ≻ A) = 1. SLP3's derivation shows the gap equals the
  **log-odds**: `z_A − z_B = log[P(A ≻ B) / P(B ≻ A)]`.
- **Learning it:** the scores aren't known, so a reward model `r(x, o)` is trained so that observed
  winners get higher scores: loss `−log σ(r(x, o_w) − r(x, o_l))`. That's binary cross-entropy:
  the same machinery as logistic regression, applied to a score gap.

**Check questions posed to the learner:** (1) z_A = 2, z_B = 0: what's P(A ≻ B)? (2) Add 10 to
both scores: what changes?

### D5 · Bradley–Terry with a simple example: ice cream taste tests (2026-10-27)

**Learner's request:** explain Bradley–Terry with a simple, easy-to-understand example.

**Example given:** each flavor has a hidden "tastiness score": chocolate 2, vanilla 1,
strawberry 0. A taste test asks 100 people to choose between two flavors, and the score gap
predicts the split:

| Taste test | Gap | Predicted share for the first flavor |
|---|---:|---:|
| chocolate vs. vanilla | 1 | ~73 of 100 |
| vanilla vs. strawberry | 1 | ~73 of 100 |
| chocolate vs. strawberry | 2 | ~88 of 100 |
| vanilla vs. vanilla | 0 | ~50 of 100 |

**Points drawn from it:**
1. **Never 100%:** even a much tastier flavor loses sometimes. People differ, and that's built
   into the model (links to noisy data, D3).
2. **Only gaps matter:** scores 102 / 101 / 100 give the same predictions.
3. **Learning runs backwards:** in reality we see only the votes. Training finds the scores that best
   explain them, e.g. if chocolate won 73 of 100 against vanilla, their gap should be ≈ 1.
4. **One number per flavor is efficient and generalizes:** three scores replace a table of
   every matchup, and a matchup never tested directly (chocolate vs. strawberry) can be predicted.
5. **Mapping to LLMs:** flavors → candidate answers; tasters → annotators or LLM judges;
   tastiness score → the reward model's output `r(x, o)`; taste-test results → preference pairs.

### D6 · "But one example doesn't have 100 variants, right?" (2026-10-27)

**Learner's confusion:** the ice cream example (D5) had 100 tasters per pair, but a real preference
dataset has one judgment per example, not 100. So where does the probability come from?

**Correction of the earlier example:** the learner is right. Real datasets usually have **one label
per pair**: one annotator picked a winner, once. The 100 tasters were only there to make the
probability visible. That was a misleading simplification.

**Explanation given:**
1. **The probability is a model of the judge, not something counted per pair.** One label is like
   one coin flip: from a single flip you can't read off the coin's bias. Bradley–Terry *assumes*
   that each label was drawn from `σ(gap)`.
2. **The evidence comes from many different pairs together.** Analogy: a weather forecaster says
   "70% rain". Each day it either rains or doesn't, a single outcome. But across all the days they
   said 70%, it rained on about 70% of them. The probability is checked across many single
   outcomes, never inside one.
3. **The reward model shares knowledge across pairs.** In LLM data almost every answer is unique
   and appears in only one pair, so per-answer scores can't be counted the way chess ratings are
   (each "player" plays one game). Instead, the reward model scores answers by their *content*,
   and one network scores them all. Across thousands of single-label pairs it learns which
   features tend to win (careful, correct, relevant), so similar answers get similar scores.
   Analogy: a food critic who tastes thousands of different dishes once each still learns what
   makes a dish good.
4. **What one label does in training:** the loss `−log σ(r_w − r_l)` nudges that pair's winner up
   and loser down a little. A wrong label nudges the wrong way, but the other pairs, which share
   the same network, pull back. That's why random noise mostly averages out (D3).
5. **When pairs do have several votes** (some datasets collect them), the vote share can be used
   directly as a soft target, e.g. 0.7 (Chapter 10 §10.2's `q`).

### D7 · Making SLP3's half-page Bradley–Terry derivation easy (2026-10-27)

**Learner's question:** SLP3 §8.3.2 has half a page of math deriving the Bradley–Terry sigmoid. How
can it be understood easily? Are animations possible?

**Explanation given:**
- **What the math does:** it isn't proving BT is true. It *assumes* "score gap = log-odds" and then
  solves that equation for the probability. Every line after the first is algebra.
- **Odds first:** odds = P(A wins) / P(B wins). "3 to 1" means P = 3/4. Log-odds is the log of that.
- **The derivation in two plain steps:** gap → odds by `odds = e^gap`; odds → probability by
  `P = odds / (odds + 1)`. Dividing top and bottom by `e^gap` gives `1/(1 + e^{−gap}) = σ(gap)`.
- **Worked number:** gap 1 → odds e ≈ 2.72 → P = 2.72 / 3.72 ≈ 0.73.
- **Line-by-line map of SLP3's algebra:** take exp of both sides; multiply out (1 − P); collect the
  P terms on one side; factor out P; divide. Each line undoes one operation.
- **The easier route (Chapter 10 §10.2):** give each answer a *strength* `e^score`; the win
  probability is A's share of the total strength, `e^{z_A} / (e^{z_A} + e^{z_B})`, which simplifies to
  the same sigmoid. Same result, no log-odds needed.
- **Animations:** existing Chapter 10 loop `preference-margin.gif` shows gap → probability →
  gradient. A new candidate, `CAND-ANIM-033`, would show *why*: strength bars `e^score` stacking
  into a share, odds, and the S-curve traced by a slider. Recorded in PROPOSALS.md; production
  needs the learner's approval and happens on the Mac Studio.

### D8 · Is the reward learned, i.e. does it exist only after training? (2026-10-27)

**Learner's question (SLP3 §8.3.3):** "We'll use this preference data and the Bradley–Terry formulation
to learn a function r(x, o)…": so this reward is something learned, available after training?

**Answer: yes.** `r(x, o)` is a neural network, the **reward model**. "Learning the function" means
training that network.

- **Humans never write reward numbers.** They only give comparisons (A ≻ B). The numbers are
  *invented by training*: the network's scores are adjusted until they explain the comparisons
  through Bradley–Terry.
- **Before training** its scores are meaningless. **During training**, each pair nudges the winner's
  score up and the loser's down (loss `−log σ(r(x, o_w) − r(x, o_l))`). **After training** it's frozen
  and can score any new (prompt, answer) pair, including answers no judge ever saw.
- **Architecture (SLP3 Fig. 8.10):** start from a pretrained LLM, remove the next-token head (which
  outputs ~150k vocabulary logits) and attach one linear layer that outputs **one number**.
- **Two separate trainings, in order:** (1) train the *reward model* on preference data
  (§8.3.3); (2) use the frozen reward model to train the *chat model* to produce higher-scoring
  answers (§8.4, RL with a KL leash). DPO (§8.4.2) later shows how to skip the separate reward model.
- **Other uses besides training (SLP3):** best-of-N (generate several answers, keep the
  highest-scoring), and filtering SFT data.

### D9 · How cross-entropy and Bradley–Terry connect (2026-10-27)

**Learner's request:** explain `L = −log P(o_w ≻ o_l | x) = −log σ(r(x, o_w) − r(x, o_l))` in plain
language: how are cross-entropy and Bradley–Terry connected?

**Explanation given:**
- **Cross-entropy, in one sentence:** "−log of the probability your model gave to what *actually*
  happened." Same rule as next-token training: there, `−log p(correct next token)`.
- **What "actually happened" here:** the judge picked the winner `o_w` over the loser `o_l`.
- **Bradley–Terry supplies the model's probability of that event:** `σ(r(x, o_w) − r(x, o_l))`.
- **Put together:** loss = −log(BT probability of the observed preference). BT turns scores into a
  probability; cross-entropy grades that probability against what the judge did.
- **Loss in numbers** (gap = winner score − loser score):

  | gap | P(winner wins) | loss |
  |---:|---:|---:|
  | +3 | 0.95 | 0.05 (confident and right: tiny loss) |
  | 0 | 0.50 | 0.69 (undecided) |
  | −2 | 0.12 | 2.13 (confident and wrong: big loss) |

- **The update it drives:** the gradient with respect to the gap is `−(1 − σ(gap))`. So training raises
  the winner's score and lowers the loser's, pushing hard when the model ranks the pair wrongly and
  barely at all when it's already confidently right.
- **Another view:** it's ordinary binary classification ("did the first answer win?") where the
  logit is the score gap, i.e. logistic regression. With a soft label q (e.g. 7 of 10 votes), the
  general binary cross-entropy `−q log σ − (1 − q) log(1 − σ)` reduces to SLP3's formula when q = 1.

### D10 · What is "predicted" and what is "ground truth" here? (2026-10-27)

**Learner's question:** cross-entropy compares what's generated with the ground truth. In preference
learning, what is generated and what is ground truth?

**Explanation given:** the learner's framing is right; the confusing part is that the *answers* aren't
the prediction.

| Role | Next-token training | Reward-model training |
|---|---|---|
| Input | the text so far | prompt + **both** answers (fixed, from the dataset) |
| Model's prediction | probabilities over the next token | the probability that A beats B, `σ(r(x, A) − r(x, B))` |
| Ground truth | the actual next token | **the judge's choice**: A is better, i.e. target probability 1 |
| Loss | −log P(actual next token) | −log P(judge's choice) |

- **The answers are inputs, not predictions.** In SLP3 the answers were sampled from an LLM *earlier*,
  when the dataset was built. During reward-model training they're fixed text. The reward model
  generates no text; it outputs two numbers, which become one probability.
- **Worked example (garlic):** ground truth = "chosen beats rejected". Untrained scores: chosen 1.2,
  rejected 1.5, so predicted P(chosen wins) = σ(−0.3) ≈ 0.43, compared with truth 1.0, giving
  loss −log 0.43 ≈ 0.85. Training raises the chosen score and lowers the rejected one until the
  prediction approaches the truth.

### D11 · "The model is still generating, right?" Grader vs. generator (2026-10-27)

**Learner's question:** using the garlic example: the model is still generating, just generating
the chosen answer instead of the rejected one? Where does the picking happen? And isn't it a
sequence-generation action rather than a simple pick?

**Diagnosis:** the learner merged two different models. Their intuition (generate the chosen
sequence, not the rejected one) is correct, but it belongs to a *later* step, not to §8.3.3.

**Explanation given: a cooking analogy with three phases.**

| Phase | Cooking | Preference learning | Who generates? | Who picks? |
|---|---|---|---|---|
| 0 · make the data | a cook makes two dishes; a famous critic tastes both and says "dish A is better" | an LLM wrote both garlic answers; a human chose "may make you stink" | the old LLM, *before* training | the **human** |
| 1 · train the grader (§8.3.3) | train a **robot taster** to agree with the critic: it tastes both dishes and gives each a score; if it scores B higher, it's corrected | train the **reward model**: it *reads* both answers and outputs two scores; loss `−log σ(r(chosen) − r(rejected))` | **nobody**: the robot only tastes, never cooks | the robot's "pick" = whichever it scores higher, compared with the critic's |
| 2 · train the cook (§8.4) | a student cook makes new dishes; the robot scores them; the student learns to cook what scores well | the **chat model** generates new answers token by token; the frozen reward model scores them; RL pushes toward high-scoring answers, with a KL leash | the **chat model**: full sequences | the reward model scores |

**Answers to the three questions:**
1. *Is the model generating?* Not in §8.3.3. The reward model is a reader/grader: whole text in,
   one number out. Generation happens in phase 0 (data) and phase 2 (training the chat model).
2. *Where does picking happen?* The human picked in phase 0. In phase 1 the reward model's
   "pick" is just which answer it scores higher; the loss checks that against the human's pick.
3. *Isn't it sequence generation?* For the chat model, yes, and that's exactly where the
   learner's intuition fits. In phase 2 the chat model generates whole sequences. And **DPO**
   (§8.4.2) does what the learner described: it trains the generator directly to make the chosen
   sequence more likely than the rejected one, where a sequence's probability is the product of its
   token probabilities, with no separate grader.


**Learner's follow-up (D11):** "So we trained a reward model, not an LLM?" **Yes, in §8.3.3.** One
nuance given: the reward model is usually *built from* an LLM (pretrained transformer, next-token
head replaced by a one-number head). It's LLM-shaped inside, but its job is grading, not generating,
and it's a separate copy from the chat model. The chat LLM itself is trained afterwards, in §8.4.
The learner's summary is correct.

### D12 · Then the reward model trains the LLM? (2026-10-27)

**Learner's question:** then we use the reward model to further train an LLM?

**Answer: yes.** This is RLHF (reinforcement learning from human feedback), SLP3 §8.4.

**The loop:**
1. take a prompt
2. the chat LLM (starting from the instruction-tuned / SFT model) **generates** an answer
3. the frozen reward model **scores** it
4. an RL update (classically PPO) makes high-scoring answers more likely and low-scoring ones less
   likely
5. repeat over many prompts

**Contrast with SFT:** no target answer is given. The model writes its *own* answers and learns only
from their scores, more like practice with a judge than copying from an answer key.

**The KL leash:** the objective is `reward − β · KL(new model ‖ starting model)`. The model is
penalized for drifting too far from where it started. Without that, it would forget its general
abilities and chase the reward model's mistakes (reward hacking, plan P2). SLP3 eq. 8.6–8.7.

**Alternative:** DPO (§8.4.2) skips the separate reward model and the RL loop, training the LLM
directly on the chosen/rejected pairs.

**Course:** Chapter 12 (generation as a policy, PPO), Chapter 11 (DPO), Chapter 14 (reward hacking,
KL as an anchor).

### D13 · "Training a reward model seems extremely important" (2026-10-27)

**Learner's observation:** training a reward model seems extremely important in post-training.

**Agreed, with nuance:**
- **Why it's central in RLHF:** the chat model optimizes *the reward model*, not human judgment
  directly. The reward model is the only channel from human preferences to the model, so its
  quality caps the result, and its mistakes get amplified (reward hacking, plan P2).
- **Labs invest heavily in it:** e.g. Llama 2 trained *separate* helpfulness and safety reward
  models, because one model struggled to balance both. Reward models are also used for best-of-N
  answer selection, rejection-sampling SFT data (Llama 3) and data filtering.
- **But not every method needs a separately trained one:**
  - **DPO** (§8.4.2) learns from preference pairs without a separate reward model. It optimizes an
    *implicit* reward hidden in the policy itself ("your language model is secretly a reward model").
  - **RLVR** (the third post-training family) replaces the learned reward model with a *program*
    that checks answers (math, code), so there's nothing learned to be fooled.
  - **LLM-as-judge**: a strong model prompted with a rubric can act as the reward, instead of a
    trained one.
- **Connecting the families:** what really matters across all post-training is the **reward
  signal**: where "better" comes from. Preference alignment learns it from human or AI
  comparisons; RLVR computes it with a checker. Its quality decides what the model becomes.

### D14 · Learner synthesis: "DPO and RLVR are predefined recipes; what matters is the data?" (2026-10-27)

**Learner's statement:** with a reward model (and preference data), DPO and RLVR are kind of
predefined recipes, and what matters is the data.

**Assessment: the core instinct is right; two categories needed untangling.**

*Right:* separate the **signal** (where "better" comes from) from the **recipe** (the algorithm that
uses it). Recipes like PPO, GRPO and DPO are fairly standard, and the signal largely decides
*what* the model learns. Same idea as topic 4 in three-post-training-families: RLVR names the
reward source, GRPO the optimizer.

*Corrections:*
1. **DPO and RLVR aren't the same kind of thing, and neither uses the trained reward model.**

   | Signal (what "better" means) | Recipe (how the model is updated) | Name |
   |---|---|---|
   | preference pairs → trained reward model | RL: PPO, GRPO | RLHF |
   | preference pairs, used directly | DPO's loss (no reward model, no sampling) | DPO |
   | a program that checks answers | RL: GRPO, PPO | RLVR |

   DPO is a recipe that skips the reward model. RLVR is a signal (a verifier) used with an RL recipe.
2. **The recipe still matters**, just less for *what* is learned than for *how well and how safely*:
   - **offline vs. online:** DPO learns only from fixed, pre-collected pairs; RL samples the model's
     *own* new answers and can explore beyond the dataset
   - **stability and cost:** PPO juggles several models and is finicky; DPO is simple and cheap
   - **resistance to exploitation:** the KL strength β and the training length decide how far the
     model can chase a flawed signal (reward hacking, plan P2)

**Refined statement:** the algorithm is a fairly standard recipe; the signal (preference data, a
reward model or a verifier) largely decides what the model learns; the recipe decides how
efficiently and safely that signal is used.

### D15 · Learner: "A reward model is a neural network that learns the patterns of what to prefer on a specific dataset?" (2026-10-27)

*Right in spirit.* Four refinements:

1. **What kind of network:** usually not a new small one. It's a **pretrained (often SFT) LLM** whose
   next-token output is replaced by a **one-number head**. It already understands language, and
   training only teaches it *taste*.
2. **What it outputs:** one **score** per (prompt, answer), not a preference. A preference only appears
   when two scores are compared, `σ(r_A − r_B)`. Only **gaps** mean anything: adding 10 to every
   score changes nothing (Bradley–Terry is shift-invariant).
3. **What "patterns of what to prefer" really is:** the **judges' taste as it shows in the data**,
   including their habits and biases (longer answers, confident tone, flattery). It's a **proxy** for
   "good", not "good" itself.
4. **"On a specific dataset" is how it's trained, not what it's for:** its job is to score **new
   answers it has never seen**, written by the chat model during RL. Outside the kind of answers in
   its training data, its scores can be wrong. The chat model can find and exploit those spots
   (reward hacking, plan P2).

*Everyday example:* a robot taster trained on 10,000 past critic verdicts. It's useful only if it can
also judge **new** dishes. If the critics always favored extra salt, the robot learned "salt = good",
and a student cook will learn to over-salt.

**Refined statement:** a reward model is a neural network, usually a pretrained LLM with a one-number
output head, that learns from many comparisons a scoring function reproducing the judges' taste. It
can then score new answers, but only as well as that taste and its training data allow.

### D16 · "So the reward model is the LLM itself, and gets further trained later?" (2026-10-27)

*Gap:* D15 said the reward model "starts from a pretrained LLM". The learner read that as **one
model** playing both roles. They're **two separate copies** that start from the same (or similar)
weights and then go different ways:

| | Reward model (grader) | Chat model (generator, the "policy") |
|---|---|---|
| Starts from | a copy of the pretrained/SFT LLM | another copy of the SFT LLM |
| Changed how | next-token head → one-number score head | keeps its next-token head |
| Trained when | §8.3.3, on preference pairs | §8.4, by RL using the reward model's scores |
| During RL | **frozen**: only scores | **trained**: this is the model that gets better |
| Shipped to users? | no | yes |

*Everyday example:* identical twins from the same cooking school. One becomes a **food critic**
(tastes and scores, never cooks), the other a **cook**. They share the same training up to
graduation, then the critic grades the cook's new dishes. The critic doesn't change while the
cook trains.

*Extra details:* the two needn't be the same size (InstructGPT used a 6B reward model for a 175B
policy). PPO-style RLHF holds up to four models at once: the policy (trained), a frozen reference
copy of the SFT model for the KL leash, the frozen reward model, and a value model (trained, PPO
only; GRPO drops it).

### D17 · How do you choose the reward model's base LLM? Is one reward model reused to train later LLMs? (2026-10-27)

**Choosing the base.** Common practice is the **same family as the chat model**, often its own SFT
or pretrained checkpoint:
- **Same knowledge:** the grader should know what the generator knows. Llama 2 initialized its
  reward models from its chat-model checkpoints, arguing that a knowledge mismatch could make the
  grader favor hallucinations (confident answers it can't check).
- **Same tokenizer and style:** it reads the generator's outputs natively.
- **Size is a trade-off:** a bigger reward model is harder to fool (Gao et al. 2022: larger reward
  models overoptimize less), but it's called on **every** sampled answer during RL, so it costs
  compute. InstructGPT used a 6B reward model for a 175B policy (it found 175B reward models
  unstable to train). Llama 2 used reward models the same size as the chat model, and two of them
  (helpfulness and safety).
- Ready-made open reward models (e.g. from public reward-model benchmarks) are also an option.

**Two-model setting?** RLHF is basically **grader + learner** (plus a frozen reference and, with PPO,
a value model). DPO is closer to a one-model setting: no grader, only the learner and a frozen
reference.

**Reused for later LLMs?** It *can* be: a trained reward model is just a scorer, so it can grade any
model's answers. But it works well only on answers like those it was trained on. As the chat model
improves, its answers drift away from that data and the scores become less reliable (an opening for
reward hacking). So labs **refresh** it in rounds: sample the current chat model, collect new human
comparisons on *its* answers, retrain the reward model, train the chat model again (Llama 2 did
several such rounds).

*Everyday example:* the critic twin graded beginner dishes for years. When the cook starts making
fancy fusion food the critic has never tasted, the critic's scores get shaky, so the critic needs
new tasting sessions too.

### D18 · Why same-family graders? Can one universal reward model grade every family? How do OpenAI and Anthropic do it? (2026-10-27)

**Why same family (it's mostly practical, not a law):**
1. **Knowledge parity:** the grader must know at least what the generator knows, or it can't catch
   confident mistakes. A lab's strongest knowledge sits in its own pretrained model.
2. **It's what a lab has:** one pretraining run, one tokenizer, one training stack. Starting the
   grader from it is cheap; licensing someone else's model as a grader is awkward.
3. **Data match:** the comparisons were made on that family's answers, so a backbone that "speaks the
   same way" fits them best.
Tokenizers are *not* a real barrier: a reward model reads text and re-tokenizes it with its own
tokenizer.

**A universal grader?** Possible, and it partly exists:
- open reward models (built on Llama, Gemma or Qwen backbones) are routinely used to score answers
  from *other* families; RewardBench-style benchmarks test them on answers from many models
- **LLM-as-judge**: a strong general model grading with a rubric is a universal grader
Limits: (a) **whose taste?** one fixed taste for everyone (D2); (b) **distribution shift**: new
models write answers it never saw; (c) **a fixed public target gets gamed**: optimize hard against
it and each policy finds its own weak spots (reward hacking, P2). So universal graders are good for
**evaluation and data filtering**; for heavy RL training, labs keep refreshing their own.

**OpenAI and Anthropic (public information only; current internals aren't published):**
- **OpenAI, InstructGPT (2022):** labelers ranked 4–9 answers per prompt; a 6B GPT-3-family reward
  model trained on all pairs from each ranking; PPO with a KL penalty. GPT-4 report: added
  **rule-based reward models** (a model checks answers against written safety rules). Later work:
  "Rule-Based Rewards" (2024) and deliberative alignment (a reasoning model judges against a written
  spec).
- **Anthropic, HH-RLHF (Bai et al. 2022a):** "preference models" from the same model family as the
  policy, up to 52B, with an extra **preference-model pretraining** stage on public comparison data
  (e.g. Stack Exchange votes) before fine-tuning on human labels.
- **Anthropic, Constitutional AI (Bai et al. 2022b) = RLAIF:** an AI model, guided by a written
  constitution, picks the better answer for harmlessness; the preference model trains on these AI
  labels plus human helpfulness labels. Anthropic publishes the constitution it trains Claude with.
- Shared trend: less pure "human clicks A or B", more **written principles or specs + AI judges +
  verifiers** (RLVR), with humans writing the rules and auditing.

### D19 · §8.4: what does the data look like (chat and agentic), and is it the same as reward-model data? (2026-10-27)

*Learner started SLP3 §8.4 (LLM alignment via preference-based learning).*

**Chat alignment, reward-model data:** `{prompt, chosen, rejected}`, e.g. the garlic example.
**Agentic, reward-model data:** the same shape, but each side is a whole **trajectory**:
task → (thought, tool call, tool result)* → final answer. The chosen one might call the right tool
with correct arguments and use the result; the rejected one calls a nonexistent tool, passes wrong
arguments, invents a tool result or stops early. Labels come from humans, an LLM judge, or the
**outcome** (did the task succeed? → that's a verifier, i.e. RLVR). Preferences can be on the
whole trajectory (outcome) or on single steps (process). Tool results are written by the
environment, not the model, so they're **masked out** of the policy's loss, like the `-100` user
tokens in SFT.

**Is the §8.4 data the same as reward-model data? Depends on the recipe:**

| Recipe | Data for training the chat model | Same as reward-model data? |
|---|---|---|
| RLHF (PPO/GRPO) | **prompts only** (tasks for agents); the model writes the answers, the frozen reward model scores them | **no**: no answers, no labels; prompts often from the same distribution (InstructGPT: ~33k prompts for the RM, ~31k *unlabeled* prompts for PPO) |
| DPO | `{prompt, chosen, rejected}` | **yes**: the same kind of data, since DPO folds the reward model into the chat model |
| RLVR | prompts/tasks + a checker (answer key, unit tests, task-success check) | **no**: no pairs at all |

*Why:* in RLHF the preference knowledge has already been **compressed into the reward model**; the
chat model's job is to explore and get scored, so it only needs questions. Answers in a fixed
dataset would be written by *another* model; RL needs the chat model's **own** answers (online).
DPO never builds a grader, so it must read the pairs directly (offline).

### D20 · "No answers and no labels in RLHF? Then how is anything graded? Token level?" (2026-10-27)

*Gap:* D19's "prompts only" sounded like "no grading". There are **two datasets for two phases**:

| Phase | Data | Who supplies the label |
|---|---|---|
| 1 · train the reward model (§8.3.3) | `{prompt, chosen, rejected}`, **labeled** | humans (or an AI judge), once |
| 2 · train the chat model (§8.4) | prompts only | the **frozen reward model**, live, for every new answer |

The reward model *replaces* the human labeler: humans label ~tens of thousands of pairs once, then
the reward model grades millions of fresh answers for free.

*One prompt through phase 2:* chat model writes an answer token by token → reward model reads
prompt + whole answer → the hidden state at the **last token** goes through the score head → one
number, e.g. 2.3 → update the chat model toward high-scoring answers.

*Token level?* The reward model gives **one score per whole answer** (sequence-level), not per
token. Turning it into credit per token is the RL recipe's job:
- the final score lands on the **last token**; every token also pays a small **KL penalty**
  (how far it strayed from the reference model)
- **PPO:** a value model guesses "how good is this answer so far?" at each token, spreading the
  final score back over earlier tokens
- **GRPO:** sample several answers per prompt; each answer's advantage = (its score − group mean) /
  group std, and all its tokens share it. No value model.
- **process reward models** score each *step* (useful for maths and agents), an optional extra

*Everyday example:* the critic tastes the finished dish and gives one score; they don't score each
spoonful of salt. Working out *which* step made the dish good is the cook's (the algorithm's) job.

### D21 · PPO, GRPO and DPO with easy examples (2026-10-27)

Same cooking school throughout: the **student cook** is the chat model, the **critic** the reward
model, and the **graduation-day cook** the frozen reference (SFT) model.

**PPO: the critic plus a coach, small steps.** The cook makes one dish; the critic scores it 7.
Before tasting, a **coach** (the value model) predicted "about 6". Advantage = 7 − 6 = +1: better
than expected, so do more of what you did. Score 5 → −1, do less. Two safety rules: change the
recipe only a little per lesson (**clipping**: the new probability of each token can't move more
than ~20% from the old one in one update), and don't drift far from your graduation-day style
(**KL leash**). Four models in memory: cook, critic, coach, graduation-day copy. Powerful but heavy
and fiddly.

**GRPO: cook a batch, compare within it.** The cook makes **8 versions** of the same dish; the critic
(or a checker: "is the maths answer right?" → 1/0) scores them, e.g. average 5. Version scoring 8 →
above average, reinforce; scoring 2 → below, discourage. Advantage = (score − group mean) / group
std. The **group is the baseline**, so no coach is needed. Keeps clipping and the KL leash. Cheaper
than PPO; popular for RLVR (DeepSeek-R1).

**DPO: study the old verdict notebook, no critic, no cooking.** The cook reads past verdicts ("dish
A beat dish B") and adjusts directly: become **more likely** to make A and **less likely** to make
B, *compared with the graduation-day cook*. In Bradley–Terry form, the "score" of an answer is
`β·log(π(o|x) / π_ref(o|x))`: how much more the cook now likes it than at graduation. Loss =
`−log σ(score(chosen) − score(rejected))`, the same pairwise loss as the reward model (D9), but
applied to the chat model itself. Offline: simple, stable, cheap, but can't explore beyond the
notebook.

| | Needs the reward model? | Cooks new dishes during training? | Baseline for "better than expected" | Models in memory |
|---|---|---|---|---|
| PPO | yes | yes, one per prompt | value model (coach) | 4 |
| GRPO | yes, or a checker | yes, a group per prompt | group average | 3 |
| DPO | no | no (fixed pairs) | the frozen reference | 2 |

### D22 · PPO's coach: what is it really, how does it know "6", and why not let it grade? (2026-10-27)

**What it is:** the **value model**, another LLM copy (often started from the reward model or the
SFT model) with a one-number head. At **every token** of a partial answer it predicts "the final
score this cook will probably get from here" (reward-model score minus the KL penalties still to
come).

**How it knows "6":** it doesn't at first. It's trained **alongside** the cook by plain regression:
after each answer is finished and scored, each of its predictions is compared with what actually
happened, with loss (prediction − actual)². Over many answers it becomes a decent forecaster of
*this* cook's results. It needn't be accurate: a baseline doesn't bias the update on average, it
only reduces noise. A bad coach makes learning slower and noisier, not wrong.

**Why not let it grade?** It learns **from** the critic's scores; it has no taste of its own. Remove
the critic and it has nothing to learn from. The two answer different questions:
- critic: "how good is this **finished** answer?" (fixed, the source of truth; trained on human
  pairs)
- coach: "how well is **this student** likely to do, **from this point**?" (changes as the student
  improves)
Like betting odds vs. the final scoreboard: odds are useful forecasts, but they're learned from past
scoreboards and can't replace the match result.

**Its second job, credit per token:** the *change* in its prediction from one token to the next
shows which token helped. Garlic: after "Raw garlic is" → 5; after "toxic" → 1. That drop puts the
blame on "toxic", even though the critic scored only the whole answer (D20). (PPO uses GAE to do
this smoothly.)

### D23 · The value model in plain technical terms (2026-10-27)

*Learner feedback:* the coach / betting-odds metaphor in D22 was confusing; asked for plain
technical language. Restated:

**Setup.** Prompt x; policy π_θ samples tokens a_1…a_T. State s_t = (x, a_<t). Per-token reward
r_t = −β·[log π_θ(a_t|s_t) − log π_ref(a_t|s_t)] (KL penalty), plus the reward-model score R(x, a)
added at t = T. Return G_t = Σ_{k≥t} r_k.

**Policy gradient with a baseline.** ∇_θ J = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · (G_t − b(s_t)) ]. Any b(s_t)
that doesn't depend on a_t keeps it unbiased, since E_{a~π}[∇ log π(a|s)]·b(s) = b(s)·∇Σ_a π(a|s) = 0.
A good b lowers variance; the best is close to V^π(s_t) = E[G_t | s_t].

**Value model.** V_φ: a transformer with a scalar head, giving one output per token position. It's
trained by regression on the same rollouts as the policy: L_V = (V_φ(s_t) − Ĝ_t)², with Ĝ_t the
observed (or GAE) return. It's inaccurate at the start and improves as rollouts accumulate.

**Advantage (GAE).** δ_t = r_t + V(s_{t+1}) − V(s_t); A_t = Σ_k (γλ)^k δ_{t+k}. With λ = 1,
A_t = G_t − V(s_t) (unbiased, high variance); with λ < 1 it leans on V (lower variance, some bias if
V is wrong). A_t is how the single end-of-sequence score is spread across tokens.

**PPO update.** ρ_t = π_θ(a_t|s_t)/π_old(a_t|s_t);
L = −E[min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) A_t)] + c·L_V.

**Why V can't replace the reward model:**
1. V's regression target is built from R; the reward model is the only external signal in the loop.
   Using V as the reward makes the target its own output: circular, with no information about quality.
2. V estimates the expected return of the **current policy** from a prefix, averaged over that policy's
   continuations. It describes the policy, and it changes every iteration. R scores a completed
   response against a fixed, human-derived criterion.
3. V is an approximation of returns built on R. Optimizing against it would chase its fitting errors,
   on top of R's.

GRPO replaces V(s_t) with the group mean of R over G samples for the same prompt (normalized by the
std): one advantage per sequence, no per-token baseline.

### D24 · "What is the coach? The chat model itself?" Worked example (2026-10-27)

*Gap:* the word "coach" (D21–D22) never got tied to a concrete network. It's the **value model**
V_φ, a **separate network**, not the chat model. Networks in PPO-RLHF:

| Network | Input → output | During RL |
|---|---|---|
| policy π_θ (the chat model) | prefix → next-token distribution | trained |
| reference π_ref (SFT copy) | prefix → next-token distribution, for the KL term | frozen |
| reward model R | prompt + full response → 1 number | frozen |
| value model V_φ (the "coach") | prefix → 1 number at every position | trained |

V_φ is usually initialized from the reward model (InstructGPT did this). Some implementations put a
value head on the policy's own transformer instead of using a separate copy; it's still a separate
output trained with a separate loss.

**Worked example** (KL ignored, γ = 1). Prompt "Is it safe to eat raw garlic?" The policy samples
"No, raw garlic is toxic." R = −2.0.

| Prefix s_t | V(s_t) | Token a_t | δ_t = V(s_{t+1}) − V(s_t) (last: R − V) |
|---|---|---|---|
| (prompt only) | 0.5 | "No" | −0.4 |
| …"No" | 0.1 | ", raw garlic is" | −0.1 |
| …"is" | 0.0 | "toxic" | **−1.8** |
| …"toxic" | −1.8 | "." | −0.1 |
| …"." | −1.9 | end | −0.1 |

- λ = 0: A_t = δ_t, so the blame lands mostly on "toxic".
- λ = 1: A_t = R − V(s_t), so every token gets a large negative value (e.g. "No": −2.5); no localization.
- GAE λ ≈ 0.95 sits in between.

Policy update: lower log π(a_t|s_t) in proportion to −A_t (clipped). Value update: regress V(s_t)
toward the observed return (−2.0); e.g. V(prompt) = 0.5 is pulled down, since this policy produced −2.0.
Where V's numbers come from: earlier rollouts of the same policy on similar prompts.

### D25 · "Is the value model a snapshot of the current policy?" (2026-10-27)

*Right:* the policy is the chat model being updated.
*Correction:* V is **not** a snapshot (a weight copy) of the policy. It's a separate network that
outputs a **number**, not token probabilities, and is trained by **regression on returns** to
**predict the policy's outcomes**. It *tracks* the policy, but it's not a copy of it.

The snapshots in PPO are policy copies:

| | What it is | Refreshed |
|---|---|---|
| π_θ | the policy being updated | every gradient step |
| π_old | weight copy of π_θ taken when the rollouts were sampled; used in ρ = π_θ/π_old for clipping | every iteration (each new batch of rollouts) |
| π_ref | weight copy of the SFT model; used for the KL penalty | never |
| V_φ | separate scalar-output network: prefix → expected return under the current policy | trained every iteration with (V − return)² |

Why V must be retrained continually: V(s) = E[return | s] *under the current policy*. When the
policy improves (it stops writing "toxic"), the same prefix leads to better continuations, so the
correct V(s) rises. A V fitted to the old policy would then be stale.

### D26 · ★ [NB04 · Part A] "Is this output of the trajectory mask correct?" (2026-10-27)

The learner pasted the Predict 4 output: GRADED `'ool_call>…</tool_call><|im_end|>'`, context
`'…<tool_response>…</tool_response><|im_end|>\n<|im_start|>assistant\n'`, GRADED
`'<think>\n\n</think>\n\nTomorrow in Paris…<|im_end|>'`, context `'\n'`.

*Answer: yes, correct.* Line by line:

- `'ool_call>'` is not a bug. The print shows only the last 110 characters (`s[-110:]`), so the
  `<t` of `<tool_call>` was cut off. The whole tool call is graded. The first context segment
  (system prompt, tool schema, user question) isn't printed, because the loop starts at
  `segments[1:]`.
- The tool turn is context. The model didn't write `{"high_c": 14, "low_c": 8}`; the tool did. If it
  were graded, the policy would be trained to *invent* tool results, which is hallucination.
- `<|im_start|>assistant\n` is context. The serving code inserts it at each turn start; the model
  never chooses it.
- `<|im_end|>` is graded. Emitting it is the model's decision to stop.
- The final `'\n'` is context. The template adds it after `<|im_end|>`; generation has already stopped.

*Nuance:* the empty `<think>\n\n</think>\n\n` is marked graded, but with `enable_thinking=False`
Qwen3's template puts it into the generation prompt, so at inference it is template text, not model
output. A stricter mask would exclude it. Including it costs almost nothing, because those tokens are
near-certain (log-prob ≈ 0) and contribute a tiny gradient. With thinking on, the reasoning inside
`<think>` *is* model-written and graded.

### D27 · [NB04 · Part C] Explain the core lines of the reward-model training loop (2026-10-27)

The learner pasted the Part C training cell and asked for the core lines.

**One step** (batch of 8 pairs):

1. `prompts = [...] * 2` and `answers = chosen + rejected` build 16 conversations. Rows 0–7 are
   the chosen answers, rows 8–15 the rejected answers *to the same prompts*.
2. `rm_scores(...)`: each row is rendered with the chat template, which ends at `<|im_end|>`.
   Rows are right-padded to the batch length and cut at 320 tokens. One forward pass through
   Qwen3 + the 1024→1 head gives a scalar per row, read at the last non-pad token. Output: `[16]`.
3. `loss = -logsigmoid(scores[:8] - scores[8:]).mean()` is the Bradley–Terry loss on the 8 gaps,
   i.e. binary cross-entropy with target "chosen wins" (D9, Predict 5).
4. `loss.backward()`: each pair's gradient is −(1 − σ(gap)) on its chosen score and
   +(1 − σ(gap)) on its rejected score, back-propagated into *all* weights: the head and every
   transformer layer.
5. `clip_grad_norm_(…, 1.0)` rescales the whole gradient vector if its norm exceeds 1, so one bad
   batch can't cause a huge step. `opt.step()` is AdamW at lr 1e-5; `zero_grad()` clears.
6. 3000 / 8 = 375 steps, one pass (one epoch), each pair seen once.

**Around the loop:** `train_pairs[:20000]` vs `[20000:]` keeps the RL prompts unseen by the RM.
After training, `rm.eval(); rm.requires_grad_(False)` freezes the reward model, which from then on only
grades (D16).

*Nuance flagged:* `truncation=True, max_length=320` cuts from the right, so on a long conversation
the score is read before `<|im_end|>`, part-way through the answer.

### D28 · ★ [NB04 · Part C] "The logic of `rm_scores` is important: show the full code" (2026-10-27)

Shown in full: `render` → `rm_scores` → Hugging Face's `GenericForSequenceClassification.forward`
(transformers 5.16.1), which does the pooling.

Step by step for a batch:

1. `render(prompt, answer)`: chat template with `enable_thinking=False`, trailing `\n` stripped, so
   the text ends on `<|im_end|>`, the same token a sampled answer ends on in PPO/GRPO. Train-time
   and score-time text must match. An earlier bug read the score at `\n` during training but at
   `<|im_end|>` during RL.
2. `tok(..., padding=True, padding_side="right", truncation=True, max_length=320)` gives
   `input_ids [B, T]` and `attention_mask [B, T]`; shorter rows are filled with pad id 151643.
3. `model(**enc)`: the transformer runs → `last_hidden_state [B, T, 1024]` → `self.score` (a
   1024→1 linear layer, no bias) → `logits [B, T, 1]`, i.e. a number at every position.
4. Pooling: `non_pad_mask = input_ids != pad_id`; `last = (arange(T) * non_pad_mask).argmax(-1)`
   gives the rightmost real token. `pooled = logits[arange(B), last]` → `[B, 1]`.
5. `.logits.squeeze(-1).float()` → `[B]`, cast to fp32 for the loss. The batches are concatenated.

Why the last token: with causal attention, only the last position has attended to the whole
prompt + answer. The notebook asserts per-position[-1] == returned score.

Failure modes to remember:
- If pad id == `<|im_end|>`, the pooling would skip the real end token.
- If no pad id is set, HF raises an error for B > 1 (the notebook sets `rm.config.pad_token_id = PAD`).
- Right truncation at 320 drops the end of long conversations, so the score is read mid-answer.
- The same function serves training (with grad) and evaluation (`pair_accuracy` under `no_grad`).

### D29 · ★ [NB04 · Part C] "Is Predict 7's code training the reward model or the chat model?" (2026-10-27)

*Answer: neither. It doesn't train anything. It **measures the reward model before training**.*
`pair_accuracy` runs under `@torch.no_grad()` and `model.eval()`. It scores 500 held-out test pairs,
counts how often s(chosen) > s(rejected), and reports ≈ 47.6–50% (chance), because the new head is random.
It defines the measuring tools (`rm_scores`, `pair_accuracy`); the training loop comes in the next
cell (D27) and reuses `rm_scores` *with* gradients.

Which model is trained where:
- Part C trains the **reward model** (`rm`, `AutoModelForSequenceClassification`, outputs one number).
- Parts E–G train the **chat model** (policy, `AutoModelForCausalLM`, outputs tokens), via PPO and
  GRPO (using the frozen `rm` as the grader) and via DPO (no `rm`).

How to tell from the code: the variable is `rm`, not `policy`, and there is no `opt.step()` in the
cell.

*Clarified by the learner:* they meant the cell **after** Predict 7 ("Training: the pairwise loss on
real pairs"). Answer: that cell trains the **reward model**. `rm` is the model being optimized
(`AdamW(rm.parameters())`), the loss is the Bradley–Terry pairwise loss on its two scores, and the
chat model doesn't exist yet in Part C. "All weights train" means all ~0.6 B weights of `rm`
(Qwen3 body + new head), not the chat model. After the loop, `rm` is frozen and becomes the grader
for Parts E–F. It was a location mix-up, not a grader/generator confusion.

*Learner confirmed:* "ok, it is the process of training a reward model." ★ Marked important by the learner.

> ★ **Reward-model training, five steps:** build (`rm` = Qwen3 + 1024→1 head) → measure untrained
> (≈50%) → train (`AdamW(rm.parameters())`, pairwise loss, 375 steps) → measure again (≈62–66%)
> → freeze (`rm.requires_grad_(False)`). Only `rm` changes in Part C; the chat model is trained
> in Parts E–G.

### D30 · ★ [NB04 · Part C] "Is there a prompt to the reward model? It's an LLM, so can't it generate any token?" (2026-10-27)

*First part: right.* It reads (prompt + answer) and outputs one score.

*Second part: no instruction prompt, and it cannot generate tokens.*
- **Input:** just the rendered conversation (`render(prompt, answer)`). There's no "Please rate this
  answer" text. The "prompt" in the data is the *user's question*, part of what is scored, not an
  instruction to the reward model. It never needs to be told its task, because the head + training
  *are* the task.
- **Why it can't generate:** a causal LM ends in `lm_head`, Linear(1024 → 151,936), one logit per
  vocabulary token, which is what lets it pick a next token. `AutoModelForSequenceClassification`
  drops `lm_head` (load report: `lm_head.weight UNEXPECTED`) and adds `score`, Linear(1024 → 1)
  (`score.weight MISSING`, newly initialized). The output is a real number, e.g. 1.37, not the
  token "7". There's no vocabulary distribution, so no sampling and no `generate()`.
- What it keeps from the LLM: the transformer body (embeddings + 28 layers), i.e. its understanding
  of text. Only the output layer changed: from "which token next?" to "how good is this?".

*Contrast: generative reward model / LLM-as-a-judge.* A normal chat LLM *is* given an instruction
prompt ("Rate the response 1–10: …") and *generates* text such as "Score: 8 because …", which is then
parsed.
UltraFeedback's GPT-4 ratings (Part A) and Constitutional AI / RLAIF judges work this way. That is a
different design: slower (generation per score), parseable output can fail, but rubrics can change
without retraining. Scalar reward models (this notebook, InstructGPT, Llama 2) are what PPO/GRPO
usually call thousands of times per iteration.

> ★ **Marked important by the learner: the head swap.** When we loaded
> `AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1)`, Hugging Face replaced
> the output layer. The load report in Part C shows it:
> `lm_head.weight | UNEXPECTED` (the 1024 → 151,936 token layer in the checkpoint, discarded) and
> `score.weight | MISSING` (the new 1024 → 1 layer, randomly initialized). So the reward model is
> the same transformer body with a different last layer. It outputs a number and cannot generate
> tokens. The random `score` head is also why Predict 7's untrained accuracy is ≈50%.

### D31 · [NB04 · Part D] Learner synthesis: "RL on a chat model sometimes needs no labels, only prompts, because we already have the reward model" (2026-10-27)

*Correct.* Refinements recorded:

1. **The labels didn't disappear; they moved earlier.** Humans labelled pairs once (phase 1, Part C).
   That knowledge now lives in `rm`'s weights. In phase 2, `rm` produces a fresh label (a score) for
   every new answer.
2. **The answers come from the policy itself.** A phase-2 row is only a question string. The chat
   model writes the answer during training (rollout), `rm` scores it, and the update uses that score.
   This is why it is *on-policy*: it is graded on its own current outputs, not on dataset answers.
3. **"Sometimes"**, i.e. it depends on the recipe:

| Recipe | Training data for the chat model | Where the grade comes from |
|---|---|---|
| PPO / GRPO with a reward model (RLHF) | prompts only | learned `rm` |
| RLVR (math, code) | prompts + reference answer or tests | rule-based verifier (no `rm`) |
| DPO | (prompt, chosen, rejected) pairs | the labels themselves; no `rm`, no generation |
| SFT | (prompt, answer) | the target tokens |

Risk that follows (lead-in to plan P2, reward hacking): the policy is graded only by `rm`, so it
learns whatever `rm` rewards, including `rm`'s biases (length, +0.40 in Part C), not human
preference itself.

### D32 · [NB04 · Part D] PPO, GRPO, RLVR, DPO again: why each does or doesn't need labels (2026-10-27)

Key correction to the framing: these aren't four parallel options. Two questions are mixed:
**(a) where the reward comes from** (learned `rm` · rule-based verifier · none) and **(b) how the
weights are updated** (PPO · GRPO · DPO loss). RLVR answers (a); PPO and GRPO answer (b). So
"GRPO + RLVR" (e.g. DeepSeek-R1) and "PPO + `rm`" (InstructGPT) are both valid combinations.

| | Reward source | Update rule | Labels needed at training time | Generates during training? |
|---|---|---|---|---|
| PPO + `rm` (RLHF) | learned `rm` | clipped ratio × advantage from value model (GAE) | prompts only; labels were used earlier to train `rm` | yes |
| GRPO + `rm` | learned `rm` | clipped ratio × group-relative advantage, no value model | prompts only | yes, G per prompt |
| PPO/GRPO + verifier (= RLVR) | code: answer match, unit tests | same as above | prompt + reference answer / tests; no human preference | yes |
| DPO | none | `−log σ(β[(log π−log π_ref)(c) − (log π−log π_ref)(r)])` | (prompt, chosen, rejected) pairs | no, offline |

Why, in one line each:
- **RL with `rm`:** the policy writes new answers nobody has labelled. A grader is needed for unseen
  text, and `rm` generalizes from the old labels → no labels at RL time.
- **RLVR:** correctness of math/code can be checked by a program, so no learned grader is needed. But
  the checker needs the ground truth (the final answer or tests) → per-prompt reference, not preferences.
- **DPO:** no sampling, so nothing new to grade. It trains directly on the labelled pairs and puts the
  "reward" inside the policy (implicit reward β·log π/π_ref) → it needs the pairs themselves.
- **PPO vs GRPO** differ only in the update rule (how the advantage is computed: value model vs
  group mean/std). Their label needs are the same; they depend on the reward source, not on the
  algorithm.

### D33 · [NB04 · Part D] Which tasks suit PPO, GRPO, DPO, RLVR? (2026-10-27)

Choice made by three questions, in order:
1. **Can a program check the answer?** → yes: RLVR (verifier reward), usually optimized with GRPO.
2. **If not, do you have preference pairs, and is offline enough?** → DPO.
3. **Need on-policy improvement and can afford generation + a reward model?** → RLHF with `rm`
   (PPO or GRPO).

| Task | Typical choice | Why |
|---|---|---|
| Math with a final answer, code with tests, SQL, format/JSON compliance | RLVR + GRPO (DeepSeek-R1; Tülu 3's RLVR stage used PPO) | the reward is exact and can't be gamed the way a learned `rm` can. GRPO: no value model, and G samples per prompt suit 0/1 rewards |
| Open-ended chat: helpfulness, tone, style, refusals | DPO first; RLHF with `rm` if more is needed (InstructGPT, Llama 2: PPO; Llama 3: DPO) | "better" has no checker, only human judgment, so preference data is needed |
| Small team / small budget, existing pair dataset | DPO | 2 models, no generation, simple supervised-style loss |
| Agents with checkable outcomes (tests pass, task completed) | RLVR on trajectories (GRPO-style); mask tool tokens (D26) | the outcome can be checked; the path can't be written as one gold answer |
| Safety / policy adherence with written rules | rule-based or judge reward (OpenAI RBR, Anthropic CAI/RLAIF) + RL, or DPO on AI-labelled pairs | rules → an AI judge can produce labels at scale |

Trade-offs:
- **DPO:** cheapest in compute (no sampling); offline, so it learns only from answers in the
  dataset, not from its own mistakes; can drift when the pairs come from another model.
- **PPO:** most general and most expensive (4 networks, value-model tuning).
- **GRPO:** drops the value model; needs G samples per prompt; zero signal when all G get the same score.
- **RLVR:** reliable reward, but only for verifiable tasks; sparse 0/1 reward → needs prompts the model
  solves sometimes, not never or always.

Real pipelines combine them: SFT → DPO or RLHF for chat quality → RLVR for reasoning (Tülu 3:
SFT → DPO → RLVR). Notebook memory caveat: in NB04, DPO peaked *higher* (24.6 GiB) than PPO (18.3) and
GRPO (14.6), because each step back-propagates through the policy on 16 full-length sequences
(8 chosen + 8 rejected) plus a reference forward pass. "Cheaper" for DPO means no generation, not necessarily less memory.

### D34 · ★ [NB04 · Part D] Learner synthesis on when to use DPO, PPO, GRPO/RLVR (2026-10-27)

Learner: "DPO is very domain specific, maybe suitable for limited workflows in a specific company. PPO
is very general, but the reward model for PPO needs to be general too, so it is expensive.
GRPO/RLVR is very good for tasks where the solution can be verified."

1. **DPO is domain-specific: partly right.** *Right:* cheap and simple, so a good fit for a company
   with pairs from its own workflow. *Correction:* DPO isn't tied to a domain. Its scope = its data's
   scope. On broad pairs it gives a general chat model (Zephyr on UltraFeedback; Llama 3's
   post-training). Its real limit is being **offline**: it learns only from answers in the
   dataset, never grades its own new answers.
2. **PPO is expensive because the RM must be general: partly right.** *Right:* PPO is general.
   *Correction:* the RM's data cost isn't what separates it from DPO. Both use the same kind of
   pairs (NB04 Part G trains DPO on the exact pairs the RM used). PPO's cost is **compute**:
   generation at every step and 4 networks (policy, ref, rm, value). What the RM buys is being
   **on-policy**: it grades new answers the policy writes, so the policy learns from its own outputs.
3. **GRPO/RLVR for verifiable tasks: right for RLVR, imprecise for GRPO.** RLVR means "reward from
   a verifier", so yes, by definition. GRPO is an update rule and works with a learned `rm` too
   (NB04 Part F runs GRPO on `rm` scores for chat). It is *popular* with RLVR because group comparison
   suits 0/1 rewards and it saves the value model.

> ★ **Marked important by the learner.** Refined summary: **data decides the scope** (any method); **online vs offline decides the cost and
whether the model learns from its own mistakes**; **the reward source decides verifiable vs not**.
> Common mix-ups: DPO ≠ domain-specific; PPO's cost = compute (online), not RM data; GRPO ≠ RLVR-only.

### D35 · ★ [NB04 · Part E] KL to the reference model, in detail (2026-10-27)

Learner: "we need to address KL understanding by fine description and animation as well."

**1. Definition, one token position.** Given the same prefix, the policy and the reference each output
a distribution over the vocabulary. KL(π‖π_ref) = Σ_y π(y)·log[π(y)/π_ref(y)]: the log-ratio of
each token, averaged with weights π. Properties: ≥ 0; = 0 only if the distributions are identical;
not symmetric.

Worked example (3-token vocabulary): π_ref = [0.5, 0.3, 0.2], π = [0.7, 0.2, 0.1].
Log-ratios: +0.336, −0.405, −0.693. KL = 0.7·0.336 + 0.2·(−0.405) + 0.1·(−0.693) = **0.0851**.

**2. Over an answer:** sequence KL = sum over the answer's positions of the per-position KL, taken
along answers the policy itself samples.

**3. Why RLHF uses it.** Objective: maximize E[R(x, y)] − β·KL(π‖π_ref). `rm` was trained on answers
near the SFT model's distribution. Far from it, `rm` scores are unreliable and can be exploited
(plan P2). KL keeps the policy close to where `rm` is valid and keeps the fluency of the SFT model.
β sets the trade-off. The optimum has a closed form, π*(y) ∝ π_ref(y)·exp(R(y)/β), which is the
starting point of DPO's derivation.

**4. Estimating it from samples.** The exact sum over 151,936 tokens at every position is never
computed. Each sampled token gives an estimate:
- **k1 = log π(y) − log π_ref(y)** (NB04 `ppo_rewards`). Unbiased; *can be negative* for one
  sample (tokens B, C above). Its mean over π-samples = 0.0851.
- **k3 = e^d − d − 1, d = log π_ref − log π** (NB04 GRPO). Values 0.051, 0.095, 0.307, always
  ≥ 0, mean over π-samples also = 0.0851, lower variance.
- Both are valid only if the tokens are sampled **from π itself**. The NB04 temperature-0.6 bug
  sampled from a sharpened distribution, so the k1 mean stopped estimating KL and went negative.
  Fixed with `temperature=1.0`.

**5. Where it enters.**
- **PPO:** in the **reward**, per token: r_t = −β·k1_t (+R at the last token). So it also flows
  through GAE into the advantages. NB04: β = 0.05; logged KL 0.095 → 0.318 over 40 iterations.
- **GRPO:** in the **loss**: clipped surrogate + β·mean(k3), β = 0.04. The advantage stays the pure
  group z-score.
- **DPO:** no explicit KL term. The β·log(π/π_ref) inside its loss *is* the KL-regularized objective,
  solved in closed form.

### D36 · [NB04 · Part E] "Is the PPO loss token-level? How? Is it assured by the four networks?" (2026-10-27)

*Yes, token-level.* NB04 `ppo_policy_loss`:
`loss = −Σ_t mask_t · min(ρ_t A_t, clip(ρ_t, 0.8, 1.2) A_t) / Σ_t mask_t`. There is one ratio and
one advantage per response token, averaged over all real response tokens in the batch.

*How: the formulation makes it possible, and the networks supply the per-token tensors.*
1. **Formulation (the root cause):** generation is an MDP where each token is an action.
   State s_t = prompt + tokens so far, action a_t = next token. Autoregressive factorization:
   log π(y|x) = Σ_t log π(a_t|s_t). A per-answer quantity splits exactly into per-token terms.
2. **Per-token tensors, each `[B, T]` over response positions:**

| Quantity | Comes from | Code | Per token? |
|---|---|---|---|
| log π_θ(a_t\|s_t) | policy, one forward pass, logits at P−1+t gathered at token P+t | `token_logps(policy, ro)` | yes |
| log π_old(a_t\|s_t) | stored at rollout | `old_logp` | yes |
| log π_ref(a_t\|s_t) | reference | `token_logps(ref, ro)` | yes |
| R | reward model | `score_rollout` → `[B]` | **no, one per answer** |
| r_t = −β(log π_old − log π_ref) + R·[t = last] | combined | `ppo_rewards` | yes (R placed on the last token) |
| V(s_t) | value model, head read at every position | `values_of` | yes |
| A_t, return_t | GAE over r_t and V | `gae` | yes |

3. **The value model is what makes the advantage differ token by token.** Without V, a single
   score R would give every token the same credit (as in GRPO). With V, δ_t = r_t + V(s_{t+1}) − V(s_t)
   measures how much token t changed the expected outcome (the garlic example: λ = 0 blames "toxic").

So the four networks don't "assure" token-level by themselves. Three of them produce per-token outputs
(policy, ref, value), and the reward model's single score is converted to per-token signal by placing it
on the last token and propagating it with GAE. The *mask* restricts all sums to response tokens:
prompt and padding get no loss.

Also token-level: the value loss Σ_t mask_t·(V(s_t) − return_t)² / Σ mask, and the advantages are
whitened over all masked tokens in the batch. Contrast GRPO: still a per-token loss (one ratio per
token), but A is the same for every token of an answer.

### D37 · [NB04 · Part E] "The reward model is answer-level, so how does grading work on token level?" (2026-10-27)

*The learner is right: the reward model is answer-level.* It is trained on (prompt + full answer)
and gives one number. It never grades individual tokens. (Its head *does* output a number at every
position, D28, but only the last one was trained. The others are meaningless and never used.)

*Resolution:* "grading" and "credit" are different steps done by different networks.
- **Grading = reward model, answer-level:** one R per answer, placed on the last token.
- **Credit assignment = value model + GAE, token-level:** splits that one R across the tokens.

How V can be token-level when the only label is answer-level: V is trained by regression, from
every prefix of an answer to that answer's final return:
`(V(s_t) − return_t)²` at every t, where return_t ≈ R (γ = 1, plus the small KL terms). Over many
rollouts, prefixes that tend to end badly get low V. One answer-level label becomes a training target
for every prefix. This is the same move as the RM learning per-answer scores from pair-level labels.

Worked (garlic, D24, λ = 0): tokens "No" · ", raw garlic is" · "toxic" · "." · `<end>`;
r = [0, 0, 0, 0, −2.0] (only the last token carries R); V = [0.5, 0.1, 0.0, −1.8, −1.9].
A_t = δ_t = r_t + V(s_{t+1}) − V(s_t) = [−0.4, −0.1, **−1.8**, −0.1, −0.1].
"toxic" gets the blame because V *dropped* there (0.0 → −1.8), not because the RM graded that token.

Chain: humans label pairs → RM learns answer scores → RM gives R per rollout → V learns per-prefix
predictions of R → δ_t compares consecutive predictions → per-token advantage → per-token loss.
Without V (GRPO): every token of an answer gets the same advantage.

### D38 · [NB04 · Part E] "'Credit'? First time I see it. Explain more" (2026-10-27)

The learner hadn't met the term. It was introduced in D37 without a definition.

**Definition.** *Credit assignment* is the RL problem of deciding how much each action contributed
to an outcome when the feedback arrives only once, at the end. (Term from Minsky, 1961. *Temporal*
credit assignment is the case where the actions happen in sequence over time.) In an LLM: action =
token, outcome = the reward-model score R of the whole answer. An answer with 120 tokens and one R
means 120 decisions sharing 1 number.

**The quantity that carries credit is the advantage A_t.** The policy-gradient update for token t
is ∝ A_t · ∇ log π(a_t | s_t):
- A_t > 0: token t did better than expected → raise its probability in that context.
- A_t < 0: worse than expected → lower it.
- A_t ≈ 0: no change.
So "assigning credit" = deciding the vector [A_1, …, A_T] from the single R.

**Why it is hard.** R is *sparse* (one number per answer) and *delayed* (it arrives after the last
token). Most tokens ("the", "is") had no effect on R. Giving them the same A as the decisive token
adds noise: their probabilities get pushed up or down at random across many answers.

**Ways to assign it:**

| Method | Credit per token | Cost / trade-off |
|---|---|---|
| SFT | not needed: every token has its own target token | needs gold answers |
| REINFORCE / GRPO | the same A for every token of an answer (GRPO: group z-score of R) | no extra network; noisy, so many samples are needed (G per prompt) |
| PPO + GAE | A_t from V: δ_t = r_t + V(s_{t+1}) − V(s_t), mixed over k steps with (γλ)^k | needs a value model; λ trades bias (λ → 0, trust V) vs variance (λ → 1, trust the actual R) |
| Process reward model (PRM) | a grader scores *each reasoning step* (Lightman et al. 2023, "Let's Verify Step by Step") | needs step-level labels |
| DPO | implicit: Σ_t β·log π/π_ref over tokens; whatever tokens differ between chosen and rejected get pushed | no explicit per-token credit, uses the pair's contrast |

Garlic, two methods on the same answer (R = −2.0, mean score of other answers = 0):
- Uniform (GRPO-style, before dividing by the group std): A = [−2, −2, −2, −2, −2] for "No", ", raw garlic is", "toxic", ".", `<end>`.
  "No" and "." are pushed down as much as "toxic".
- PPO + GAE (λ = 0, V from D24): A = [−0.4, −0.1, −1.8, −0.1, −0.1]. Most of the push lands on "toxic".

Link to earlier questions: D20 ("token level?"), D22–D25 (why a value model exists), D37 (grading
vs credit).

### D39 · ★ [NB04 · Part E] "What is the advantage? We need to first explain all the related RL terms in the context of LLMs" (2026-10-27)

The learner noticed a foundation gap: RL vocabulary (credit, advantage, return, value, …) had been
used since D21 without being defined in order. A glossary of 22 terms was built (section "RL terms
in the LLM setting", above Key concepts).

**Advantage, in detail.** A(s, a) = Q(s, a) − V(s).
- V(s): the average return from s, averaged over all tokens the policy might pick next.
- Q(s, a): the return from s if this particular token a is picked.
- A > 0: a is better than the policy's average choice in s → raise π(a|s). A < 0 → lower it.
  It is *relative*: a good outcome can still give A < 0 if the model usually does even better from s.
- Garlic at s_2 = "No, raw garlic is": V(s_2) = 0.0. Picking "toxic" → Q ≈ V(s_3) = −1.8 → A = −1.8.
  An illustrative alternative, "safe for dogs in small amounts", might have Q = +1.0 → A = +1.0.
- Why subtract V instead of using R directly: if every answer to a prompt scores around +3, using R
  would push *all* tokens up. Subtracting the baseline leaves only "better or worse than usual",
  which has the same expected gradient and much less variance.
- How it's estimated in practice: Q isn't known, so δ_t = r_t + V(s_{t+1}) − V(s_t) stands in for
  it (one-step), or GAE mixes several steps. GRPO estimates it per answer as (R − group mean)/std.

