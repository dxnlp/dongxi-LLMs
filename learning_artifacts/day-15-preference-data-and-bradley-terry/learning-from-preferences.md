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

## ★ Study path (approved 2026-10-29, D40)

Two axes: **feedback source** (human pairs · reward model · LLM judge · process RM · verifier) ×
**optimizer** (PPO, GRPO and relatives; DPO as the RL-free loss).

| Stage | Topic | Course | Status |
|---|---|---|---|
| S1 | Reward models: BT, training, bias audit, process labels | Day 15, Day 16; [S1 notebook](../../notebooks/study-path/s1_reward_models.ipynb) | **in progress**: S1 notebook Part A (2026-10-30) |
| S2 | KL-regularized optimum → DPO | Day 17, Day 18; [S2 notebook](../../notebooks/study-path/s2_kl_optimum_and_dpo.ipynb) | — |
| S3 | REINFORCE → baselines/RLOO → PPO → algorithm comparison | Days 19–21; [S3 notebook](../../notebooks/study-path/s3_policy_gradient_to_ppo.ipynb) | — |
| S4 | GRPO | Day 22; [S4 notebook](../../notebooks/study-path/s4_grpo.ipynb) | — |
| S5 | RLVR (GRPO + verifier) | Day 23; [S5 notebook](../../notebooks/study-path/s5_rlvr.ipynb) | — |
| S6 | Combinations: RM-labelled iterative DPO, RM + verifier mix, SFT → DPO → RLVR | [S6 notebook](../../notebooks/study-path/s6_combinations.ipynb) (new) | — |
| S7 | When optimization goes wrong: reward hacking, length/entropy | Day 24 (plan P2); [S7 notebook](../../notebooks/study-path/s7_failure_modes.ipynb) | — |

Workbooks (built 2026-10-30 at the learner's request: "seven notebooks, one per stage, with good
exercises and granular code blocks"): [study-path README](../../notebooks/study-path/README.md). Each
runs toy first, then Qwen3-0.6B; 74 fill-in exercises with checks, 26 predictions. All trained models
use fp32 master weights. While building them, a precision problem was found: AdamW on bf16 weights
loses most small updates. It affected Day 15 · 04 too; **fixed 2026-10-30** (float32 master weights,
bf16 autocast, new Part C cell showing the rounding). Re-run, bf16 → fp32: reward-model accuracy
62.2% → 62.2% (unchanged); PPO KL, last 5 iterations, 0.32 → 7.9 nats; PPO eval reward
+0.03 ± 0.12 → +0.34 ± 0.11 (≈ 1.8 SE, suggestive); GRPO within noise (KL 0.03); DPO held-out implicit
accuracy 57.3% → 62.7%, margin +0.07 → +0.43, with chosen up (+0.27) and rejected down (−0.16).

### S1 notebook study log (started 2026-10-30)

Questions raised while working through the S1 notebook go into **Discussions** with an `[S1 · Part X]`
tag; requests for visuals go into the plan table.

| Part | Status | Exercises / predictions | Discussions and requests |
|---|---|---|---|
| A · Bradley–Terry: the loss and its gradient | studied | — | beginner Bradley–Terry animation, plan P9 → `CAND-ANIM-041`; D41 (plain explanation), D42 (score head; backprop + AdamW); article section + animation, plan P10 → `CAND-ANIM-042`; D43 (the loss is always computable from a labeled pair), D44 (scores vs rubrics), D45 (why the output is a number before training), D46 (score range; logit), D47 ★ (chat logits vs reward scores; plan P11 ★ → `CAND-ANIM-043`), D48 ★ (the numbers through the network, step by step; key point of `X-POST-005`) |
| B · linear reward model with a known truth | studied | — | D49 (what "Bradley–Terry is the true model here" means), D50 ★ (the reward model learns `w` from noisy labels: two layers of noise; random noise vs systematic bias), D51 ★ (is Bradley–Terry the only model used in practice? no: alternatives), D52 ★ (gradient descent recap and how it trains a reward model; key point 3 of `X-POST-005`), D53 (random flips and calibration, plainly, with counts), D54 (the same with metaphors, at the learner's request: scorekeeper, weather forecaster) |
| C · from a sequence to one number | in progress | — | D55 ★ (Part C revisits D28, D30, D42, D47 ★, D48 ★; new: a score at every position, padding, causal attention, bf16 noise); animation set for the whole article, plan P12 → `CAND-ANIM-044`–`051` |
| D · train a Qwen3 reward model | not started | — | — |
| E · length audit | not started | — | — |
| F · outcome and process labels | not started | — | — |

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
| P8 ★ | **Deep-study path: feedback sources × optimizers** | the learner, 2026-10-29: "I want to do deep study in this area, which path should we go?" | D40; course Days 15–25 (Chapters 10–14) | proposed: S1 reward models (D15–16) → S2 KL optimum + DPO (D17–18) → S3 REINFORCE → baselines → PPO (D19–21) → S4 GRPO (D22) → S5 RLVR (D23) → S6 combinations (new: RM-labelled iterative DPO, mixed RM + verifier rewards, Tülu 3-style pipeline) → S7 failure modes (D24, plan P2); **approved by the learner 2026-10-29: "ok, follow this path"**. Current position: S1 (Day 15 · 04: Predict 5, Parts E–G, then Day 16) |
| P9 | **Animation: Bradley–Terry for beginners** | the learner, at S1 Part A (2026-10-30): "we should produce a beginner friendly animation to explain Bradley–Terry" | S1 Part A (Exercises 1–2); Chapter 10 §10.2–10.5; D4–D7, D9 | proposed as `CAND-ANIM-041`: one pair, a person's label, two scores, the gap (shift doesn't matter), gap → σ → probability, loss, equal-and-opposite training pushes, then many pairs on the curve. Entry point before 033–035; state *suggested*, produce on the Mac Studio with them |
| P10 | **Article section + animation: chat-model head vs reward-model head** | the learner, at S1 Part A after D42 (2026-10-30): "the header difference between a chat model and a reward model deserves an explanation in the article and an animation" | D28, D30, D42; Day 15 · 04 Part C; S1 Part C | required section of `X-POST-005`, which the learner set (2026-10-30) to be **solely about reward models** and to be written from this note (plan `publications/x-articles/x-post-training-005/plan.md`); animation `CAND-ANIM-042`: one 28-block body, LM head `1024 → 151,936` (sample, loop) swapped for a `1024 → 1` score head (one pass, last-token number), two answers → two scores → the pick; state *suggested*, produce on the Mac Studio |
| P11 ★ | **Article section + animation: chat-model logits vs reward-model scores** | the learner, at S1 Part A after D47 (2026-10-30): "the above is important, we need to explain this well in article and animation" | D47 ★, D48 ★, D46, D41, D42 | required section of `X-POST-005` (reward models), right after the head-swap section; animation `CAND-ANIM-043`: one softmax over 151,936 tokens from one pass vs one softmax over 2 answers from two passes; same `−log softmax(correct option)` loss; raw score used as the reward. Companion to 042; state *suggested*, **high priority**, produce on the Mac Studio |
| P12 | **Animation set covering the whole reward-model article** | the learner, at S1 Part C (2026-10-30): "suggest more animations, so it can support the whole article" | `X-POST-005` plan sections 1–8; D1–D3, D42, D50 ★, D52 ★, D53–D55 ★, P2 | proposed `CAND-ANIM-044` (opening: one click → one row), 045 (comparisons vs grades), 046 ★ (random noise vs systematic bias), 047 (calibration), 048 ★ (which position gives the score), 049 (training at scale), 050 (length and reward hacking preview), 051 (one reward model, three uses). With 033–035 and 041–043: 14 candidates; the plan marks a core of 7 (041, 035, 042, 048, 043, 046, 050) for the learner to choose; state *suggested*, produce on the Mac Studio |
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
| ★ The numbers through the network: every logit is `h · one head row` | tokenize → embed → 28 layers → final hidden vector `h` (1024) → head. Chat model: 151,936 rows → 151,936 logits → softmax → pick one token ID → detokenize → repeat. Reward model: same body, **one** row `w` → one logit `w · h` = the score, used raw as the reward; softmax/σ over two answers' scores only inside the training loss | **marked important by the learner** (D48 ★); key point of `X-POST-005` |
| ★ Noisy labels: random noise vs systematic bias | the reward model learns `w` only from (chosen, rejected) labels, which are noisy twice: each clean label is one Bradley–Terry draw (thousands of pairs pin down `w`), and judges corrupt some. **Random flips** cancel in the gradient: direction kept, scale shrinks → under-confident (S1 Part B, ε = 0.2: `w` ≈ (0.72, 0.18), ratio 0.25, ranking 99.9%). **Systematic bias** adds up in the gradient: direction changes → the bias is rewarded (length judge, ε = 0.2, ≈ 8% labels changed: ratio 0.25 → 0.53, ranking ≈ 92%); more data doesn't fix it | **marked important by the learner** (D50 ★; also D3) |
| ★ Reward-model families beyond Bradley–Terry | BT (`−log σ(r_c − r_r)`) is the common *assumption* on real data, not a known truth. Alternatives: BT + margin (Llama 2), BT with ties, Plackett–Luce (rankings; = BT at K = 2), pointwise regression on ratings (HelpSteer2), multi-objective heads (ArmoRM), general pairwise preference P(A > B) from both answers (PairRM, NLHF; can fit A > B > C > A), generative judges (keep the LM head), process reward models (per step), verifiers (no learned model). DPO assumes BT too | **marked important by the learner** (D51 ★); rows from memory, verify before publishing |
| ★ Which position gives the score | the score head runs at every position (`w · h_t`, `[B, T]`); the reward is the value at the last real token `<|im_end|>`, the rest discarded. Last, because causal attention makes `h_t` depend only on tokens 1…t: a one-word change alters outputs only from that word on (S1 Predict 2). Find it under padding with `(arange(T) · non_pad).argmax` (left and right padding; `sum − 1` fails for left). bf16 moves a score by ≈ 0.008 | **marked important by the learner** (D55 ★, D28); `X-POST-005` section 4a |
| ★ Chat logits vs reward-model scores | Bradley–Terry is a softmax over the 2 compared answers, `σ(a − b) = softmax([a, b])₀`; chat logits are a softmax over the whole vocabulary from one pass, reward scores come from separate passes, have no absolute meaning, and the raw score is the reward; both train with `−log softmax(correct option)` | discussed (D47 ★) |
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

### D40 · ★ Learner framing: "DPO, PPO, GRPO are separate optimization algorithms; RM and RLVR are where the judge feedback comes from". And which deep-study path? (2026-10-29)

*Framing: right, with one refinement.* Two axes (as in D32/D34):
- **Feedback source:** human pairs (offline labels) · learned reward model (scalar) · LLM judge /
  generative RM · process reward model · verifier (RLVR = "RL with a verifier as the feedback source").
- **Optimizer:** PPO, GRPO (and relatives: REINFORCE, RLOO, DAPO). These are *RL* algorithms:
  they need samples + a reward from some source.
- **DPO is a different kind of optimizer.** It isn't an RL algorithm that takes a reward. It is a
  supervised loss derived from the KL-regularized RL objective (π* ∝ π_ref·exp(R/β), D35) that
  *removes* the RM and the sampling. Its feedback is the pair labels directly. So "DPO + RM" doesn't
  mean "DPO optimizes RM scores". It means the RM (or a judge) *labels new pairs* sampled from the
  policy, then DPO trains on them (iterative / online DPO; e.g. Self-Rewarding LMs, Yuan et al. 2024).

> ★ **Study path approved by the learner (2026-10-29), marked important.** Follow S1 → S7 in
> order; record progress against these stage IDs. Current position: S1.

*Path (follows the course order, Days 15–25, with one added stage):*

| Stage | Topic | Course | Why here |
|---|---|---|---|
| S1 | Reward models: BT, training, bias audit, process labels | D15 (done in part), D16 | every later stage uses a feedback source |
| S2 | KL-regularized optimum → DPO | D17, D18 | DPO's derivation needs only the KL objective (D35), not PPO |
| S3 | REINFORCE → baselines/RLOO → PPO (clip, KL, critic) → algorithm comparison | D19, D20, D21 | builds the policy gradient from scratch; the glossary (D39) applied |
| S4 | GRPO: group advantages, token ratios, weighting/filtering | D22 | PPO minus the critic; needs S3 |
| S5 | RLVR: verifier contracts, reasoning tasks | D23 | swap the feedback source: GRPO + verifier |
| S6 | Combinations (new, not in the course yet) | — | RM-labelled iterative DPO; RM + verifier mixed rewards; SFT → DPO → RLVR (Tülu 3) |
| S7 | When optimization goes wrong: reward hacking, length/entropy | D24 (plan P2) | needs all of the above |

Change vs the learner's draft: DPO comes *before* PPO (not alongside), because its derivation is
self-contained given the KL objective, and the course is built that way. RLVR is a feedback source,
so it is studied *with* GRPO (S5), not after all the optimizers in the abstract.


### D41 · [S1 · Part A] "Explain Bradley–Terry in plain language with easy examples" (2026-10-30)

**Learner's request:** at S1 Part A, a plain-language explanation of Bradley–Terry with easy examples
(after asking for the beginner animation, plan P9 / `CAND-ANIM-041`). D4–D7 covered the first pass;
this one adds learning scores back from votes, a three-answer example and the model's assumptions.

**Explanation given:**
- **Input and output.** Input: only comparisons ("A preferred over B"). Output: one score per item,
  chosen so that the **score gap** predicts how often one item is preferred:
  `P(A ≻ B) = σ(s_A − s_B) = 1 / (1 + e^{−(s_A − s_B)})`.
  Gap −2 → 0.12 · −1 → 0.27 · 0 → 0.50 · +1 → 0.73 · +2 → 0.88 · +3 → 0.95. Never exactly 0 or 1.
- **Example 1, scores → probabilities.** Prompt "How do I boil an egg?": A (clear steps and times)
  score 2, B (correct but vague) 1, C (wrong) −1. Then P(A ≻ B) = σ(1) = 0.73, P(B ≻ C) = σ(2) = 0.88,
  P(A ≻ C) = σ(3) = 0.95. Three scores give all three pairwise probabilities, consistently.
- **Example 2, votes → scores (what training does).** 10 annotators compare A and B; 7 pick A.
  The loss over the votes is `−7 log σ(Δ) − 3 log σ(−Δ)`; its derivative `−7(1 − σ(Δ)) + 3σ(Δ)` is 0 at
  σ(Δ) = 0.7, so the fitted gap is `Δ = log(7/3) = 0.85`. The 7 votes push the gap up, the 3 push it
  down, and they balance where the predicted probability equals the observed win rate. If all 10 pick
  A, nothing balances and the gap keeps growing: maximum likelihood is overconfident on unanimous,
  small data.
- **A real use of the same model: chess Elo.** Elo predicts the expected score with base 10 and
  scale 400: a 200-point gap → `1 / (1 + 10^{−200/400}) = 0.76`.
- **In a reward model.** The item is (prompt, answer); its score is `r(x, y)`, the reward model's
  output. Each pair contributes `−log σ(r_chosen − r_rejected)`: gap −1 → loss 1.31 and a push of
  size 0.73 on each score; gap +3 → loss 0.05, push 0.05. A step raises the chosen score and lowers
  the rejected one by equal amounts.
- **Only the gap carries information.** Adding 10 to every score changes nothing, so a score has no
  absolute meaning; compare scores of answers to the same prompt.
- **Assumptions (limits).** (1) All items lie on one line: preferences A ≻ B, B ≻ C, C ≻ A (a cycle)
  can't be represented, and the fit puts the three near equal scores. (2) No ties: datasets drop them
  or need an extended model. (3) One standard for everyone: annotator disagreement appears as
  probabilities near 0.5, not as separate groups (D3).

### D42 · [S1 · Part A] "To pick rather than generate, we replace the head with a classification head? And which algorithm optimizes the loss?" (2026-10-30)

**Learner's synthesis:** training a reward model always needs labeled data, so we know when it ranks a
pair right or wrong; because the reward model is a neural network (an LLM of the same family), its
picking has to be turned into a loss that can be optimized. Correct, with one refinement: the labels
are **relative** (which of two answers is better), not absolute grades.

**Q1 · Replace the head?** Yes, for the standard scalar reward model:
- The LM head (`1024 → 151,936` logits, one per vocabulary token) is discarded and a new, randomly
  initialized linear layer `1024 → 1` is added (`AutoModelForSequenceClassification(num_labels=1)`;
  the name says "classification", but with one output it is a **score head**: one unbounded real
  number, no classes, no softmax). D30.
- The model doesn't generate and doesn't pick. One forward pass reads (prompt, answer), and the head
  is read at the last non-pad token: one score per answer (D28). The **pick** is the comparison of two
  separately computed scores, `r(x, y_A) > r(x, y_B)`.
- Other designs keep the LM head: an LLM judge or generative reward model reads the probability of a
  verdict token ("A" vs "B") or writes a verdict; DPO's implicit reward uses the policy's own
  log-probabilities. The scalar head is the standard for RLHF reward models.

**Q2 · Which algorithm optimizes it?** Ordinary supervised training: **backpropagation + AdamW**, the
same optimizer as pretraining and SFT. No RL is involved.
1. Forward the chosen and the rejected answer through the same network (shared weights) → `r_c`, `r_r`.
2. Loss `ℓ = −log σ(r_c − r_r)`, averaged over the batch (8 pairs in S1 / Day 15 · 04).
3. `loss.backward()`: autograd starts from `∂ℓ/∂r_c = −(1 − σ(Δ))`, `∂ℓ/∂r_r = +(1 − σ(Δ))` (S1
   Exercise 2) and applies the chain rule through the head and every transformer layer.
4. Clip the gradient norm to 1.0, then `opt.step()`: AdamW keeps, per weight, a running mean of the
   gradient `m` and of its square `v`, and updates `w ← w − lr · m̂ / (√v̂ + ε) − lr · λ · w`. Each weight
   moves by roughly `lr` per step (1e-5 here), whatever the gradient's raw size. That is why bf16
   weights lost the updates (spacing 1.2e-4 near 0.02) and the trained models now keep fp32 weights.
5. One pass over the data (one epoch, as in InstructGPT); more epochs overfit the noisy labels.

**Two levels of "algorithm".** The **loss** defines what is wanted (Bradley–Terry here; later the PPO,
GRPO or DPO loss). The **optimizer** (AdamW, via backprop) changes the weights to lower it. PPO and GRPO
are needed for the policy because a sampled answer's reward can't be differentiated through the
sampling; their losses are still minimized with AdamW. The reward model needs no RL at all, because its
loss is a differentiable function of its own outputs on fixed data.

### D43 · [S1 · Part A] "Because we know the chosen and the rejected answer, we can always calculate the loss, right?" (2026-10-30)

**Learner's synthesis:** for one comparison, knowing which answer is chosen and which is rejected
means the Bradley–Terry loss can always be computed. **Correct.** `ℓ = −log σ(r_chosen − r_rejected)`
needs only the two scores and the label, and is finite and positive for any finite scores.

**Refinements given:**
- **Computed whether the ranking is right or wrong, and continuous.** Gap −3 → 3.05 · −1 → 1.31 · 0 →
  0.69 · +1 → 0.31 · +3 → 0.05. A correctly ranked pair still has loss > 0 and still pushes the gap
  wider. Accuracy (0/1) has zero gradient almost everywhere, which is why training uses this loss.
- **The model doesn't pick.** It scores each answer separately; the label states the correct order;
  the loss measures how well the two scores agree with it (D42).
- **Computable is not the same as correct.** A mislabeled pair (D3) gives a valid loss that pushes the
  wrong way; averaging over many pairs lets correct labels dominate, and noise caps accuracy.
- **Ties** have no chosen answer: drop them, or use a soft target 0.5,
  `−½ log σ(Δ) − ½ log σ(−Δ)`, minimized at Δ = 0. **Stability:** compute with `F.logsigmoid`
  (S1 Exercise 1, gap −200).

### D44 · [S1 · Part A] "Are the scores based on a rubric?" (2026-10-30)

**Answer given:** not in a Bradley–Terry reward model. Its score is learned; a rubric can shape it only
indirectly, through the labels.
- **No rubric in the model.** The input is (prompt, answer) only, with no instructions (D30); the only
  signal is "A ≻ B". The score means "predicted to win comparisons": the number that makes σ(gap)
  match the observed preferences. It has no named components and no absolute zero (shift invariance).
- **A rubric enters through the labelers.** Written guidelines shape annotators' choices, and the model
  learns from the choices. HH-RLHF (S1's data) gave crowdworkers a short "pick the more helpful
  response" instruction and left the judgment largely to them (Bai et al. 2022).
- **So it learns whatever predicted the labels, intended or not:** in the Day 15 · 04 fp32 run the score
  gap correlated with the length gap at +0.25, though no rubric asked for length (S1 Part E audits this).

**Where explicit rubrics are used:** annotator guidelines (→ pairwise labels → BT reward model); an LLM
judge with a rubric in its prompt (UltraFeedback: GPT-4 rated 1–5 on instruction-following,
truthfulness, honesty, helpfulness, then pairs were built from the ratings); Constitutional AI (written
principles applied by an AI labeler, then a BT reward model); per-attribute reward models (one rating
per criterion, e.g. HelpSteer); rubric/checklist rewards (per-prompt checks graded by an LLM judge,
weighted sum); verifiers in RLVR (the rubric written as code, S5).

**Trade-off:** BT learns from cheap comparisons and can capture unwritten preferences, but its score
can't be decomposed into reasons; a rubric is transparent but covers only what was written, and a
rubric-following judge adds its own biases (position, length).

### D45 · [S1 · Part A] "How does the reward model know to output a score before training, without an instruction?" (2026-10-30)

**Answer given:** it doesn't know anything; the architecture guarantees the output is one number.
- **Architecture, not instruction.** The score head is one linear layer, `score = w · h = Σᵢ wᵢ hᵢ`
  over the 1024-dim hidden state `h` at the last token. Any input text gives some `h`, so the output
  is always exactly one number; it can't emit tokens. A chat model needs a prompt because its output
  is chosen from 151,936 tokens; the reward model's output format is fixed by the layer.
- **Untrained = meaningless.** `w` starts small and random, so the score is an arbitrary mix of
  features: held-out accuracy 47.6% before training in Day 15 · 04 (chance). S1 Part D measures
  untrained vs trained scores (`sc0` vs `sc1`).
- **Training gives it meaning.** Bradley–Terry gradients change `w` (and the body) until chosen answers
  score higher: 62% after 3,000 pairs.
- **The pretrained body supplies the features.** Its 28 layers already encode topic, fluency, whether
  the question is addressed, tone and length in `h`; training mainly learns which combination predicts
  the labels. That is why a few thousand pairs suffice, and why a random body wouldn't work.

### D46 · [S1 · Part A] "Is there a range of the score? Is it a logit?" (2026-10-30)

**Range: none built in.** The score is the raw output `w · h` of the linear head, with no sigmoid,
softmax or clipping after it, so it is any real number. In practice it stays within a few units
because only gaps matter (Day 15 · 04 fp32 run: held-out gap std 1.67).
- **Shift is free:** +10 to every score changes no gap or probability; the zero point is meaningless.
- **Scale is learned:** ×2 doubles every gap and makes every prediction more confident, so training
  fixes the scale where σ(gap) best matches label consistency (noisy labels → small gaps; unanimous
  labels → growing gaps, D41). S1 Exercise 1 shows both rows.

**Logit?** In the loose ML sense (raw pre-activation output; Hugging Face's field is `.logits`), yes. In
the strict sense (log-odds), the **gap** is a logit: `σ(gap) = P(chosen wins)`, so `gap = log(P/(1 − P))`;
gap 0.85 → odds e^0.85 = 2.33 → 70 : 30. A single score is a **log-strength** (original BT form
`P = e^a / (e^a + e^b)`), not a log-odds, because its zero point is arbitrary.

**Why the scale matters later:** PPO optimizes `R − β·KL`, so β converts nats into reward units; a
reward model with scores twice as large halves β's effect. Hence normalization: InstructGPT shifted the
reward model so labeler demonstrations average 0 before RL; others whiten per batch or use group
z-scores (GRPO). S2–S3.

### D47 · ★ [S1 · Part A] "Chat LLMs also have logits, turned into probabilities by softmax. What's the difference?" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30):** "we need to explain this well in article and
> animation" (plan P11 ★, `X-POST-005` section, `CAND-ANIM-043`).

**Key point:** Bradley–Terry *is* a softmax over two answers: `σ(a − b) = eᵃ / (eᵃ + eᵇ) = softmax([a, b])₀`.
A reward-model score plays the role of a logit; what differs is the set of candidates and the use.

**Example.** Chat model after "The capital of France is": logits Paris 2.0, Lyon 1.0, cat −1.0 →
softmax 0.705 / 0.259 / 0.035 (over these three). Reward model: score(A) = 2.0, score(B) = 1.0 from two
separate passes → softmax([2, 1]) = [0.731, 0.269] = σ(1).

| | Chat-model logits | Reward-model scores |
|---|---|---|
| per pass | 151,936 per position | 1 per sequence |
| competitors | same output vector | separate forward passes |
| softmax over | whole vocabulary (complete set) | only the compared answers (2); all possible answers is impossible |
| absolute meaning | real probabilities, sum to 1 | none for one score; only gaps |
| loss | cross-entropy, target = actual next token | 2-way cross-entropy, target = chosen answer |
| shift invariance | per position | per prompt |
| scale | temperature | learned (D46) |
| use | softmax → sample → generate | raw score is the reward; σ only in the loss |

**Generalization:** K ranked answers → Plackett–Luce (softmax over K, then K − 1, …); InstructGPT ranked
4–9 answers and trained on all pairs within each ranking. **Link to S2:** DPO's implicit reward is built
from the chat model's own log-softmax, `β[log π(y|x) − log π_ref(y|x)]`, so no separate reward model.

### D48 · ★ [S1 · Part A] "Explain again, plainly: how are logits transformed in the network? logits → probabilities → detokenizer? logits → score?" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30):** "the above is important, mark it in note and
> should be a key point in the article for reward model." A **key point of `X-POST-005`** (plan,
> "Key points" and section 4b) and of animations `CAND-ANIM-042/043`.

**Learner's view:** the key understanding is how the numbers flow through the network. Corrections:
chat model: logits → probabilities → **pick one token ID** → the detokenizer turns that ID into text (it
never sees probabilities). Reward model: there is **one** logit, and it *is* the score; nothing follows it
at use time.

**Chat model, step by step** (real Qwen3 IDs; toy 4-dim vectors instead of 1024):
1. Tokenize "The capital of France is" → `[785, 6722, 315, 9625, 374]`.
2. Embed: each ID → 1024 numbers. 3. 28 layers → `h` (1024 numbers) at the last position.
4. LM head: 151,936 rows × 1024; `logit_k = row_k · h`. Toy `h = [0.5, −1.0, 2.0, 0.0]`: Paris (12095) row
   `[2, 0, 0.5, 1]` → 2.0; Lyon (55201) `[0, −0.5, 0.25, 3]` → 1.0; cat (8251) `[0, 1, 0, 2]` → −1.0.
5. Softmax → 0.705 / 0.259 / 0.035. 6. Pick an ID (greedy or sampled) → 12095.
7. Detokenize → " Paris". 8. Append and repeat: one forward pass per generated token.

**Reward model:** steps 1–3 identical on prompt + answer + `<|im_end|>`, `h` read at `<|im_end|>`; the score
head is **one row** `w`: `score = w · h`. Toy `w = [1, −0.5, 0.5, 2]`: answer A (`h_A = [0.5, −1, 2, 0]`) →
2.0; answer B (`h_B = [1, 0, 0, 0]`) → 1.0. At use time, 2.0 goes straight to RL as the reward: no
softmax, no ID, no detokenizer. Only in training: σ(2.0 − 1.0) = 0.731, loss −log 0.731 = 0.31.

**Key point:** every logit is one dot product of the final hidden vector `h` with one row of the head.
Many rows (one per token) → logits rank next tokens; one row → one logit ranks whole answers. Everything
before the head is the same. (Tokenizer: 151,669 tokens; the head has 151,936 rows, the rest padding.)
Belongs with D47 ★ in `X-POST-005` (key point; section 4b) and `CAND-ANIM-042/043`.

### D49 · [S1 · Part B] "Bradley–Terry is the true model here: why call it the true model? What does true model mean?" (2026-10-30)

**Two meanings of "model".** In "reward model" it is a neural network. In "Bradley–Terry is the true
model" it is a **probability model of how the labels are produced**: a rule that says, for a pair (A, B),
with what probability the label is "A wins". The **true model** is the rule that actually produced the
data (statisticians call it the data-generating process).

**Why "true" in Part B.** The labels are simulated, and the simulator's code *is* Bradley–Terry: it
computes `r*(f) = w*·f` with `w* = (2.0, 0.5)` and draws "A wins" with probability `σ(r*_A − r*_B)`.
Example: `f_A = (1, 0)`, `f_B = (0, 1)` → `r*_A = 2.0`, `r*_B = 0.5`, gap 1.5, `σ(1.5) = 0.82`; the simulator
says "A wins" on 82 of 100 draws, "B wins" on 18. Fitting Bradley–Terry to these labels uses the same
formula that made them, so the fitted model's form matches the truth (**well-specified**), and a linear
`r_w(f) = w·f` can equal `r*` exactly.

**What that buys.** (1) The fit can be checked: fitted `w` ≈ `(2.0, 0.5)` on 4,000 clean pairs.
(2) Calibration: predicted probabilities match label frequencies (ECE < 0.01). (3) Even at `w*` the loss
is above 0: the labels are random draws (18% of the example pairs say B), so no `w` predicts every label.

**Real human data.** Nobody computes `σ(gap)` before clicking. Bradley–Terry is an *assumption* there,
and it fails in known ways: annotators disagree, preferences can be intransitive (A > B, B > C, C > A,
which no set of scores can produce), ties, and the probability can depend on more than one score
difference. Training then finds the **best Bradley–Terry fit** to the labels, not "the truth", and
there is no `w*` to compare with. Part B uses a simulation for exactly this reason: on real data,
recovery cannot be checked.

### D50 · ★ [S1 · Part B] "So the reward model learns w from noisy labels?" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30).** Random label noise lowers the reward model's
> confidence (smaller `w`, same direction, ranking intact); systematic bias changes what is rewarded
> (direction of `w` shifts toward the bias), and more data does not remove it.

**Yes.** Training sees only `(f_chosen, f_rejected)` per pair, never `w*` or the true gap, and minimizes
`−log σ(w · (f_c − f_r))` by gradient descent. The labels are noisy at two levels:

1. **Even clean labels are random.** Each is one draw of the Bradley–Terry coin (gap 1.5 → A wins 82%,
   B 18%). One label says little; 4,000 labels on different pairs pin down one shared `w`
   (fit ≈ (2.0, 0.5), ECE < 0.01). Loss starts at log 2 = 0.693 and stays above 0.
2. **Corrupted labels (ε = fraction of pairs a bad judge affects):**
   - **Random flips** (ε = 0.2): fitted `w` ≈ (0.72, 0.18). Ratio stays 0.25, ranking agreement with
     `r*` 99.9%: the flips point in no consistent direction and cancel in the gradient. The scale shrinks
     (labels never exceed 1 − ε = 0.8), so the model is **under-confident** against clean labels
     (ECE ≈ 0.15; ≈ 0.04 against the noisy labels it trained on).
   - **Length-biased judge** (ε = 0.2, only ≈ 8% of labels changed): ratio 0.25 → ≈ 0.53, ranking
     agreement ≈ 92%. The bias points in one direction every time, so it adds up in the gradient and is
     learned as part of the reward; more data does not remove it.

Rule (D3): random noise lowers confidence; systematic bias changes what is rewarded. A real reward model
is the same procedure with `w` = all network weights and human labels as the noisy judge.

### D51 · ★ [S1 · Part B] "Is Bradley–Terry the only true model for reward models in the LLM context, in practice?" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30).** Bradley–Terry is the most common *assumed* model,
> not the true one and not the only one. The table below was written from memory: verify each row
> against its paper before it goes into an article or `source-map.md`.

**Correction of the wording.** On real data no one knows the true model (D49). Bradley–Terry is the most
common **assumed** model: InstructGPT, Llama 2 and Anthropic's reward models all train with
`−log σ(r_c − r_r)` or a close variant. It is not the only one. Models used in practice:

| Model | Label it needs | Training target | Used in / example |
|---|---|---|---|
| Bradley–Terry | pair, A or B | `−log σ(r_c − r_r)` | InstructGPT (rankings of K = 4–9 answers, trained as all K·(K−1)/2 pairs), most RLHF reward models |
| BT with a margin | pair + strength ("much better" / "slightly better") | `−log σ(r_c − r_r − m)`, larger `m` for stronger preference | Llama 2 |
| BT with ties | pair, A / B / tie | adds a tie probability (Rao–Kupper, Davidson) | some preference datasets with a tie option |
| Plackett–Luce | full ranking of K answers | product of softmaxes: first place over all K, second over the rest, … | ranking data; reduces to BT at K = 2 |
| Pointwise regression | absolute rating per answer (e.g. 0–4 per attribute) | mean squared error to the rating | HelpSteer2 / NVIDIA Nemotron reward models; HelpSteer2-Preference compared and combined BT and regression |
| Multi-objective | several ratings per answer | one head per attribute, combined by learned weights | ArmoRM |
| General pairwise preference | pair | predict P(A > B) directly from both answers in one input, no single score per answer | PairRM; Nash learning from human feedback; can represent A > B > C > A, which BT cannot |
| Generative judge / verifier | pair or rating, often with written reasoning | the LM's probability of a token such as "Yes" or "A" | LLM-as-judge, GenRM; keeps the LM head |
| Per-step classification | label per reasoning step | binary cross-entropy per step | process reward models (PRM800K) |
| No learned model | checkable answer | rule or program output | RLVR verifiers (S5) |

DPO also assumes Bradley–Terry inside its derivation (S2); IPO was proposed partly because that
assumption makes DPO over-fit when preferences are deterministic.

### D52 · ★ [S1 · Part B] "Briefly remind me of gradient descent and its connection to training a reward model" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30):** in the article, right after the Bradley–Terry
> loss, use gradient descent to minimize it. **Key point 3 of `X-POST-005`** (plan section 3b; section 5
> scales it to the full network).

**Gradient descent:** to make a loss `L(w)` smaller, repeat `w ← w − lr · ∇L(w)`. The gradient `∇L` says,
for each weight, how much the loss rises if that weight rises a little; stepping against it lowers the
loss. `lr` (learning rate) sets the step size.

**For a reward model** the loss is Bradley–Terry's, averaged over labeled pairs. Linear case (Part B,
Exercise 3), with `Δf = f_chosen − f_rejected`:
`L = −log σ(w · Δf)`, `∇L = −(1 − σ(w · Δf)) · Δf`.

**Example:** `w = (0, 0)`, `Δf = (1, −1)`, `lr = 1`.
- Step 1: `w · Δf = 0`, σ = 0.5, loss 0.693, gradient `−0.5 · (1, −1) = (−0.5, 0.5)` → `w = (0.5, −0.5)`.
- Step 2: `w · Δf = 1.0`, σ = 0.731, loss 0.313, gradient factor 0.269 (smaller push: this pair is
  already mostly right).
Each step raises the chosen answer's score relative to the rejected one; the push `1 − σ` shrinks as
the model gets the pair right (Part A).

**From Part B to a real reward model:** same loss, same rule, three changes. `w` becomes all the
network's weights (≈ 600M for Qwen3-0.6B); backpropagation computes `∇L` through the 28 layers and the
score head; each step uses a minibatch of pairs (stochastic gradient descent), and AdamW replaces the
plain update (per-weight step sizes from running gradient averages, plus weight decay). Still
supervised learning, no RL (D42).

### D53 · [S1 · Part B] "Explain random flips and calibration in easy language with examples: what and why" (2026-10-30)

**Random flip:** on a fraction ε of pairs, the label is reversed, whatever the two answers contain
(misclicks, inattention). Why simulate it: it is the simplest kind of label error, and we can compare the
fit with the known truth.

**Counts, one pair, 100 judges, true gap 2.2 (σ = 0.9).** Clean: 90 say A, 10 say B. With ε = 0.2: of the
90 A votes 18 flip to B, of the 10 B votes 2 flip to A → 72 + 2 = **74 say A**
(`ε + (1 − 2ε)·σ = 0.2 + 0.6·0.9 = 0.74`). However large the gap, at most 80 say A (ceiling) and at least
20 (floor). A still wins more often than B (74 > 50), so the order is kept; the margin shrinks from
90/10 to 74/26. Flips hit every pair the same way and favour no feature, so training keeps the
direction of `w` (ranking 99.9%) and shrinks its size (`w` ≈ (0.72, 0.18)).

**Calibration:** a model is calibrated when its stated probabilities match how often things happen.
Check: take the 100 pairs where the model said "A wins with 0.7" and count A wins. 70 → calibrated;
90 → under-confident (it said 0.7, reality is 0.9); 50 → over-confident. ECE averages the size of
these gaps over all prediction groups, weighted by how many pairs each group has.

**The flipped-label model, checked two ways.** Against the noisy labels it trained on: it says ≈ 0.7 and
noisy A wins ≈ 0.7 → nearly calibrated (ECE ≈ 0.04; not 0, because σ has no floor and ceiling).
Against clean preferences: it says 0.7, clean A wins 0.9 → under-confident (ECE ≈ 0.15).

**Why it matters.** Ranking is correct, so picking the better answer still works. The probabilities and
score gaps are too small, which matters when the score is read as a confidence (filtering pairs,
comparing reward models) and in RL, where the size of reward differences is weighed against the KL
penalty β. Calibration measures whether the numbers mean what they say, separately from whether the
order is right.

### D54 · [S1 · Part B] "Still don't fully understand. Use metaphors and even easier examples" (2026-10-30)

The learner asked for metaphors here (an exception to the default plain technical style).

**Random flips = a scorekeeper who sometimes presses the wrong button.** Team A truly beats team B in
9 of 10 games. The scorekeeper records the wrong winner in 2 of every 10 games, for either team, at
random. Of A's 9 wins about 1.8 get recorded for B; of B's 1 win about 0.2 gets recorded for A.
Scoreboard: A ≈ 7, B ≈ 3. A is still ahead (order kept), but the lead looks smaller than it is.
Mapping: games = comparisons of two answers; scorekeeper = the labelers; scoreboard = the training labels.

**Contrast, systematic bias = a scorekeeper who always gives the mistakes to the taller team.** If B is
taller, B gains every mistaken point; with enough mistakes B can look like the better team. The order
itself can change. (Part B's length-biased judge: the "taller team" is the longer answer.)

**What the reward model learns.** It only sees the scoreboard and learns to predict it: "A wins 70% of
the time". To say 70% instead of 90% it needs a smaller score gap, so `w` shrinks
((2.0, 0.5) → (0.72, 0.18)) while still pointing the same way.

**Calibration = a weather forecaster who is right about how often.** Collect every day she said "70%
chance of rain". If it rained on 7 of 10 such days, she is calibrated. If it rained on 9 of 10, she is
too timid (under-confident). If 5 of 10, she is too bold (over-confident).

**Our reward model is the timid forecaster.** It learned from the faulty scoreboard, so it says "A
better, 70%". Checked against the scoreboard it learned from: A wins about 7 of 10 → it looks right.
Checked against the real games: A wins 9 of 10 → it is too timid. It still picks the right winner;
it is only too unsure. Why care: when the RL step or a person reads the score as "how much better",
the reward model understates real differences.

### D55 · ★ [S1 · Part C] "Part C mainly reflects our discussion of chat LLMs vs reward models, and the logits" (2026-10-30)

> ★ **Marked important by the learner (2026-10-30).** In `X-POST-005` as section 4a, "Which position
> gives the score".

**Mostly right.** The load report (`lm_head.weight` 151,936 × 1024 UNEXPECTED, dropped; `score.weight`
1 × 1024 MISSING, random) is the head swap of D30/D42, and `score = w · h` is the "one head row" of
D47 ★/D48 ★. **Four things Part C adds:**

1. **The score head runs at every position.** `w · h_t` gives a `[B, T]` tensor: one number per token, like
   the chat model gives 151,936 logits per token. The reward is **one** of these, at the last real token
   (`<|im_end|>`); the rest are discarded (D28). D48's "h at `<|im_end|>`" is that selection.
2. **Finding that position under padding (Exercise 6).** Batches pad short rows with `<|endoftext|>`
   (151643). `(arange(T) · non_pad).argmax` gives the rightmost real token for right and left padding;
   `non_pad.sum − 1` works only for right padding.
3. **Why the last position (Predict 2).** Causal attention: `h_t` depends only on tokens 1…t. Two answers
   differing in one middle word have identical outputs before that word and different outputs from it to
   the end. Only the last position has seen the whole prompt and answer; position 0 would give the same
   number for every answer.
4. **bf16 rounding noise.** Right vs left padding: equal within 1e-5 in float32, ≈ 0.008 apart under bf16
   autocast, a floor on how precisely one score can be read.

