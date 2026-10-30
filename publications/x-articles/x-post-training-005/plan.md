# Article plan — X-POST-005 Reward models

Status: **scope set by the learner (2026-10-30)**; outline below for learner review. No drafting or
rendering yet.

## Scope

**This article is solely about reward models** (learner, 2026-10-30). It explains what a reward
model is, what data it learns from, how Bradley–Terry turns comparisons into a loss, what changes in
the network, how it is trained, and what it learns besides quality. PPO, GRPO and DPO are **not**
explained here: a reward model's use as a grader is stated in one or two sentences at the end, with a
pointer to a later article. (The earlier plan, "preference alignment: reward models, PPO, DPO", is
replaced; the policy-optimization article still needs a slot, for the learner to decide.)

## Source: read the learning notes first

**Before outlining or drafting, read the learning note**
[`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`](../../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md)
in full. It records the learner's own questions, misunderstandings and the explanations that worked;
the article should answer those questions in that order of difficulty. The relevant parts:

| Topic | Learning-note entries |
|---|---|
| What a preference is; can preferences be bad; label noise and the agreement ceiling; ★ random noise vs systematic bias (what each does to the learned reward) | D1, D2, D3, D50 ★ |
| Bradley–Terry: from "A beats B" to a probability; examples; the derivation; cross-entropy link; predicted vs ground truth; plain explanation with votes → scores | D4, D5, D6, D7, D9, D10, D41 |
| Is the reward learned; grader vs generator; does the reward model train the LLM | D8, D11, D12, D13 |
| What a reward model is; is it the LLM itself; choosing its base; same-family graders | D15, D16, D17, D18 |
| Reward-model data vs RL data | D19, D20 |
| How it turns text into one number (`rm_scores`, last-token pooling); the training loop; is there a prompt to it | D27, D28, D29, D30 |
| Head swap (chat-model head vs score head); which algorithm optimizes the loss (backprop + AdamW, no RL) | D42; plan P10 |
| ★ Chat logits vs reward scores (BT is a 2-way softmax); score range, scale and logit meaning; why the output is a number before training; scores vs rubrics | D47 ★, D46, D45, D44, D43; plan P11 ★ |
| ★ Gradient descent on the Bradley–Terry loss (`w ← w − lr · ∇L`, push `1 − σ`; linear example → full network with backprop + AdamW) | D52 ★ |
| ★ Which position gives the score: a score at every position, read at the last real token; causal attention (one-word-change example); padding; bf16 noise | D55 ★, D28 |
| ★ The numbers through the network, step by step (every logit = `h · one head row`; chat: logits → probabilities → token ID → detokenizer; reward model: one logit = the score) | D48 ★ |
| Key concepts ★ rows: `rm_scores`, reward-model training process, head swap | **Key concepts** table |
| Reward hacking as the motivation for caution | plan P2; D3 |
| ★ Bradley–Terry is an assumption, not the only choice: margin, ties, Plackett–Luce, regression on ratings, multi-objective, general pairwise preference, generative judges, process reward models, verifiers (from memory: verify each against its paper) | D51 ★ |

Also: the study-path S1 notebook (`notebooks/study-path/s1_reward_models.ipynb`), Day 15 · 04 Part C,
Chapter 10 (`book/chapters/10-*.md`) and SLP3 §8.3. Every claim maps to `source-map.md`.

## Key points (the reader must leave with these)

1. ★ **Every logit is one dot product, `h · one head row`** (learner, 2026-10-30: "should be a key
   point in the article for reward model"; D48 ★). The text goes through the same steps in both models:
   tokenize → embed → layers → final hidden vector `h`. Only the head differs. The chat model's head has
   151,936 rows, so it produces 151,936 logits → softmax → probabilities → pick one token ID → the
   detokenizer turns the ID into text → repeat. The reward model's head has **one** row, so it produces
   **one** logit, and that logit is the score: it goes to RL as the reward, with no softmax, no token
   and no detokenizer. A softmax (σ) over two answers' scores appears only in the training loss.
   Use the toy 4-dim example of D48 (Paris/Lyon/cat 2.0/1.0/−1.0; answers A/B 2.0/1.0).
2. ★ **Bradley–Terry is a softmax over the two compared answers** (D47 ★): same loss family as
   next-token training, `−log softmax(correct option)`.

3. ★ **The Bradley–Terry loss is minimized by gradient descent** (learner, 2026-10-30: "after explaining
   the loss of Bradley–Terry, then naturally use gradient descent to minimize the loss"; D52 ★). Right
   after the loss `−log σ(r_c − r_r)` is introduced, show how it is made smaller: `w ← w − lr · ∇L`, with
   the push `1 − σ` that raises the chosen score and lowers the rejected one, and shrinks once the pair
   is right. Worked example of D52: `w = (0, 0)`, `Δf = (1, −1)`, `lr = 1` → loss 0.693 → 0.313, push
   0.5 → 0.269.

Points 1 and 2 are developed in sections 4 and 4b, point 3 in section 3b; all three must also appear in
the summary list.

## Outline (draft, for learner review)

1. **Opening example.** Two answers to one prompt, a person picks one. The question: how does that
   single choice become a number a model can be trained on?
2. **Why comparisons, not grades.** People agree more on "which is better" than on "how good"; real
   labels are noisy (HH-RLHF agreement), which caps reward-model accuracy. (D1–D3)
3. **Bradley–Terry in plain words.** Scores, the gap, the S-curve, votes → fitted gap
   (7 of 10 → 0.85), only the gap matters. Animation `CAND-ANIM-041`; optionally `CAND-ANIM-033` (why a sigmoid). (D4–D10, D41)
3b. ★ **Minimizing the loss: gradient descent** (key point 3; follows directly from the loss in
   section 3). What the gradient is (for each weight, how much the loss rises if the weight rises a
   little); the update `w ← w − lr · ∇L`; for Bradley–Terry the gradient is `−(1 − σ(gap)) · Δf`, so each
   step raises the chosen answer's score relative to the rejected one, by less once the pair is right.
   Two-step worked example (D52 table). Keep it to a two-weight linear scorer here; section 5 scales the
   same rule to the full network. Animations `CAND-ANIM-034` (gap → probability → loss → gradient) and
   `CAND-ANIM-035` (the gradient goes where the model is still wrong). (D52 ★; S1 Part A gradient, Part B Exercise 3)
4. **Same body, different head.** The chat model's LM head `1024 → 151,936` is replaced by a
   `1024 → 1` score head; one forward pass, the number at the last token, no generation; the pick is a
   comparison of two scores. Animation `CAND-ANIM-042`. (D15, D16, D28, D30, D42)
4a. ★ **Which position gives the score** (learner: "important", 2026-10-30; D55 ★). The score head runs
   at every position, `w · h_t`, one number per token, just as the chat head gives 151,936 logits per
   token; the reward is the number at the last real token (`<|im_end|>`), the rest are discarded. Why
   the last: causal attention, `h_t` depends only on tokens 1…t. Example from S1 Predict 2: two answers
   differing in one middle word ("two" vs "six inches deep") have identical outputs before that word and
   different outputs from it to the end, so only the last position has seen the whole prompt and answer.
   Brief practical notes, one sentence each: in a batch, shorter rows are padded, so the code must find
   the last *real* token (rightmost non-pad, correct for left and right padding); bf16 arithmetic moves a
   single score by about 0.008, so tiny score differences are rounding noise. (D28, D55 ★)
4b. ★ **Logits and softmax: chat model vs reward model** (learner: "important, explain this well",
   2026-10-30; **key point 1 and 2**). Bradley–Terry is a softmax over two answers, `σ(a − b) = softmax([a, b])₀`. Chat model:
   one pass → 151,936 logits → softmax over the whole vocabulary → a real probability, sampled to
   generate. Reward model: one pass per answer → one score each → softmax over just the compared answers,
   only inside the loss; the raw score is the reward, and one score alone means nothing. Same loss
   family, `−log softmax(correct option)`: the target is the next token in one case, the human's choice
   in the other. Worked example: Paris/Lyon/cat logits 2, 1, −1 → 0.705/0.259/0.035; answers scored 2
   and 1 → 0.731/0.269. Mention Plackett–Luce for rankings in one sentence. Animation `CAND-ANIM-043`.
   Walk the numbers through the network step by step (tokenize → embed → layers → h → head rows → logits),
   with the toy 4-dim example of D48. (D46, D47 ★, D48)
5. **Training.** The same gradient descent as section 3b, scaled up: `w` becomes all the network's
   weights (≈ 600M for Qwen3-0.6B), backpropagation computes the gradient through the layers and the
   score head, each step uses a minibatch of pairs, and AdamW replaces the plain update. Ordinary
   supervised training, no RL; one epoch; accuracy well below 100% is normal on human data.
   (D27, D29, D42, D52 ★)
6. **What it learns besides quality.** Length and style correlate with the label; a reward model is a
   proxy; optimizing hard against it finds its errors (reward hacking, preview only). (D15; P2)
7. **Which model, who trains it.** Base choice, same family, reuse across policies. (D17, D18)
8. **Where it is used (short).** As the grader in RLHF, as a labeler for new preference pairs, for
   best-of-n selection: one paragraph, with the pointer to the next article. Summary list (must include key points 1, 2 and 3).

## Animations

Every animation for reward models, as of 2026-10-30 (storyboards in
`visuals/animations/PROPOSALS.md`; production on the Mac Studio). 044–051 were proposed at the
learner's request ("suggest more animations, so it can support the whole article") so every section
has one. The series target is about 5–6 GIFs; **Core** marks a recommended set of 7 (one over target,
for the learner to cut), the others are optional extras or course/notebook use. The learner chooses.

| Section | Candidate | Shows | Source | Core |
|---|---|---|---|---|
| 1 | `CAND-ANIM-044` | one click becomes a `(prompt, chosen, rejected)` row; thousands of rows = the training set | D1; Day 15 · 04 Part A | optional (hook) |
| 2 | `CAND-ANIM-045` | grades scatter, choices agree more; agreement caps accuracy | D1–D3 | optional |
| 3 | `CAND-ANIM-041` | Bradley–Terry for beginners: one pair, two scores, gap → probability → loss → push | P9, D41 | **core** |
| 3 | `CAND-ANIM-033` | why a score gap becomes a sigmoid (strength `e^score`, A's share = `σ(gap)`) | P3, D4–D7 | optional |
| 3b | `CAND-ANIM-034` | four pairs through gap → probability → loss → gradient; +10 shift changes nothing | P4 | optional |
| 3b | `CAND-ANIM-035` | gradient steps: the push `1 − σ` concentrates on wrong or uncertain pairs | P4, D3, D52 ★ | **core** |
| 4 | `CAND-ANIM-042` | same body, different head: `lm_head` 1024 → 151,936 swapped for `score` 1024 → 1 | P10, D42 | **core** |
| 4a | `CAND-ANIM-048` ★ | a number at every token; a one-word change alters only later positions; the last is the score | D55 ★ | **core** |
| 4b | `CAND-ANIM-043` ★ | logits and softmax: next tokens (one pass, 151,936-way) vs compared answers (two passes, 2-way) | P11 ★, D47 ★, D48 ★ | **core** |
| 5 | `CAND-ANIM-049` | the same gradient step on 596 M weights; accuracy 47.6% → 62.2% | D42, D52 ★; Day 15 · 04 | optional |
| 2 or 6 | `CAND-ANIM-046` ★ | random noise shrinks `w` (order kept); systematic bias rotates it (order changes) | D50 ★, D54 | **core** |
| 2 or 6 | `CAND-ANIM-047` | calibration: does 70% mean 70%? under-confident against the truth | D53, D54 | optional |
| 6 | `CAND-ANIM-050` | length correlates with the score (+0.25); filler probe; proxy vs true quality (illustrative) | D15, P2; Day 15 · 04 Part C | **core** |
| 8 | `CAND-ANIM-051` | one frozen reward model, three uses: best-of-n, labeler, RLHF grader | D11–D13 | optional (close) |

Possible merges if the learner wants fewer GIFs: 041 + 035 (the push step at the end of 041), 042 + 048
(the head applied at every position, then the last kept), 046 + 047 (calibration as 046's last frame).
Not for this article: `CAND-ANIM-036` (PPO), `CAND-ANIM-037` (KL to the reference), which belong to the
policy-optimization article; `CAND-ANIM-038` is X-POST-004's set. Section 7 (which base model) needs no
animation: a short table.

## Conventions

Follow `../README.md` (post-training series conventions): English at the package root and an
independently written Chinese version in `zh/`; core technical terms in English (reward, reward
model, policy, …); animated GIFs by default, about 5–6 per article; series cover style; no links in
the public body.
