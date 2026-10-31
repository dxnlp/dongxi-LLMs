# Article plan — X-POST-005 Reward models

Status: **EN and ZH drafts written (2026-10-30)**, awaiting learner review; paper facts checked
(`source-map.md`). All decisions are set (below). The GIFs are rendered (eleven after the 04 and 06 splits)
(`visuals/animations/projects/post-training-reward-models/`), awaiting learner review.

## Purpose and audience

The article introduces **reward models** to readers who use or follow LLMs. It explains:
- what a reward model is and what data it learns from;
- how Bradley–Terry turns a comparison into a loss, and how gradient descent lowers that loss;
- what changes in the network, and which position gives the score;
- what the model learns besides quality;
- where it is used.

It presents the method. It does **not** show our own work (learner, 2026-10-30): no notebook results,
accuracies, correlations, fitted weights or run settings, and no "we trained". The numbers in the body
are of three kinds, each labelled:
- **public facts:** papers, model configs;
- **exact arithmetic** on small illustrative numbers;
- **illustrations:** labelled as such.

**Scope:** solely reward models. PPO, GRPO and DPO are not explained. The reward model's use as a
grader gets one paragraph at the end, with a pointer to the next article.

## Decisions (learner, 2026-10-30)

| # | Decision |
|---|---|
| Cover | **upright** variant: `assets/cover-post-training-05.png` |
| Outline | the **11-section** outline below; section 10 (beyond Bradley–Terry, which base model) removed by the learner after the draft review, 2026-10-30 |
| Experiments | **no details of our experiments**; the article is an introduction to reward models, not a report |
| Example | X-POST-004's ***Is OpenAI open?*** runs through the whole article |
| GIFs | the recommended **9**: 044, 041, 035, 042, 048, 043, 046, 050, 051; after review, 042 and 043 were split into chat-model / reward-model pairs (**11 GIFs**) |
| Title | *Reward Models: How a Choice Between Two Answers Becomes a Score* / *Reward model：人的选择怎样变成一个分数* |
| Brand | every GIF and MP4 carries DONGXI LABS (`visuals/animations/brand.py`) |

## Key points (the reader must leave with these)

1. ★ **Every logit is one dot product, `h` · one head row** (D48 ★). Both models take the same steps:
   tokenize → embed → layers → final hidden vector `h`. Only the head differs.
   - The chat model's head has 151,936 rows: 151,936 logits → softmax → pick one token ID →
     detokenize → repeat.
   - The reward model's head has **one** row: one logit, and that logit is the score. It goes to RL
     as the reward, with no softmax, no token and no detokenizer.
   - A softmax over two answers' scores appears only in the training loss.
2. ★ **Bradley–Terry is a softmax over the two compared answers** (D47 ★): `σ(a − b) = softmax([a, b])₀`.
   It is the same loss family as next-token training, `−log softmax(correct option)`.
3. ★ **The Bradley–Terry loss is minimized by gradient descent** (D52 ★). The update is
   `w ← w − lr · ∇L`. Its push `1 − σ` raises the chosen score, lowers the rejected one, and shrinks
   once the pair is ranked right.

Key point 1 is developed in section 5, key point 2 in section 7, and key point 3 in section 4. All
three return in the summary.

## Running example

The prompt is X-POST-004's: *Is OpenAI open?* Continuity:
- X-POST-004 named the reward model as its judge and called it "the topic of the next article".
- Its judge scored *OpenAI is open. It's in the name.* 2, and the *partly* answers 8 and 10.
- This article opens by asking where such scores come from.

Facts as in 004: OpenAI has released open-weight models (gpt-oss-120b and gpt-oss-20b, Apache 2.0),
and its flagship models are available only through its API and apps.

| Role | Text or numbers | Source / status |
|---|---|---|
| prompt | *Is OpenAI open?* | 004, X-TOKEN-001 |
| answer A, chosen | *Partly. Some weights are open.* | X-TOKEN-001's reply; accurate |
| answer B, rejected | *Yes. OpenAI is open. It's in the name.* | 004's naive answer |
| one-token variant of A (section 6) | *Partly. All weights are open.* | ` Some` (4329) → ` All` (2009), at the same position; both answers 8 tokens (Qwen3 tokenizer, checked 2026-10-30) |
| votes (section 3) | 10 people compare A and B; 7 pick A → fitted gap `log(7/3) = 0.85` | illustration, exact arithmetic |
| two-feature scorer (section 4) | features: states the facts, argues from the name; A = (1, 0), B = (0, 1), `Δf = (1, −1)`; `w = (0, 0)`, `lr = 1` | illustration (D52 recast) |
| gradient steps (section 4) | step 1: both scores 0, loss 0.693, push 0.5 → `w = (0.5, −0.5)`; step 2: gap 1.0, σ 0.731, loss 0.313, push 0.269 | exact arithmetic |
| toy hidden vectors (sections 5, 7) | `h = [0.5, −1, 2, 0]`; chat-head rows `Part` (5800) `[2, 0, 0.5, 1]` → 2.0, `Yes` (9454) `[0, −0.5, 0.25, 3]` → 1.0, `No` (2753) `[0, 1, 0, 2]` → −1.0; score row `w = [1, −0.5, 0.5, 2]`: `h_A = [0.5, −1, 2, 0]` → 2.0, `h_B = [1, 0, 0, 0]` → 1.0 | illustration (D48 recast; 4 numbers stand for 1,024) |
| softmaxes (section 7) | chat: `Part` / `Yes` / `No` 2, 1, −1 → 0.705 / 0.259 / 0.035 (over these three); reward: A 2.0, B 1.0 → 0.731 / 0.269 = σ(1) | exact arithmetic |
| model facts | Qwen3-0.6B: hidden size 1,024; 28 layers; head 151,936 rows (tokenizer 151,669 entries); 596,049,920 parameters | public config, as in X-TOKEN-001 |

## Outline (11 sections, accepted)

About 2,500–3,000 words. One quote block per key idea, about 5 in all.

1. **One choice, one row.**
   - Hook from X-POST-004: a judge gave *OpenAI is open. It's in the name.* a 2. Who decides such
     numbers?
   - Nobody writes them: a person reads A and B and picks A.
   - That click is one row, `(prompt, chosen, rejected)`. Thousands of rows are the training set.
   - The question: how does a choice become a number a model can be trained on? (D1, D8)
2. **Compare, do not grade.**
   - People agree more on which answer is better than on how good one is.
   - Picking needs an opinion about the result, not the skill to write it (SLP3).
   - Labels are still noisy. Published annotator agreement is about 73% for InstructGPT (to
     verify), and it caps a reward model's accuracy on held-out human labels.
   - Forms of preference data: a pair, best of N, a ranking. (D1, D3)
3. **Bradley–Terry: a score gap becomes a probability.**
   - Each answer gets a score; `P(A ≻ B) = σ(s_A − s_B)`.
   - Gap → probability: −2 → 0.12, 0 → 0.50, 1 → 0.73, 3 → 0.95. Never exactly 0 or 1.
   - Only the gap matters: +10 to both changes nothing, so one score alone means nothing.
   - Votes → fitted gap: 7 of 10 → 0.85. Chess Elo is the same idea.
   - The loss is `−log σ(r_A − r_B)`, cross-entropy on "A won".
   - Assumptions, briefly: one scale, no cycles, no ties. (D4–D7, D9, D10, D41)
4. **Lowering the loss: gradient descent.** (key point 3)
   - The gradient: for each weight, how much the loss rises if the weight rises a little.
   - `w ← w − lr · ∇L`. For Bradley–Terry, `∇L = −(1 − σ(gap)) · Δf`.
   - The two-step example on the two-feature scorer.
   - The push is large on pairs ranked wrong and small on pairs already right.
   - Section 8 scales the same rule to the full network. (D52, D43)
5. **Same body, one head row.** (key point 1)
   - A reward model starts from a pretrained or SFT LLM. The LM head (1,024 → 151,936) is removed and a
     score head (1,024 → 1) added.
   - No instruction prompt; it cannot generate.
   - Walk the numbers: tokenize → embed → 28 layers → `h` → head.
     - Chat side: `h` · each row gives `Part` 2.0, `Yes` 1.0, `No` −1.0, and the loop continues.
     - Reward side: `h` · the one row gives A 2.0, then the same for B gives 1.0.
   - Why it outputs a number before any training (architecture), and why that number is meaningless
     until trained. (D15, D16, D30, D42, D45, D48)
6. **Which position gives the score.**
   - The score head runs at every position, one number per token, as the chat head gives 151,936 per
     token. The reward is the number at the last real token, `<|im_end|>`.
   - Why the last: causal attention. In *Partly. Some / All weights are open.*, the outputs are
     identical before ` Some`/` All` and differ from there to the end. Only the last position has
     read the whole prompt and answer.
   - One sentence: batches pad short rows, so the code must find the last *real* token. (D28, D55)
   - bf16 noise is left out (implementation detail).
7. **Logits and softmax: chat model vs reward model.** (key point 2)
   - Bradley–Terry = `softmax([a, b])₀`.
   - Chat model: one pass → 151,936 logits → softmax over the whole vocabulary → real probabilities
     → sample.
   - Reward model: one pass per answer → one score each → softmax over the compared answers, only
     inside the loss. The raw score is the reward.
   - The softmax numbers from the running example. Same loss: the target is the next token in one
     case, the person's choice in the other.
   - Rankings of K answers: Plackett–Luce, one sentence. (D46, D47, D48)
8. **Training the whole network.**
   - The rule of section 4 with `w` = all weights (596 M for Qwen3-0.6B, as a size reference).
   - Backpropagation through the head and every layer; minibatches of pairs; AdamW.
   - Ordinary supervised training, no RL.
   - Usually one epoch (InstructGPT); more overfits noisy labels.
   - Held-out accuracy well below 100% is normal (public figure to verify).
   - Afterwards it is frozen. (D27, D29, D42, D52)
9. **What it learns besides quality.**
   - A reward model learns whatever predicted the labels.
   - **Random noise** (misclicks) mostly cancels: the order is kept and the gaps shrink.
   - **Systematic bias** (judges who favour confident, agreeing or longer answers) adds up and
     becomes part of the reward; more data does not remove it. Example: a judge pool that prefers
     the confident *Yes …* answer.
   - Public evidence: length correlations, sycophancy (to verify).
   - It is a proxy. Optimizing hard against it finds its errors: reward hacking, in preview only
     (Gao et al. 2022; 004's flattering answer). (D2, D3, D15, D50)
10. **Beyond Bradley–Terry, and which base model.**
    - A short table of alternatives: margin, ties, Plackett–Luce, rating regression, multi-objective,
      general pairwise preference, generative judges, process reward models, verifiers. Every row
      is verified (D51).
    - Base choice: same family as the chat model, for shared knowledge (Llama 2); size trade-off
      (InstructGPT 6B for 175B).
    - Refresh rounds as the chat model drifts. (D17, D18, D44)
11. **Where it is used, and summary.**
    - Uses: the grader in RLHF (004's judge), a labeler for new preference pairs, best-of-n selection.
    - Pointer: the next article trains the chat model against it.
    - Summary list with key points 1–3.

## Animations

Storyboards are in `visuals/animations/PROPOSALS.md`; production happens on the Mac Studio.

**Every storyboard is recast to the running example before production:** the egg, Paris and garlic
examples are replaced. GIFs show no numbers from our runs; mechanisms are drawn as labelled
illustrations.

Every GIF and MP4 carries the DONGXI LABS mark at the bottom-right. `visuals/animations/brand.py`
stamps it at encode time; see `STYLE_GUIDE.md` (learner, 2026-10-30).

| Section | Candidate | Shows | Recommendation |
|---|---|---|---|
| 1 | `CAND-ANIM-044` | one click on A or B becomes a `(prompt, chosen, rejected)` row; rows stack into a dataset | **use** (hook) |
| 2 | `CAND-ANIM-045` | grades scatter, choices agree more; agreement caps accuracy | optional |
| 3 | `CAND-ANIM-041` | Bradley–Terry for beginners: two scores, gap → σ → probability → loss | **use** |
| 3 | `CAND-ANIM-033` | why a gap becomes a sigmoid (strength `e^score`, A's share) | optional |
| 4 | `CAND-ANIM-035` | gradient steps: the push `1 − σ` concentrates on wrong or uncertain pairs | **use** |
| 4 | `CAND-ANIM-034` | four pairs through gap → probability → loss → gradient | optional |
| 5 | `CAND-ANIM-042` | same body, different head: `lm_head` 1,024 → 151,936 swapped for `score` 1,024 → 1 | **use** |
| 6 | `CAND-ANIM-048` ★ | a number at every token; ` Some` → ` All` changes only later positions; the last is the score | **use** |
| 7 | `CAND-ANIM-043` ★ | one pass, 151,936-way softmax vs two passes, 2-way softmax | **use** |
| 8 | `CAND-ANIM-049` | the same step on all weights | **drop**: its storyboard rests on our accuracy numbers; section 8 is short prose |
| 9 | `CAND-ANIM-046` ★ | random noise shrinks the gaps, order kept; systematic bias tilts what is rewarded | **use** (illustration, no fitted values) |
| 9 | `CAND-ANIM-047` | calibration: does 70% mean 70%? | optional |
| 9 | `CAND-ANIM-050` | a biased signal correlates with the score; proxy vs true quality | **use** (illustration, no measured correlation) |
| 11 | `CAND-ANIM-051` | one frozen reward model, three uses | **use** (close) |

The recommended set is **9 GIFs**: 044, 041, 035, 042, 048, 043, 046, 050, 051. Section 10 is a static
table. Sections 2 and 8 are prose. Possible merges for fewer GIFs: 041 + 035, 042 + 048.

## Cover

`build_cover.py`, series style:
- deep ink and stipple dust;
- engraved ripple lines;
- white Avenir Next Heavy title;
- gold labels `POST-TRAINING SERIES · 05` and `DONGXI LABS`;
- 2000×800.

**Reward Model** is the title, standing on a water line. Its reflection in the water reads
**Bradley–Terry** (learner's direction, 2026-10-30): violet, upright letters, broken into bands by the
ripples and fading with depth. The reward model is the visible part; Bradley–Terry is the model
underneath that gives its scores meaning.

**Chosen variant: upright** (learner, 2026-10-30). Default output is `assets/cover-post-training-05.png`; `--mirror` renders the
upside-down alternative to `assets/cover-post-training-05-mirror.png` (not used).

## Sources and verification (into `source-map.md` before drafting)

The learning note
[`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`](../../../learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md)
records the questions the article answers, in order of difficulty: D1–D30 and D40–D55, the
key-concepts table and the knowledge gaps.

Its numbers from our notebooks stay out of the article. Only the explanations and the arithmetic
examples carry over, recast to the running example.

Also, as background: Chapter 10 (`book/chapters/10-preferences-and-reward-models.md`).

Verify, and record each fact in `source-map.md`:
- **SLP3 §8.3–8.4:** read in full; the series' anchor text.
- **InstructGPT (Ouyang et al. 2022):**
  - labeler agreement, about 73%;
  - a 6B reward model for the 175B policy, with 175B reward models unstable;
  - rankings of K = 4–9 answers, trained as all pairs;
  - one epoch;
  - the reward model's held-out accuracy.
- **Anthropic HH-RLHF (Bai et al. 2022a):** crowdworker agreement; the instruction given to raters.
- **Llama 2:**
  - margin loss;
  - separate helpfulness and safety reward models, initialized from chat checkpoints;
  - rounds of reward-model refresh;
  - reported reward-model accuracies.
- **Gao et al. 2022:** overoptimization, and its scaling with reward-model size.
- **Length and sycophancy:** a length-correlation study (e.g. Singhal et al. 2023) and a sycophancy
  study (e.g. Sharma et al. 2023), each to be confirmed.
- **Alternatives table (D51 rows):**
  - Bradley–Terry with ties (Rao–Kupper, Davidson);
  - Plackett–Luce;
  - HelpSteer2 and HelpSteer2-Preference;
  - ArmoRM;
  - PairRM;
  - Nash learning from human feedback;
  - generative reward models (GenRM);
  - PRM800K;
  - IPO.
- **OpenAI's openness:** gpt-oss-120b and gpt-oss-20b, Apache 2.0; flagship models closed. Re-check
  at drafting time.
- **Qwen3-0.6B facts:** reuse X-TOKEN-001's source map.

## Conventions

Follow `../README.md`, the post-training series conventions:
- English at the package root and an independently written Chinese version in `zh/`;
- core RL and LLM terms in English in the Chinese text (reward, reward model, policy, …);
- GIFs by default;
- the series cover style;
- no links in the public body;
- few quotation marks, quote blocks and adverbs; no metaphors.
