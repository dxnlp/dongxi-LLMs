# Article plan — X-POST-004 RL terms for LLMs

Status: **plan for learner review** (2026-10-27). No drafting or rendering yet.

## Reader and goal

- **Reader:** someone who knows LLMs (tokens, next-token probabilities, SFT, loss) but has little or
  no reinforcement-learning background.
- **Goal:** after reading, they can take one sampled answer and follow it through every RL term:
  policy → state/action → reward/return → value/Q → advantage → update. They will also know why
  baseline, credit assignment, exploration, clipping and KL exist.
- **Thesis:** every term in this vocabulary answers one question. **A judge gives one score to a
  whole answer. Which tokens should become more likely, and which less?**
- **Role in the series:** this is the vocabulary bridge before `X-POST-005` (preference alignment:
  reward models, PPO, GRPO, DPO) and `X-POST-006` (RLVR). It names the pieces, and those two
  articles then use them without stopping to define them.

## Titles (working)

- EN: **RL Terms for LLMs: How One Score Reaches Every Token**
- ZH: **写给 LLM 读者的强化学习术语：一个分数如何分到每个 token**

## Running example (one example for the whole article)

The learner chose it on 2026-10-27: **Is OpenAI open?** The name says open; the accurate answer
is partly. OpenAI has released open-weight models (gpt-oss-120b and gpt-oss-20b, Apache 2.0),
while its flagship models are available only through its API and apps. The article states this
plainly and as of the time of writing. The humour comes from the name, not from claims about
the company.

At the state "Is OpenAI open? → OpenAI is", the policy's next-token probabilities are:

| Next token | π(a\|s) | Judge score (0–10) | Why |
|---|---|---|---|
| open | 0.5 | 2 | It follows the name ("It's in the name.") |
| closed | 0.3 | 5 | Half right: misses the open-weight releases |
| partly | 0.2 | 9 | Accurate |

The model's favourite token is the naive one. All the numbers follow exactly from this table:

- V(s) = 0.5·2 + 0.3·5 + 0.2·9 = **4.3**. In the figure, 10 sampled continuations (5 open,
  3 closed, 2 partly) average exactly 4.3.
- Q(s, open) = 2, Q(s, closed) = 5, Q(s, partly) = 9. In figure 05 (revised 2026-10-28) answers
  with the same first token still score differently, so Q is an average. The scores are open 3, 1,
  2, 1, 3; closed 5, 4, 6; partly 10, 8. All ten still average to 4.3. The article text should
  likewise call Q the average score after a token, not the score of one answer.
- The advantages and their probability-weighted average:

  | Token | Advantage |
  |---|---|
  | open | **−2.3** |
  | closed | **+0.7** |
  | partly | **+4.7** |
  | weighted average | 0 (−1.15 + 0.21 + 0.94) |

  "closed" gets a *positive* advantage although it is not the best answer, because it beats the
  model's average. This makes the article's point that advantage is relative to the current
  policy.
- Sampled answer: "OpenAI is open. It's in the name." The reward is 0 on every token and 2 at the
  end. With γ = 1, the return is 2 at every position.
- The value line along that answer is 4.3 up to "open", then 2. The TD error is 0 everywhere except
  at "open" (−2.3).
  - λ = 0 puts all the blame on "open".
  - λ = 1 gives −2.3 to "OpenAI", "is" and "open" alike, and 0 to the tokens after "open", because
    V there is already 2. (Corrected 2026-10-28: an earlier line called this the same as GRPO.
    GRPO has no per-token V, so all ten tokens of the answer get −2.3.)
  - With an exact V, λ = 0 is right: every answer starts "OpenAI is", so those two tokens did nothing
    wrong. λ = 1 also blames them because this one sample went badly. That blame is noise and
    averages out over samples. λ = 0 is only as good as V. In practice V is learned, so λ ≈ 0.95.
- Policy gradient, as a first-order direction: the change in each logit is proportional to π·A.
  "partly" rises, "open" falls, and "closed" rises slightly at first. As V climbs past 5,
  "closed"'s advantage turns negative and it starts to fall. The step sizes are illustrative.
- Exploration: greedy decoding always writes "open", scores 2, has advantage 0 and learns nothing.
  Sampling sometimes writes "partly" and receives the 9.
- Clipping: π_old(partly) = 0.2 and ε = 0.2 give the window [0.16, 0.24].
- KL: a flattering answer, "partly. What a brilliant question!", gets 9.6 from a judge that likes
  flattery, against 9 for the plain answer.
  - The reference writes the flattering answer 5% of the time.
  - As the policy's share drifts to 90%, the judge's average rises to 9.54 and KL(π‖π_ref)
    reaches 2.38.
  - With β = 1, R − β·KL falls from 9.03 to 7.16.
- Policy gradient: exact expected-gradient steps on the logits.
  - Probabilities .50/.30/.20 → .15/.36/.49 → .04/.11/.86.
  - V 4.3 → 6.5 → 8.3.
  - At the second step the advantage of "closed" is −1.5.

Values are illustrative, and the article says so. V is shown as exact; in practice V is a learned
estimate. That is why λ ≈ 0.95 is used and not 0.

**Earlier candidates.** CAND-ANIM-038 proposed the garlic example. It had a sign conflict: a
correct answer received R = −2.0, which is recorded there. A capital-of-Australia version was also
drafted on 2026-10-27; the learner preferred this example.

## Outline (H2 sections; figures in brackets)

1. **One score, thousands of tokens.** Hook: DeepSeek-R1-Zero. RL with a rule-based reward that
   checks only the final answer raised AIME 2024 pass@1 from 15.6% to 71.0%, and responses grew
   longer during training. The reader's question: how does one end score teach thousands of tokens?
   [01]
2. **The loop: agent, environment, reward.** The LLM is the agent and its next-token distribution is
   the policy π_θ. The environment is the prompt set plus the judge. [02]
   - Define **judge** once, on first use: the part of the environment that scores the whole answer.
     It can be a reward model (RLHF, trained on preference pairs; X-POST-005), a rule-based verifier
     (RLVR, as in R1-Zero; X-POST-006), executed tests, or another LLM grading against a rubric.
   - Say why the judge sits inside the environment: during RL the reward model is frozen, so from
     the policy's side it is as external as the prompts.
   - Say why the article writes judge and not reward model: R1-Zero (figure 01) has no reward
     model, only rules, and the term must hold for both later articles.
   - The figure keeps the single label "judge" (learner decision, 2026-10-28 GIF review).
3. **State, action, trajectory.** A state is the prompt plus the tokens so far, an action is the next
   token, and a trajectory (rollout, episode) is one full answer. [03]
4. **Reward and return.** The reward is sparse and delayed: 0 on every token and the score at the
   end. The return G_t is everything still to come, and γ = 1 in LLM RL. [04]
5. **Value, Q and advantage.** V(s) is the average score from here if the model keeps writing. Q(s,
   a) is the average score if it writes token a first. A = Q − V is how much better this token is
   than the model's usual choice. [05]
   - **Advantage must be fully clear here (learner request, 2026-10-28).** Build it in this order:
     1. Start from the question a score cannot answer: the judge gave 2. Is 2 good or bad? It
        depends on what the model usually scores at this point. The reward is an absolute score;
        the advantage measures it against the model's own expectation.
     2. The sign gives the direction of the update (raise or lower this token) and the size gives
        how much.
     3. It is relative to the current model. "closed" gets +0.7 although it is not the best
        answer, because it beats the model's average of 4.3. When the model improves and V rises
        past 5, the same "closed" turns negative (figure 08).
     4. The probability-weighted average of A is 0 (−1.15 + 0.21 + 0.94). The update cannot raise
        everything; it moves probability from below-average tokens to above-average ones.
     5. Read it off figure 05: the arrow from the V line to each branch's average is A.
   - A comparison table with three rows (open, closed, partly) and the columns probability, Q,
     A, and update direction makes points 2–4 concrete.
6. **Baseline.** Judge scores are all positive, so raw scores would push every answer up, including
   "open". Subtracting V leaves only better or worse than usual. The direction stays the same on
   average and the noise drops. [06]
7. **Credit assignment: TD error and GAE.** The token where V drops gets the blame. The λ dial runs
   from trusting V (λ = 0) to trusting the final score (λ = 1). GRPO has no V, so every token of an
   answer shares one advantage, including the tokens after "open". [07]
8. **Policy gradient.** Raise log π of tokens with A > 0 and lower those with A < 0, in proportion to
   A. Over many samples the "partly" bar grows. [08]
9. **Exploration.** A greedy policy that always writes "open" never sees a 9. Sampling at temperature
   1 sometimes writes "partly", and that sample is what teaches the model. [09]
10. **Small steps: on-policy, ratio, clipping, KL.**
    - On-policy means learning from your own fresh samples. The ratio π_θ / π_old corrects for small
      drift.
    - Clipping stops the gain outside [1 − ε, 1 + ε]. [10]
    - The KL penalty keeps the policy near the SFT reference, where the judge is reliable. Example:
      a flattering answer that games a judge who likes flattery. [11]
11. **Summary.** The term map (figure 12) places the core terms on 05's picture. The term cards
    (figure 13) follow as a quick reference with all the terms by step, and a short list covers what
    the pictures leave out. Then comes the pointer to the next article:
    preference alignment, where V belongs to PPO, the group average is GRPO's baseline, and DPO
    works offline.

Quote blocks (about 4):

- An LLM is a policy: the prefix is the state, the next token is the action.
- The advantage is how much better this token is than what the model usually writes here.
- A baseline changes the noise, not the direction.
- Credit goes to the token where the expected score changed.

## Writing notes from the GIF review (2026-10-28)

Points the learner raised while reviewing the GIFs. The article must answer each one; no figure
change was requested.

- **Judge (figure 02).** See section 2 of the outline.
- **Why the reward is sparse (figure 04).**
  - The judge can only score a finished answer. A prefix such as "OpenAI is" has no correct score:
    it can still end well or badly.
  - Per-step scores need a process reward model, with step-level labels that are costly and a
    step judge that can be gamed. Sparse rewards are cheap, harder to game, and do not dictate a
    reasoning path, so the model can find its own (R1-Zero's responses grew longer).
  - The cost: one number must be split over thousands of tokens. This credit-assignment problem is
    what V, advantage, the baseline and GAE (figures 05–07) solve.
  - Check the DeepSeek-R1 paper's section on unsuccessful attempts (process reward models) before
    citing it for this point.
- **What γ = 1 means (figure 04).** The return is G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …; γ < 1
  makes later rewards count less. It comes from tasks with long or endless horizons. An LLM answer
  ends, and the score belongs to the whole answer. With γ < 1, early tokens would get less of the
  score even when they decide it, as "open" does, and with a positive score shorter answers would
  be favoured. So LLM RL uses γ = 1.
- **Why every position's return equals the final score (figure 04).** With γ = 1 and a single
  reward at the end, the sum from any position to the end is that one score. The return therefore
  cannot tell which token was good; that bridge leads to V and the advantage A = G − V (figures
  05–07). The article must say this so readers do not take the return as the credit itself.
- **Right answer, wrong or unreadable trajectory (learner question on figure 04).** Yes: a reward
  that checks only the outcome cannot tell how the answer was reached.
  - Example in our own case: "OpenAI is partly. It's in the name." has the right verdict and a wrong
    reason. A judge that checks only the verdict would give it full marks.
  - Forms: lucky guesses; a correct number from a flawed derivation; reasoning that reads well but
    is not what produced the answer (unfaithful chain of thought); drift into unreadable text.
    R1-Zero had poor readability and language mixing, which led to DeepSeek-R1's cold-start SFT
    and a language-consistency reward (verify the wording in arXiv 2501.12948 before citing).
  - Why it still mostly works: the update follows the average over many samples, and lucky paths
    win less often than sound ones, so sound reasoning tends to gain probability. The trend is
    statistical, not a guarantee, and nothing rewards readability unless a term is added for it.
  - Remedies to name briefly: format or language rewards, the KL anchor to the SFT model, process
    reward models or step verifiers, and evaluating the reasoning separately.
  - Placement: one paragraph in X-POST-004 (section 4, the cost of sparse reward). The full
    treatment belongs in X-POST-006 (RLVR). Any further citation, such as work on chain-of-thought
    faithfulness, must be checked before use.
- **Which algorithm is figure 05? (learner question).** None in particular: 05 shows the
  definitions, with V and Q computed exactly. How V is obtained is what separates algorithms:
  - PPO: a learned critic network predicts V; A comes from GAE (figure 07).
  - GRPO: V is the mean judge score of a group of answers sampled for the same prompt, and A is
    divided by the group's standard deviation. With 05's ten scores, A ≈ −0.86 / +0.26 / +1.75
    (population std 2.685; −0.81 / +0.25 / +1.66 with the sample std). Every token of an answer gets
    that answer's A.
  - RLOO: the baseline is the mean of the other samples, giving −2.56 / +0.78 / +5.22.
  - REINFORCE with a baseline: A = R − b, where b is any baseline (figure 06).
  - 05's ten-sample averaging is closest to GRPO and RLOO. Its group starts at "OpenAI is" rather
    than at the prompt, which is the same thing here because every answer starts that way.
  - The article states this in one or two sentences in section 5; X-POST-005 explains PPO and GRPO.
- **What R is, and why it differs from Q (learner question on figure 06).** R is the judge's
  score for one sampled answer: the reward, and with γ = 1 also the return. Q(s, a) is the average
  of R over all answers that start with a. Training never knows Q; it sees one answer and its R.
  So algorithms use A ≈ R − V for each sample (REINFORCE with a baseline, GRPO, RLOO). Averaged
  over samples, R − V equals Q − V.
  - Example, the five open samples of figure 05: R = 3, 1, 2, 1, 3 gives R − V = −1.3, −3.3, −2.3,
    −3.3, −1.3, with mean −2.3 = Q − V.
  - The figures use Q (05, 06) so that only one new symbol appears. The article introduces R − V in
    section 6 as the version used in training.
- **What the baseline does (learner discussion on figure 06).** It keeps the average update and
  cuts its noise.
  - Per sample, the logit gradient is (R − b)·(onehot(a) − π). Since Σ π = 1, subtracting an action-
    independent b adds nothing on average, so the expected direction is unchanged:
    π·A = −1.15 / +0.21 / +0.94 for b = 0, b = V and even b = 20.
  - The variance changes, measured with 200,000 samples (seed 0, 05's per-sample scores, summed
    over the three logits): 20.8 for b = 0, 3.7 for b = V, 136.1 for b = 20.
  - With batches of 4, the share of batches that push "open" down (the right direction) is 81% for
    b = 0, 100% for b = V and 69% for b = 20.
  - The article must say this plainly: figure 06's "all arrows up" describes single samples
    measured from 0. Over many samples b = 0 points the same way; it needs far more samples to get
    there. A good baseline is close to V, and it must not depend on the sampled token, or it biases
    the update.
  - Links: GRPO's group mean and RLOO's leave-one-out mean are baselines; PPO's critic learns V as
    the baseline. Dividing by the group std (GRPO) is a separate rescaling.
- **Code box.** A short pure-Python example, rewards → returns for γ = 1 and γ = 0.9, then G − V,
  may go in the article if the learner wants code in it:
  - rewards `[0]*9 + [2]`
  - return with γ = 1: `[2.0]*10`
  - return with γ = 0.9: `0.77 … 2.0`
  - G − V: `[-2.3]*3 + [0]*7`

## Terms covered (22, from the D39 glossary)

| Group | Terms |
|---|---|
| The loop | agent, policy, environment |
| One answer | state, action, trajectory/rollout/episode |
| The score | reward, sparse/delayed reward, return, discount γ |
| Expectations | value V, action-value Q, advantage A |
| Credit | baseline, credit assignment, TD error, GAE λ |
| Learning | policy gradient, exploration |
| Staying close | on-policy/off-policy, importance ratio, clipping ε, KL penalty β |

## Text rules (series conventions)

- Precise wording, no metaphors: the learner rejected the coach and betting-odds framing (D23).
  Few adverbs, few quotation marks, about 4 quote blocks, and H2 headings only.
- Chinese gives the English term first with the Chinese in parentheses on first use, e.g.
  `advantage（优势）`, and contains no 不是……而是…….
- Each term gets one plain sentence, then the number from the running example. Formulas appear only
  in short inline form (A = Q − V).
- Stay big-picture; no notebook run reports. Every claim maps to `source-map.md`.

## Sources

- Learning note `learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`:
  the D39 glossary and D20–D25 and D35–D38.
- Book Chapter 12 (§12.1 state/action/policy, §12.4 baseline, §12.6–12.8 ratio/clipping/KL,
  §12.10–12.11 critic, TD and GAE).
- SLP3 Chapter 8, preference-alignment and RL sections.
- DeepSeek-R1 paper (arXiv 2501.12948): checked 2026-10-27.
  - "the pass@1 score on AIME 2024 increases from 15.6% to 71.0%"
  - rule-based accuracy and format rewards
  - Figure 3: average response length grows during RL

## Open decisions for the learner

1. ~~Running example~~: decided 2026-10-27, **Is OpenAI open?**
2. Formula chips in GIFs: one short chip per GIF, such as `A = Q − V` (recommended for this
   audience), or none.
3. Figure count: 11 GIFs plus one summary map (recommended), or a shorter set of about 8, made by
   merging 03+04 and 10+11 and dropping 02.
