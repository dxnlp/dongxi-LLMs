# Three post-training families and where RLVR sits

Opened 2026-10-20 in discussion. Source: [SLP3 Chapter 8](https://web.stanford.edu/~jurafsky/slp3/8.pdf)
(Jurafsky & Martin), which divides post-training into three techniques: instruction
tuning, preference alignment, and reinforcement learning with verifiable rewards (RLVR). Status: framing
discussed only; no prediction, explanation or exercise from the learner yet.

## Learner's question

Does the course already teach RLVR? Yes. The three families map onto the existing chapters as follows:

| Family | Training signal | Course chapters | Days |
|---|---|---|---|
| Instruction tuning (SFT) | demonstration: target response tokens | [8 Instruction data](../../book/chapters/08-instruction-data-as-an-interface.md), [9 SFT](../../book/chapters/09-supervised-fine-tuning.md) | 11–14 |
| Preference alignment | comparison: chosen vs rejected response | [10 Preferences and reward models](../../book/chapters/10-preferences-and-reward-models.md), [11 DPO](../../book/chapters/11-direct-preference-optimization.md) | 15–18 |
| RLVR | outcome: a program grades the model's own sampled answers | [12 Generation as a policy](../../book/chapters/12-language-generation-as-a-policy.md), [13 GRPO](../../book/chapters/13-group-relative-policy-optimization.md) (§13.1 names RLVR, §13.6 the verifier contract), [14 Failures](../../book/chapters/14-when-optimization-goes-wrong.md) | 19–25 |

## Distinction worth keeping

The three families differ mainly in **where the training signal comes from**. They are not three optimization algorithms.

- RLVR names a reward *source*: an executable checker. GRPO is an *optimizer*. GRPO
  can be trained on a reward-model score, and an RLVR reward can drive PPO or RLOO
  (Day 20). The course teaches RLVR mostly through GRPO, so the two are easy to
  conflate.
- Preference alignment can be done with RL (RLHF: reward model + PPO) or without it
  (DPO). The difference between "preference alignment" and "RLVR" is therefore a
  learned reward of human judgment versus a programmatic check, not RL versus no RL.
- SFT and DPO learn from fixed, pre-collected responses. RLVR (like RLHF-PPO) learns from the
  model's own fresh samples, which is why Chapter 12's policy view and Chapter 14's
  rollout systems belong to it.
- "Verifiable" means a program decides some property; it does not certify the reasoning
  (§13.6). The native Qwen audit shows the checker *is* the task: a correct `5 + 6 = 11`
  still failed a strict extractor ([focused artifact](generated-answers-and-grader-interfaces.md)).

## Open questions for later discussion

- How does SLP3 Chapter 8 define each family, and does its boundary between them match
  the course's? Map its sections onto Chapters 8–14 once read. Read on 2026-10-20 (draft
  of August 19, 2026): §8.1 instruction tuning, §8.2 LoRA/PEFT, §8.3 preference data and
  Bradley–Terry reward models, §8.4 KL-regularized RL objective and DPO. In that draft, §8.4.1
  (PPO) is "TBD" and RLVR is named in the introduction only, with no section of its own. For
  RLVR, the course's Chapters 12–14 go further than the current SLP3 draft.
- SLP3 lists LoRA as part of post-training; check how deeply the course's Chapter 9 LoRA
  material covers the $`W+AB`$ low-rank update.
- Which tasks are actually verifiable (math, code, format) and what happens to post-training
  goals that have no checker?

## Article or lesson opportunity

A short "three signals, one model" opener for the post-training part: show the same
prompt producing an SFT target, a chosen/rejected pair, and a verifier score, then
show which update each one drives. Proposed only; not approved production.
