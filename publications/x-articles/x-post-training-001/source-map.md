# Source map — X-POST-001

The public body has no links. Every factual claim maps to a source here.

## Primary reference

Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), Chapter 8
"Post-training", draft of 2026-08-19,
https://web.stanford.edu/~jurafsky/slp3/8.pdf. Read in full on 2026-10-20;
SHA-256 `802ca6462e52530a766ef0c83c1d1d950d8c13c7e508a5d200436a574648e112`
(identical to the local Zotero copy).

| Article claim | Source |
|---|---|
| Moon-landing hook: GPT-3 answers "Explain the moon landing to a 6 year old in a few sentences." with more "Explain … to a 6 year old" prompts; InstructGPT answers that people went to the moon, took pictures and sent them back | OpenAI, "Aligning language models to follow instructions" (blog, January 2022), https://openai.com/index/instruction-following/ ; live page returned 403 on 2026-10-20, wording verified from the Internet Archive snapshot of openai.com/blog/instruction-following/. EN quotes the prompt and a 12-word excerpt; ZH translates. InstructGPT = GPT-3 after post-training (Ouyang et al., 2022). Replaces the earlier SLP3 grocery-store hook at the learner's request (more engaging). |
| Pretraining objective = predict the next word in mostly web text; mismatched with helpful/non-harmful | SLP3 Ch. 8 intro |
| Three gaps: instructions, harm (incl. false medical answers), long multi-step reasoning | SLP3 Ch. 8 intro |
| "Base model", "post-training" definitions | SLP3 Ch. 8 intro |
| Three techniques: instruction tuning (SFT), preference alignment, RLVR | SLP3 Ch. 8 intro and §8.5 summary |
| Instruction tuning uses the same next-word objective, trained only on the response; called SFT because each instruction has a correct response | SLP3 §8.1, Fig. 8.1 |
| Instruction tuning improves instruction following in general, evaluated on held-out tasks | SLP3 §8.1, §8.1.2 |
| At most millions of instruction examples vs trillions of pretraining tokens; small fraction of base-model cost | SLP3 §8.1 |
| 500 safety examples substantially reduced harmfulness | SLP3 §8.1.1, citing Bianchi et al. (2024), *Safety-Tuned LLaMAs*, ICLR |
| Spanish→English translation example | SLP3 Fig. 8.1 (adapted) |
| Preference judgments need only an opinion on the result, not skill to produce it | SLP3 §8.3 (food-critic analogy is ours) |
| Data = prompt + two outputs + preferred choice | SLP3 §8.3.1 |
| Reward model scores answers; RL (PPO) optimizes it; DPO learns from pairs without an explicit reward model | SLP3 §8.3.3, §8.4, §8.4.2 |
| KL penalty keeps model near the reference; otherwise it forgets pretraining; goal is to nudge, not radically alter | SLP3 §8.4, eq. 8.6 |
| RLVR: multi-step problems trained on tasks with a clear right answer (programming, math) | SLP3 Ch. 8 intro (one sentence; the draft has no RLVR section) |

## Supplementary sources for RLVR

SLP3's draft has no RLVR section, and §8.4.1 (PPO) reads "TBD". The article's
RLVR paragraph keeps to intuition and draws on:

- Book Chapter 13 (`book/chapters/13-group-relative-policy-optimization.md`):
  verifier definition, "a verifier can't be flattered, but it is only as good as
  the rule it runs", outcome-only checking does not verify every step.
- Book Chapter 8 introduction: three techniques in the same order and wording.
- Lambert et al. (2024), *Tülu 3*, which introduced the name RLVR; DeepSeek-AI
  (2025), *DeepSeek-R1*, rule-based accuracy rewards on math and code. Neither
  is named in the beginner body; the per-technique RLVR article will cite them
  after re-checking.

## Our own framing (not from SLP3)

- Web text often follows one question or assignment with another (worksheets,
  homework lists): offered as a plausible intuition for the example, not a
  measured cause.
- Library-reader, new-employee, food-critic and leash analogies.
- "One shows, one compares, one checks"; "pretraining gives what it knows;
  post-training shapes how it uses it" (stated as "roughly"; RLVR can also
  improve skills).

## Evidence boundary

No model was run. A1's probabilities and short replies are authored
illustrations modeled on the OpenAI example. No benchmark,
measured behavior change or cost figure beyond those quoted from SLP3.
