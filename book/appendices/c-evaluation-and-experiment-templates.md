# Appendix C — Evaluation and Experiment Templates

Use these forms to make a comparison reviewable before expensive execution.
They are reasoning aids; a filled form does not prove that its protocol was followed.

## C.1 Capability contract

Write the desired behavior as a decision on observable outputs. For storytelling,
separate sentence fluency, entity continuity, causal continuity, repetition and
termination. For arithmetic, define exact accepted answer syntax and whether
reasoning text may surround it. Specify parser failure as an outcome, not a
missing row that quietly disappears.

Record task population, split, prompt construction, tokenizer/template identity,
generation settings, seed policy, sample count, scoring rule and uncertainty
unit. If several completions share one prompt, independence at the prompt level
matters for uncertainty. Keep development and final-test identities separate.

```json
{
  "capability": "exact answer to original procedural addition prompts",
  "split_policy": "disjoint prompt identities and operand ranges",
  "generation": {"method": "greedy", "maximum_new_tokens": 32},
  "scoring": {"parser": "declared exact format", "invalid": "failure"},
  "comparison": "paired prompts, fixed scorer and decoding",
  "uncertainty_unit": "prompt",
  "status": "proposed; resolve fixture identity before execution"
}
```

This example specifies an intended contract. It is not a claim that a Qwen
checkpoint has already been evaluated under it.

## C.2 Experiment specification

State the hypothesis, independent variable, controls, budget and failure condition.
Name what remains unequal when matching compute, parameters or elapsed time.
Record source hashes when work is uncommitted; a base commit alone cannot
identify code that has already changed locally.

Use `experiments/specs/template.yaml` and include model/tokenizer/data revisions,
optimizer/schedule, loss reduction, attention and padding policy, precision,
batch geometry, update/target clocks, evaluation, checkpoints and resource cap.
Declare the smallest useful smoke run before a learning run.

## C.3 Result ledger

The report records actual commands, actual exit status, completed work, environment,
measured resources, metric denominators, representative outputs and deviations.
Classify each conclusion by its evidence: observed, analytical, interpretation,
or untested hypothesis. Preserve failed controls and unfinished runs.

For a paired checkpoint comparison, retain a row for every prompt even when
generation times out or parsing fails. An improvement claim should identify
which metric, which slice, uncertainty and computational cost. Keep contrary
slices visible.

## C.4 Checkpoint genealogy

Record parent ID/hash, objective, data identity, code/environment, completed
updates/targets, adaptation method and evaluation contract. Name whether a
checkpoint resumes optimizer state or is used as weight initialization only.
Reference and rollout snapshots also need identities in preference/RL runs.

A model card should explain intended behavior, data provenance, measured results,
known failures, resource needs and limitations. Filling a card does not authorize
publishing weights; resolve licenses and the external release decision separately.

## C.5 Review questions

Before accepting a claim, ask: could a different denominator create this result?
Could the model see the answer in its input? Were outputs selected by quality?
Did a template or parser change? Were repeated development probes called tests?
Did a sampled reward become confused with independent task success? Would another
person know which file, checkpoint and environment to use?
