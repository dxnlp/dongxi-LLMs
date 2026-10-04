# Chapters 13–15: bounded mechanism experiment contract

Written before the report-producing run. Mode: CPU mechanism, original procedural
data. Seed2223 for autoregressive policy;2628 for inference sampling. No model or
dataset download, GPU job, inference service or external publication is implied.

## Hypotheses and contrasts

1. Real TinyDecoder rollout → verifier → group advantage → backward → update
   executes with detached behavior/reference snapshots, valid EOS masks and finite
   telemetry. It need not improve four held-out arithmetic prompts.
2. Group 4 versus group8, each12 fresh synchronous updates, changes exploration
   geometry at a documented unequal sample budget. Both initialize from the same
   seed and identical60-update SFT warm start. This is a cost/geometry comparison,
   not an equal-token superiority test. Keep the two-prompts-per-update schedule,
   AdamWlr.002,clip 1,beta.02,max3response tokens fixed.
3. An exploitable expected reward increases while strict correctness falls; a
   strict verifier trained from the same initial logits repairs this three-action
   problem. The repair is not a continuation from the hacked checkpoint.
4. T² forward-KL SGD brings a three-logit student closer to its frozen teacher.
   It does not prove transferred reasoning or faster real model inference.

## Data and evaluation

Arithmetic prompts are all ordered pairs0..3. Four fixed held-out pairs
`(0,2),(1,3),(2,0),(3,1)` are excluded from all SFT/RL data; their result values
also occur in training. Token vocabulary12 contains PAD/BOS/EOS/EQ and numeric
tokens. The strict token verifier accepts one correct numeric token followed by
EOS. Truncated answers and malformed sequences fail. Held-out prompts are never
used to change the learning recipe or pick an update. All outputs are retained.

The original reward-hack actions are `35`, `35 0`, `0`; strict truth accepts only
one integer equal35. The intentionally broken proxy rewards these1,2,0.

## Acceptance and report

Run `scripts/run_grpo_capstone_experiments.py` with the verified course interpreter.
It writes bounded report JSON/Markdown with exact command, Python/Torch identity,
source hashes, elapsed time, every update, all initial/final held-out rows,
selection simulation and gradient checks. Reject nonfinite measurements. Run
`tests/test_grpo_capstone_labs.py`; disagreements block the mechanism claim.

The policy optimizer minimizes the negative response-mean clipped surrogate plus
exact forward categorical KL averaged over sampled prefix states. Population
group standard deviation and epsilon 1e-8 are declared. It is an educational
GRPO variant, not a universal or framework-default GRPO definition.
