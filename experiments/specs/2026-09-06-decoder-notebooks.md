# Chapter 5 notebook reference verification

- Mode: reference checks and one tiny learning run; CPU only, one Torch thread.
- Scope: eleven new Chapter 5 notebooks, including optional fixed recurrence.
- Source identity: working tree based on `6d62d64`; record exact source hashes in
  the execution report. Existing learner Day 3/4 edits are excluded and preserved.
- Environment: existing DGX Spark native Python, Torch, nbformat, nbclient;
  record versions, kernel, platform, execution time, and initial MemAvailable.
- No pretrained weights, datasets, network calls in notebooks, GPU jobs, serving
  benchmark, or animation rendering. Fresh kernel per notebook, 180s cell timeout.
- Shapes: B=2, T=6, vocabulary=16, D=16, H=4, d=4, 2 blocks, MLP width=32.
  Smaller explicit fixtures are allowed for isolated mechanisms.
- Float64 identity/derivative checks: generally atol=1e-10, rtol=1e-8; relaxed
  tolerances must be stated for numerical gradcheck. Training uses float32.
- Seed: each notebook explicitly resets to 505; controlled alternatives share
  weights or inputs as declared. Report failures without seed selection.
- Learning fixture: `[1,2,3,4,5,6,7]` and `[8,9,10,11,12,13,14]`, shifted into
  2x6 inputs and labels. The exact in-code sequences define revision 1.
- Training: 160 full-batch AdamW steps, learning rate .02, weight decay 0,
  no scheduler, no clipping, no dropout, no early stopping or held-out selection.
- Learning acceptance: finite losses/gradients throughout, final CE < .05,
  exact training-token accuracy 1.0; compare frozen initial and trained model.
- Mechanism acceptance: reference forward/backward matches, causal invariance,
  compact GQA and cached/full replay equivalence, expected broken variants,
  analytical/stored parameter and cache byte agreement, recurrence shared-weight
  gradient accumulation. Each notebook includes adjacent runnable answers.
- Recurrence is a fixed-block forward/backward identity experiment, not a trained
  architecture comparison; no quality or wall-clock advantage is hypothesized.
- Evidence limits: agent execution does not establish learner mastery; one-batch
  fitting does not demonstrate language understanding or generalization; analytical
  cost savings are not measured speedups. Mac execution remains unverified.
