# FP32 validation artifact after the retained BF16 merge mismatch

The original BF16 merge test failed its predeclared tolerance on the first
prefix:5.9% of logit elements mismatched, maximum absolute discrepancy0.67578125.
Its original collector, source identity and external exit1 remain unchanged.
No merged checkpoint was exported. This negative observation is not repaired by
raising its tolerance or relabeling it a pass.

BF16 merging rounds low-rank updates into a low-precision base. For a distinct
validation artifact, load the same exact Base weight values into FP32, retain
the same actual rank-8 adapter and run both unmerged/merged graphs in FP32/SDPA.
This deliberately changes the numerical precision, not the trained parent,
adapter coefficients, prompts or objective. Predeclare atol0.002/rtol0.001;
report actual max/RMS and argmax differences at all eight original prefixes.
Save full FP32 weights in a new private directory and require exact logits after
reloading those actual saved bytes with the same backend/precision.

The wider representation is not a silently equivalent replacement for the
trained BF16 decoder. Do not claim BF16 merge equivalence, original-policy
quality,400-update selected-parent status or downstream release. It tests a
legitimate explicit full-policy/tokenizer handoff and its reload independently.

One additional owned local-only diagnostic has900-second external deadline,
25GiB sampled reserve and4GiB output planning allowance. Record its export and
24 diagnostic full-prefix forwards separately from prior SFT/I/O journals.
All original byte/interface/rank/target and immutable-input checks remain.
