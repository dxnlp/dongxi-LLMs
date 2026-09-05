# Causal attention forward verification

- Mode: smoke; CPU only, tiny float64 tensors, no training or external data.
- Source base: `7d743e0`; new source and tests committed before execution.
- Inputs: fixed `teaching_inputs()` and random Q/K/V with seed 4.
- Hypothesis: pre-softmax causal masking preserves earlier outputs when only
  future inputs change. Naive post-softmax zeroing and zero-score replacement
  fail this test for the declared counterexample.
- Required evidence: correct rows sum to one, forbidden weights are zero,
  manual output agrees with PyTorch SDPA at atol=rtol=1e-12, prefix-change error
  below 1e-12, both broken-mode prefix-change errors above 1e-3.
- Additional numerical check: post-softmax zeroing followed by renormalization
  agrees in a moderate case; a future score of 10000 causes allowed weights to
  underflow before renormalization. Correct pre-softmax masking stays finite.
- Notebook acceptance: all reference cells execute from a fresh kernel, no
  error outputs; learner placeholders are optional and do not break Run All.
- Failure: any assertion failure or notebook execution error blocks readiness.
- Non-claims: no trained-model quality, general capability, GPU performance,
  gradient correctness, or KV-cache equivalence is established by this run.
- Commands: `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m
  unittest discover -s tests -p test_causal_attention_lab.py -v`; execute the
  notebook using nbclient with the registered platform kernel.
