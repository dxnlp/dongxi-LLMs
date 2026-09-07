# Day 6 chapter example verification

- Mode: smoke; documentation and deterministic arithmetic checks, not training.
- Objective: validate the new Chapter 5 Day 6 code excerpt and the normalization,
  gate-gradient, RoPE, parameter/cache/FLOP examples against the teaching source.
- Use the existing platform Python/Torch on CPU; one Torch thread. Use seed 505
  for the modern decoder, float64 for numerical comparisons, atol 1e-10 and
  rtol 1e-8 unless checking explicitly rounded prose values.
- Reuse the exact existing Day 6 notebook readiness run: three fresh kernels,
  43 code cells, 13 figures, successful source-preserving execution. Keep its
  notebook hashes when recording evidence; do not rewrite learner notebooks.
- Acceptance: the chapter's modern snippet runs; model shape and cache replay
  assertions pass; toy table and larger parameter counts match the source;
  worked gate-gradient and normalized-vector examples agree; relative RoPE
  identity holds; local links resolve; repository tests and diff check pass.
- Read-only source check: recheck pinned Qwen config and the primary paper
  abstracts, distinguish the lab's QK norm from original QKNorm, and cite both
  requested recurrent-depth papers without claiming local trained results.
- No large model allocation, model/data downloads, animation production,
  serving benchmark, training-memory estimate presented as measurement, or
  learner mastery claim. Preserve the running notebook server.
