# Day 8 course-material verification

- Mode: smoke; prepared before execution, 2026-09-09.
- Question: can a small decoder make data, update, validation, and recovery
  contracts observable without running a model-scale pretraining campaign?
- Scope: three CPU notebooks, authored in-repository text fixture, source helpers,
  tests, explanatory figures, Chapter 6 Day 8 prose, and worked solutions.
- Predictions: document-isolated windows prevent cross-document context; valid-
  token-weighted accumulation matches one full-batch gradient; averaging unequal
  microbatch means generally does not; complete boundary checkpoints replay the
  deterministic CPU fixture while omitting optimizer or data-order state changes it.
- Identity: base `f9b8d97`; preserve existing uncommitted learning/handoff edits.
  Report actual source, notebook, data and environment hashes after creation.
- Execution: existing platform Python on Spark, CPU only, one PyTorch thread,
  seed 808, synthetic authored byte-token corpus, width 16, two decoder blocks,
  vocabulary 258, sequence length 16; float64 gradient equality probes and
  float32 short update/recovery probes. At most 24 updates per recovery branch;
  no CUDA allocation, network dataset/model download, or persistent server.
- Hypothesis supported if notebooks execute in fresh kernels, solutions/assertions
  pass, all important losses/gradients remain finite, replay is exact within the
  specified same-runtime fixture, deliberate controls fail as intended, and
  plotted values agree with measured inputs. Report failed checks, not only fixes.
- Check malformed inputs, overlapping splits, clipping placement/scale, LR endpoints,
  unique parameter accounting, validation weighting, and checkpoint contract mismatch.
- Verification includes math formatting, local links, existing notebook preservation,
  and visual inspection. Saved previews are generated from executed references.
- Fail/block if nonfinite values, incorrect expected equivalence, broken fixtures,
  failed kernels, unsafe memory pressure, or missing environment requirements.
- Limits: not language quality, production data readiness, cross-device bitwise
  reproducibility, measured GPU capacity, or authorization for a Day 9 run.
