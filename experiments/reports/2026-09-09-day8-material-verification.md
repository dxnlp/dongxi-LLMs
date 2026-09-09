# Day 8 material verification — 2026-09-09

## Outcome

The requested complete Day 8 subject material is prepared: Chapter 6 foundation,
three worked visual notebooks, twelve conceptual solutions, companion lab and
bounded recipe. All three notebooks execute in fresh kernels: **29 code cells,
10 PNG outputs**. **71 repository tests pass**, including 12 new pretraining
tests. All **605 math expressions in 16 book Markdown files** pass the source
check and render locally with MathJax 3.2.2.

This establishes reference-material readiness, not learner mastery or Day 9
GPU-run completion. The learner's current continuation stays Day 8 on Spark.

## Specification, identity and execution

- Pre-execution specification: [material verification](../specs/2026-09-09-day8-material-verification.md).
- Reader-facing bounded recipe: [Day 8 control and GPU gates](../specs/2026-09-09-day8-bounded-pretraining.md).
- Base commit: `f9b8d97237763814c8fc78e5c19abac3c829f1a2`; main matched fetched
  origin before work, while existing uncommitted learning/handoff edits were
  preserved. No reset, stash, commit or push performed by this task.
- Host: `spark-aa66`, Linux 6.17.0-1031-nvidia, aarch64, glibc2.39.
- Interpreter: `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`, Python3.12.14;
  Torch2.13.0+cu130, Matplotlib3.10.8, nbformat5.11.1, nbclient0.11.0.
- Notebook kernel: `dgx-spark-native`. CPU computation, one Torch thread,
  float64 equality probes and float32 update/recovery probes. No CUDA training,
  pretrained download, external dataset, environment installation or server start.
- Final notebook verification timestamp: 2026-09-09T20:43:59 UTC; elapsed
  11.486 seconds for the verifier, not a model-throughput benchmark.
- Host MemAvailable sampled about every0.1s, 115 samples: minimum
  **125669801984 bytes (117.039 GiB)**. This exceeds the 25GiB reserve during the
  sampled verification interval; it is not continuous monitoring or GPU peak
  allocation evidence. The full test suite ran concurrently during this interval.
- Actual exit codes: notebook verifier0; complete test suite0; book math check0;
  local MathJax renderer0.
- [Machine-readable evidence](2026-09-09-day8-material-verification.json) stores
  notebook/helper/verifier/test hashes, corpus fingerprints, generated-image
  hashes, versions, complete reference stdout and sampled memory provenance.
- Executed notebook copies: `/tmp/chapter6-reference-a9d2shkw`; earlier passing
  execution: `/tmp/chapter6-reference-y_2fxyvu`. These temporary paths are not
  portable artifacts; the JSON report and ten committed-intent reference PNGs
  preserve the important outputs in the repository. No commit is claimed here.

## Observations

| Probe | Measured result | Interpretation boundary |
|---|---|---|
| Corpus targets | Training334, validation65 | Authored fixture, not representative language data |
| Unique parameters | 8832 | Modern teaching decoder, not a model-scale experiment |
| Accumulation, valid counts16 and1 | Correct max gradient error5.551115123125783e-17; wrong mean-of-means error1.3305337336094731 | Correct denominator reproduces the full-batch objective in this deterministic float64 probe |
| Hand-written AdamW vs PyTorch | Max parameter error6.938893903907228e-18 | Tested unfused variant and explicit settings; sequential recurrence covered by unit tests |
| 24-update training exposure | 668 valid targets, 768 processed positions | Presentations include repeated corpus data; not768 independent targets |
| First/last current training-batch loss | 5.530887126922607 → 3.280638813972473 | Different batches measured before their respective updates |
| First/last held-out loss | 5.5187100923978365 → 3.275008803147536 | Same tiny holdout after each update, not proof of broad capability |
| Complete checkpoint recovery | Max parameter error0; subsequent history and batches identical | Exact replay only in the tested same-runtime CPU fixture |
| Missing optimizer state | Max error0.03399824723601341; batches identical | Isolates moment-history loss |
| Missing data cursor | Max error0.02556600421667099; batches differ | Isolates a changed continuation stream |

The missing-cursor branch's final held-out loss is3.266824751633864, slightly
lower than the complete branch's3.275008803147536. It still fails recovery.
This negative control demonstrates why lower loss cannot validate experiment
identity; it is not evidence that cursor resets improve training generally.

The FP32 persistent-state ledger estimates141312 bytes for this model (weights,
gradients and two moments only). It is not measured total or peak memory. The
FP16/BF16 examples are casts on CPU; no BF16 training-performance claim is made.

## Validation scope and inspection

Tests cover normalized split overlap, exact input/label alignment, Unicode byte
targets, valid target accounting, accumulation equivalence and its broken
control, multi-update AdamW recurrence, schedule endpoints/bounds, clipping,
shuffle recovery across epochs, validation weighting/mode, checkpoint contract
rejection/RNG restoration, full/incomplete replay, and visual tensor/RNG
non-mutation. Existing model and learner notebooks are preserved.

Inspected full-size system map, input/label grids, accumulation bars, optimizer
comparison, warmup schedule, clipping/memory figure, validation curves and
recovery comparison. Axis labels, ignored targets, schematic-vs-measured labels,
and fixed-reference-vs-live-output distinctions are readable. Three process maps
reuse the same tested layout with different highlights. All ten image paths are
paired with an actual fresh-kernel PNG output by the verifier.

No assertion failures occurred. An initial helper run emitted a tensor-to-scalar
autograd warning in the parameter comparison; it was corrected to detach before
comparison, then the suite and notebooks were rerun. Ephemeral kernel startup
emitted an unencrypted-TCP transport warning; no persistent Jupyter server or
public endpoint was launched, and no security-hardening claim is made.

Book and notebook Markdown local-link/image validation checked161 targets with no
missing paths; Git diff whitespace checks pass, and Days3–7 notebooks plus the
existing decoder implementation have no diff against HEAD.
Local math validation is not a live GitHub deployment/browser check. Source
notebooks have no saved execution outputs: readers see explicitly labeled
reference images and can regenerate them with their chosen verified kernel.
Mac dependency readiness and execution remain unverified by this Spark result.

## Reproduction

```bash
PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests -q
/home/dongxi/dgx-spark-dongxi/.venv/bin/python scripts/verify_pretraining_notebooks.py --kernel dgx-spark-native --figures
python3 scripts/check_book_math.py
```

Use the optional local renderer instructions in `docs/MATH_FORMATTING.md` for
MathJax. No existing source notebooks are overwritten by these commands.

## Remaining work

Guide the learner through Day 8 with concrete recipe dilemmas, beginning with
data/windows and the valid-token objective. Earlier Days4–7 gaps remain review
items, not an automatic rewind. A larger GPU recipe still needs a corpus choice,
hardware/precision profile, resource budget and explicit execution approval.
Animation opportunities were captured without rendering; production remains on
Mac Studio after approval. No claim of course completion or public publication.
