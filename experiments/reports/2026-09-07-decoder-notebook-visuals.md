# Chapter 5 — Explanatory Notebook Visuals

## Result

Added **25 static explanatory figures across all eleven Chapter 5 notebooks**.
Each has a reading guide, a saved reference preview visible before running code,
and a plotting cell that regenerates from the current lesson state.

- All eleven notebooks passed fresh-kernel execution: **129 code cells and 25
  PNG figure outputs**. Verification process exit code: 0.
- All **44 repository tests** passed, including six new figure-data and
  side-effect checks. Unit-test process exit code: 0.
- **238 original notebook cells** were preserved exactly as JSON objects,
  including source, metadata, execution counts, and outputs. This includes
  learner-edited Notebook 01 and Notebook 02. Only new cells were inserted;
  notebook-level reference metadata was updated. Day 3/4 edits were untouched.
- [Specification](../specs/2026-09-07-decoder-notebook-visuals.md).
- [Execution manifest and source/asset hashes](2026-09-07-decoder-notebook-visuals.json).

## What the figures explain

| Notebook | Figures |
|---|---|
| 01 Embeddings | Table-to-sequence row selection; token/position/sum matrices; accumulated lookup gradients |
| 02 Multi-head attention | Four query-head views; four causal probability maps; actual weighted value contributions; head-ablation output change |
| 03 Residuals | Actual two-coordinate state addition and backward gradient contributions |
| 04 LayerNorm | Centering versus scaling; wrong-axis future leakage |
| 05 MLP | GELU/SiLU curves; expanded versus activated features; isolated position influence |
| 06 Decoder | Structural decoder route; vocabulary-wide logits per input position |
| 07 Learning | Measured loss curve and before/after token probabilities with observed labels |
| 08 Modern layers | LayerNorm/RMSNorm comparison; gate/content/product features |
| 09 RoPE | Coordinate-pair rotation; correct versus incorrect cache-offset errors |
| 10 GQA | Query-to-KV sharing diagram; stored parameters, compact cache bytes, and FLOP estimates |
| 11 Recurrence | Stored parameters versus block applications; distinct keys across shared-block applications |

Heatmap comparisons use shared scales. Attention maps have explicit query/source
axes, probabilities in [0,1], and forbidden futures shown separately from small
allowed probabilities. Vector diagrams explicitly show only two coordinates.
Schematic arrows are not attention probabilities or performance measurements.
All random-vector fixtures retain their untrained-model evidence boundary.

## Environment and dependency change

The existing CPU reference environment remains Python 3.12.14 and PyTorch
2.13.0+cu130. Matplotlib **3.10.8** was added to the active notebook environment
after inspecting an additive-only installation preview. Seven new packages were
installed; no existing package, Torch version, or platform lock file was replaced:

- matplotlib 3.10.8
- contourpy 1.3.3
- cycler 0.12.1
- fonttools 4.64.0
- kiwisolver 1.5.1
- pillow 12.3.0
- pyparsing 3.3.2

`notebooks/requirements-visuals.txt` pins the direct plotting dependency. The
notebooks still need no GPU or runtime network download once dependencies are
installed. DejaVu Sans provides an available, portable font in the Spark setup.
Mac execution remains unverified.

The final verifier ran in 30.62 seconds, including kernel startup, figure
rendering, and a repeat of the original tiny training fixture. Training remained
unchanged: loss 2.774792432785034 to 0.0007856183219701052 after 160 steps,
training accuracy 1.0. This is a regression check, not a new capability result.

## Visual and preservation checks

Inspected all 25 exported figures in three contact sheets, then inspected
Notebook 02's attention maps and weighted-value view plus the complete decoder
schematic at full size. Checked labels, mask orientation, shared scale usage,
legend meaning, and clipping. No execution or unit-test acceptance failed.

The figure tests additionally compare displayed lookup rows, attention arrays,
masked cells, value contributions, probability arrays, learning-curve update
indices, and count bars with their supplied tensors. A side-effect test verifies
that lookup plotting does not change input values, gradients, or Torch RNG state.

The static previews were exported from executed reference copies. Original
learner code and outputs were not replaced by those executed copies. Their
notebook cells were compared by stable ID before and after insertion. Notebook
plot outputs produced during future learner execution appear below the fixed
reference image; the caption explicitly identifies this distinction.

## Reproduce and maintain

Use `scripts/verify_decoder_notebooks.py`, then pass its newly printed temporary
output directory to `scripts/render_decoder_notebook_figures.py`. The latter
updates only generated PNG assets, not notebook sources or learner outputs.
The source is `src/dongxi_llms/decoder_visuals.py` plus the plotting calls in
each notebook. Design/usage contract: `docs/NOTEBOOK_VISUALS.md`.

This report supersedes the September 6 report for the visual notebook revision,
without rewriting that historical run or its source hashes. It does not claim
learner mastery, semantic interpretation of untrained heads, measured serving
speedups, new model-quality gains, or completed animation production. Existing
animation candidate records are reused; no animation was rendered.
