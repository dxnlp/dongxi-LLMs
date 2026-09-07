# Chapter 5 notebook visual enrichment — reference specification

- User request: replace a text/code-only experience with explanatory visuals.
- Scope: add 25 static, data-backed Matplotlib figures across all eleven
  Chapter 5 notebooks. No animation rendering, no new model-quality claims.
- Preserve every existing notebook cell by ID, source, metadata, execution
  count, and outputs, including learner edits in sessions 01 and 02. Insert new
  visual setup, explanation, and plotting cells only. Preserve Day 3/4 notebooks.
- Plot dependency: Matplotlib 3.10.8, added without replacing existing Torch or
  installed dependencies. Record installed additions and the execution manifest.
- Read actual lesson tensors and results. Detach for display without changing
  model weights, gradients, or random state. Label illustrative layouts,
  coordinate projections, causal masks, units, signed values, and shared scales.
- Acceptance: all eleven notebooks execute with fresh kernels, each produces
  its intended PNG figures, and all repository tests pass. Verify plotted lookup,
  mixture, attention, loss-curve, and accounting data against source tensors.
- Export executed plots as generated PNG preview assets and link them from
  newly added explanation cells, retaining all original learner outputs.
  Identify them as fixed reference previews, not fresh execution of a learner's
  later edits. Re-execution produces a current plot below the saved preview.
- Inspect all rendered figures as contact sheets and key notebook 02 figures
  at full size. Check labels, mask meaning, axis orientation, color scales, and
  absence of clipping. Static plotting is not animation production.
- Preserve the prior report as historical evidence for its hashed source files;
  write a separate report and manifest for this revision. Reuse the original
  fixed training contract without selecting a new seed or success threshold.
