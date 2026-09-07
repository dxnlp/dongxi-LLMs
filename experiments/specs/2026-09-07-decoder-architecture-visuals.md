# Architecture schematics for Chapter 5 — specification

- User approved adding whole-model orientation maps and component close-ups.
- Add a highlighted whole-model map and a detailed component schematic to every
  Chapter 5 notebook, plus a gated-MLP close-up in the RMSNorm/SwiGLU session:
  23 new diagrams, alongside the previous 25 data-backed plots.
- Schematic shapes are symbolic; diagrams do not claim measured values or
  performance. Distinguish baseline learned positions from modern RoPE and show
  Q/K/V branches, residual bypasses, normalization, MLP feature widths, training
  feedback, and parameter sharing precisely.
- Preserve every pre-existing saved notebook cell by ID, source, metadata,
  execution count, and outputs. Insert only new cells; update the notebook-level
  reference pointer. No animation rendering, no learner mastery claim.
- Execute all 11 notebooks in fresh kernels, export every figure referenced in
  the new manifest, and run all tests. Inspect architecture contact sheets and
  the embedding, attention, block, and training close-ups at full resolution.
- Keep the active Jupyter server and learner kernels running. Existing Day 3/4
  edits remain untouched. Do not reset or overwrite a stale browser copy.
