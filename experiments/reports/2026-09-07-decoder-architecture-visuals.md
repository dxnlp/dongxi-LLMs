# Chapter 5 architecture visuals — verification report

## Outcome

Added 23 architecture schematics across all eleven Chapter 5 notebooks: a
highlighted whole-model map and component close-up per notebook, plus an extra
SwiGLU diagram. Together with the 25 existing figures, there are 48 saved PNG
previews and runnable regeneration cells. All 299 pre-existing cells were
compared by ID and full parsed content and preserved, including learner outputs.
Day 3/4 edits were not touched by this work.

## Evidence

- Specification: `experiments/specs/2026-09-07-decoder-architecture-visuals.md`.
- Fresh-kernel execution: 11/11 notebooks passed; 152 code cells; 48 PNG outputs;
  process exit code 0; elapsed 30.67 seconds.
- Repository tests: 47 passed in 4.412 seconds; process exit code 0.
- Test command: `PYTHONPATH=src /home/dongxi/dgx-spark-dongxi/.venv/bin/python -m unittest discover -s tests`.
  An initial invocation without `PYTHONPATH=src` failed during import discovery;
  the corrected invocation above passed without code changes.
- Exported 48 previews; every notebook image link resolves. Verified notebook
  source hashes against the final manifest; `git diff --check` passed.
- Inspected architecture contact sheets, then full-size embedding, attention,
  residual-block, and training diagrams. Corrected the second residual bypass
  to branch from the post-attention sum before the final verification run.
- Training sanity check unchanged: seed 505, 160 steps, learning rate 0.02;
  loss 2.774792432785034 to 0.0007856183219701052; batch accuracy 1.0.
- Executed copies: `/tmp/chapter5-reference-4oppn8cs` (temporary, not a portable
  dependency). The adjacent JSON stores environment identity, source and asset
  hashes, per-notebook counts/timing, and the complete learning trace.

## Interpretation and boundaries

The diagrams explain lookup addition, Q/K/V branches, causal attention, residual
bypasses, normalization, MLP/SwiGLU, the decoder-to-loss training loop, RoPE,
grouped head sharing, and recurrent parameter reuse. Baseline learned-position
embeddings are distinguished from the modern decoder's RoPE inside attention.
Projection shapes use row-vector mathematical notation; captions explain
PyTorch's transposed weight storage. The GQA close-up explicitly omits RoPE and
optional Q/K normalization to isolate head sharing.

Schematics describe structure, not measured activation values, speed, memory,
or learner mastery. Existing numerical plots retain their separate data-backed
interpretations. No animation production or publication was performed. The
Jupyter server and learner kernels were not restarted or terminated.

The verifier emitted local-kernel TCP encryption warnings; these runs do not
establish network-security properties or authorize exposing Jupyter remotely.

Reusable sources: `src/dongxi_llms/decoder_architecture.py` and
`decoder_visuals.py`. Reproduction instructions: `docs/NOTEBOOK_VISUALS.md`.
