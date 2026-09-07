# Chapter 5 — Modern Decoder Sessions

These CPU-first notebooks are built and reference-verified. Study after the
seven baseline sessions in [the full pathway](../day-05/README.md).

Read alongside [Chapter 5, modern mechanisms](../../book/chapters/05-building-a-modern-decoder.md#511-modernize-mechanisms-not-just-names)
and [worked answers 13–24](../../book/solutions/05-decoder-notebook-solutions.md#day-6-modern-decoder--worked-conceptual-solutions).
The narrative follows the same order: local replacements, positional/cache
consistency, then grouped attention and design accounting.

8. [RMSNorm and SwiGLU](01_rmsnorm_and_swiglu.ipynb)
9. [Rotary positions and cached decoding](02_rotary_positions.ipynb)
10. [GQA, QK normalization, and costs](03_gqa_qknorm_and_costs.ipynb)

Each session contains predictions, optional implementation cells, adjacent
runnable reference answers, controlled changes, and explanations. Source outputs
now include saved reference figure previews; choose the DGX Spark Native kernel on Spark or an
equivalent local Python kernel with Torch installed. Mac execution is unverified.
Install the additive `../requirements-visuals.txt` plotting dependency first.
No runtime downloads or GPU are needed for these fixtures. Tests and original execution evidence
are linked from the [report](../../experiments/reports/2026-09-06-decoder-notebooks.md).
