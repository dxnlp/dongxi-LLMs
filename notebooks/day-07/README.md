# Day 7 — Architecture synthesis and optional recurrence

Study in this order; numeric filenames preserve older links rather than defining
the pedagogical order:

1. **Core:** [Architecture defense](02_architecture_defense.ipynb).
   Trace the actual modern decoder, reconcile unique parameters and cache bytes,
   inspect shared loss gradients, diagnose wrong-offset and wrong-axis variants,
   and write a budget-controlled comparison proposal. Includes architecture
   diagrams, data-backed plots, safe attempt cells, and adjacent worked solutions.
2. **Optional:** [Recurrent depth](01_recurrent_depth.ipynb).
   Apply one shared block repeatedly, compare independent copies, verify gradient
   accumulation, and separate stored parameters from block applications and K/V
   activations. No adaptive router or trained quality gain is implied.

Prerequisites: [Chapter 5 Days 5–6](../../book/chapters/05-building-a-modern-decoder.md)
and the [Day 6 notebooks](../day-06/README.md). Both sessions use small CPU
fixtures, Torch, and the Matplotlib dependency in `../requirements-visuals.txt`.
Choose **Python (DGX Spark Native)** on Spark or a validated local equivalent;
Mac execution is not claimed. Saved PNG previews are visible before execution.

The architecture defense is an explanation backed by controlled evidence, not
merely successful cell execution. A trained recurrence comparison requires a
separate approved experiment with concrete data/seed/budget/measurement choices.
Animations and public articles remain separately approved Mac Studio work.

See the [verification report](../../experiments/reports/2026-09-07-day7-architecture-defense.md)
and [worked-solution guide](../../book/solutions/05-decoder-notebook-solutions.md).
