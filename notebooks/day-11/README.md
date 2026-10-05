# Day 11 — Instruction Data as an Interface

These are complete CPU lessons for Chapter 8. Material readiness is separate from learner practice and real-model evidence.

- [Which tokens are we teaching the model to write?](01_roles_templates_and_masks.ipynb).
- [Where can information flow in a packed sequence?](02_padding_packing_boundaries.ipynb).
- [What distribution does the objective actually emphasize?](03_mixtures_and_data_cards.ipynb).
- [What survives teacher filtering and matched selection?](04_teacher_attempts_and_matched_rejection_sft.ipynb).

Each notebook contains predictions, adjacent explanations and runnable solutions, and plots from the actual lesson tensors or declared illustrative fixtures. Use the course kernel with PyTorch and Matplotlib. No downloads or GPU work are required.

Reader route: [chapter](../../book/chapters/08-instruction-data-as-an-interface.md), [solutions](../../book/solutions/08-instruction-data-as-an-interface.md), [lab](../../book/labs/08-instruction-data-as-an-interface.md).

The teacher extension uses an original programmatic source, all raw attempt/error
records and actual randomly initialized sequence students. Its filter retains
well-formed wrong answers. Top/random share the frozen pool and prompt counts;
the length-random sensitivity makes target exposure explicit. All fixed-seed
held-out failures remain visible. This is neither pretrained teacher evidence
nor demonstrated general instruction-following transfer.
