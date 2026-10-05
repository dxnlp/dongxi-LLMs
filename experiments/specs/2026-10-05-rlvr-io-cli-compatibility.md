# Existing RLVR CLI controls and the new explicit reader contract

Declared before editing the existing control scaffolds. The snapshot I/O
extension makes its complete contract mandatory at the actual CLI boundary.
Keep the old numerical/control questions intact while supplying that newly
required argument group through `tests/snapshot_io_test_support.py`.

Affected scaffolds are the local-random HF identity positive/encoding-alias
negative controls in `test_run_identity.py`, the authored execution-failure
evidence control in `test_grpo_capstone_labs.py`, and the no-peer FIFO and
change-after-bounded-cap-parse controls in `test_rlvr_work_budget.py`. Supply
explicit I/O metadata and the existing 16MiB teaching payload bound where the
CLI now needs them. A direct `execute_run` fixture may attach the same bounded
bootstrap fields that the real CLI would have parsed; it must still refuse its
deliberately changed model-work cap bytes before tokenizer/model allocation.

Do not change model geometry, seeds, training/evaluation data, original23 work
caps, numerical assertions/tolerances, expected negative outcomes or number of
updates. Do not count these unchanged old questions as new independent controls.
The helper's generous CPU-fixture I/O caps are explicit authored inputs, not
derived production recommendations, defaults or launch approval. Archive the
old work-test source already retained by the independent test owner; preserve
all earlier raw verification files and distinguish source revisions.

Use the existing isolated offline CPU environment, one thread and sampled25GiB
reserve. Record actual focused and full acceptance commands/footers in exclusive
new output directories. No Git mutation, pretrained acquisition, GPU job,
installation, service or platform containment change is authorized here.
