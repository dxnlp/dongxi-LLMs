# Lab 7 — Evaluate before changing the model

The CPU notebooks use declared fixture scores to expose the measurement mechanism. They are runnable on Mac or Spark with the course PyTorch/Matplotlib environment. No model checkpoint, external dataset, API key or GPU is needed.

1. Open [metrics and contracts](../../notebooks/day-10/01_metrics_and_contracts.ipynb). Explain each accepted/rejected string, derive the pass@3 subset count, and change the normalization policy only as a named new contract.
2. Open [paired uncertainty](../../notebooks/day-10/02_paired_uncertainty.ipynb). Compare all-success intervals at different sample sizes, inspect the paired-difference distribution, then reproduce the deliberate duplicate-item failure.
3. Open [slices and contamination](../../notebooks/day-10/03_slices_and_contamination.ipynb). Find the rare regression, detect the planted overlap, and preserve a repair record.

The reusable implementation is [evaluation_lab.py](../../src/dongxi_llms/evaluation_lab.py). Its tests compare pass@k against exact combinatorial counts and check interval/normalization boundaries. Run:

~~~bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_evaluation_sft_labs.py -v
~~~

To move from fixture to model evidence, freeze the original instruction evaluation generated in Lab 9, its task/format slices, and the exact decoding rule. Development contains 60 original examples across 20 value groups; publication test contains 120 examples across 40 separate groups. Baseline and candidate must share item identities, serialization, parser and attempt budget. The provided training runner scores a declared eight-ID greedy development grid and full development answer NLL; it does not substitute for the publication suite.

For stories, keep a separate original prompt population and the five-dimensional rubric from Chapter 7. A copy/extraction evaluation cannot establish story coherence. Compare checkpoint outputs blind and retain each reviewer's raw decisions.

Deliver an evaluation card with capability, suite hash, checkpoint, template, decoding, verifier, denominators, uncertainty unit, error slices, selection history and limitations. Completion of this lab requires an explained instrument and audit, not a favorable score.
