# Experiments that connect the mechanisms

These seven original companions extend the existing fifteen-chapter route.
They turn isolated mechanisms into the six [course assignments](../../../docs/COURSE_ASSIGNMENTS.md).
Start with the prerequisites registered in the [course manifest](../../../docs/course_manifest.json);
the post-training integration uses later policy material even though its SFT
connection belongs in Chapter 9. Its placement is a return visit, not a new
required prerequisite for first reading SFT.

Each notebook presents a prediction, a checkpoint, runnable reference reasoning,
a controlled failure and data-backed plots. Source notebooks retain their cells;
fresh execution produces separate output copies. All seven are CPU/offline
references. Actual Spark measurements have their own environment and protocol.

| Companion | Book connection | Experiment and reference | Boundary |
|---|---|---|---|
| [Resource accounting](01_resource_accounting.ipynb) | Chapters 5–6; Assignment C | Shape/byte ledgers, arithmetic intensity, simulated roofline, timed matrix multiplication, existing decoder checkpoint audit and decode costs. [Answers](SYSTEMS_SOLUTIONS.md). | Modeled traffic and retained storage are distinct from hardware traffic and peak memory. |
| [Online attention](02_online_attention.ipynb) | Chapter 4; Assignment C | Running maximum/denominator/numerator, omitted rescale, independent dense/SDPA outputs and gradients, masks and uneven tiles. [Answers](SYSTEMS_SOLUTIONS.md). | Ordinary autograd; no fused backward or GPU speed claim. |
| [Scaling prediction](03_scaling_prediction.ipynb) | Chapter 6; Assignment D | Frozen simulation prediction, bootstrap with failed fits, contract faults and the complete tiny empirical matrix. [Answers](SYSTEMS_SOLUTIONS.md). | Simulated laws and measured finite-corpus results are separately labeled; an unbracketed optimum is a refusal. |
| [Reward-signal audit](04_reward_signal_audit.ipynb) | Chapters 7, 10, 13–14; Assignment E | Replay retained native text, adversarial checker probes, exact finite-group gradients/variance and changed collection populations. [Answers](../../../book/solutions/cs336-posttraining-assignment.md). | Historical strict rewards remain unchanged; a richer diagnostic is a separate instrument. |
| [Tokenizer contracts](05_tokenizer_contracts.ipynb) | Chapters 2–3; Assignment A | Segmentation, reserved-marker admission, framing, save/load identity, Unicode and byte-normalized likelihood. [Answers](05_tokenizer_contracts-solutions.md). | Buffered chunks are one framed record; the reference does not promise constant-memory streaming. |
| [Corpus curation](06_corpus_curation.ipynb) | Chapters 6–8; Assignment B | HTML extraction, reason-coded filters, exact/near duplicates, verified components, source exposure and three-seed matched-target training. [Answers](06_corpus_curation-solutions.md). | Original bounded fixture; cleaner data does not guarantee better measured loss. |
| [Post-training integration](07_posttraining_integration.ipynb) | Chapters 8–14; Assignment E | Compatible SFT parent, coverage/repetition, accepted-response SFT and two GRPO reductions, source-level comparisons and constant baselines. [Answers](../../../book/solutions/cs336-posttraining-assignment.md). | Conditional symbolic rollout grammar; usable gradients do not establish English reasoning or parity generalization. |

Use the isolated CPU environment and temporary kernel described in
[Appendix D](../../../book/appendices/d-reproduction-and-environments.md).
For example, the existing verifier accepts explicit paths:

```bash
python scripts/verify_course_notebooks.py --notebooks notebooks/extensions/cs336/02_online_attention.ipynb --kernel dongxi-course --expected-prefix .venv-course
```

The [implementation record](../../../experiments/reports/2026-10-11-cs336-upgrade.md)
links the frozen protocols, raw negative results, execution identities and
independent reviews. It distinguishes reference readiness from learner practice.
