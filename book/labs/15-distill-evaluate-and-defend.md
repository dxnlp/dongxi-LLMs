# Lab 15 — Distill, Evaluate and Defend

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapters 7, 9–13 evaluation, supervised labels, reward errors and policy roles.
Deliverable: A complete comparison defense with checkpoint ancestry, output/cost evidence, failures and next question.

Read the [chapter](../chapters/15-distill-evaluate-and-defend.md) and use the
[worked solutions](../solutions/15-distill-evaluate-and-defend.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Temperature distillation](../../notebooks/day-26/01_temperature_distillation.ipynb) | Predict: Will omitting squared-temperature scaling preserve gradients? | Analytical/autograd gradients agree at temperatures one/two/four; teacher gradient is absent. | Reverse the teacher distribution or student initialization. | Three logits do not establish model compression or reasoning transfer. |
| [Selection and sampling](../../notebooks/day-26/02_selection_and_sampling.ipynb) | Predict: Must more oracle availability improve a biased ranker? | The ranker can prefer a wrong class; majority ties choose the earliest tied candidate. | Change score ordering in a copied simulator. | Independent categorical draws are not correlated LLM candidates. |
| [Complete-response distillation · extension](../../notebooks/day-26/03_response_level_distillation.ipynb) | Predict: Will a correct final answer certify its printed step? | Prompt logits are ignored; eight response/EOS targets are scored; fresh fixed-seed fit and raw IDs match retained records. | Admit a wrong-step/right-final well-formed response. | Eligibility, final accuracy and trace validity are separate. |
| [Critique and acceptance · extension](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb) | Predict: Can a format-valid revision harm a correct draft? | One FLIP changes the binary answer; two can restore it; replay retains 864 historical candidates and zero new model calls. | Cap the critique before revision or change max rounds. | Programmatic revision is not neural self-critique. |
| [Student-visited states · extension](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb) | Predict: Will changing both prefix sampling and KL identify either cause? | Same-state KL gradients detach teacher targets; bucket KL plus lost detail equals full KL; evaluation uses no teacher hints. | Match tail mass while changing its internal distribution. | Conditional KL ignores state occupancy and need not improve transfer. |
| [Genealogy and evaluation](../../notebooks/day-27/01_genealogy_and_frozen_panel.ipynb) | Predict: Will shared ancestry make different evaluation scores comparable? | Valid fixture graph passes; cycle, unknown parent and changed evaluation identity refuse. | Add a DPO sibling to the same SFT parent. | Fixture hashes name labels, not trained weight bytes. |
| [Claim defense](../../notebooks/day-28/01_defense_and_release_gate.ipynb) | Predict: Will passing CPU mechanics justify an unrun model-scale quality claim? | Illustrative missing-license/held-out checks keep readiness false; all-true fixture passes structurally. | Remove a real acceptance condition and bound the claim again. | Boolean readiness is not evidence or publication. |

The [actual-candidate](07-evaluation-is-a-contract.md) and [teacher-selection](08-instruction-data-as-an-interface.md) bridges stay in their original routes. Distillation adds teacher collection and student exposure; final-answer correctness, printed-step validity and faithful reasoning are distinct claims. Defend one real comparison using the [final campaign evidence](../../experiments/reports/2026-10-05-final-native-campaign-defense.md): keep random-init stories, Base→SFT→chosen/DPO, and acquired Instruct→RLVR as separate roots. The synthetic genealogy/release fixtures do not replace those records.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 26 27 28 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/evaluation_tools.md).

</details>
