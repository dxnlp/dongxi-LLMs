# Lab 13 — Group-Relative Policy Optimization

Machine: isolated CPU course kernel on Mac or Spark, offline.
Expected time: 25–45 minutes per notebook; longer routes span several sessions (planning estimate).
Prerequisites: Chapter 12 behavior/old/reference roles; detached advantages and token masks.
Deliverable: An annotated rollout group plus verifier counterexample and a bounded interpretation of held-out results.

Read the [chapter](../chapters/13-group-relative-policy-optimization.md) and use the
[worked solutions](../solutions/13-group-relative-policy-optimization.md) after attempting each exercise.
Write each prediction before executing the reference. Numerical checkpoints use
the notebook’s declared fixture, seed, dtype and tolerance; changing inputs may
change the result. References and interpretations remain adjacent to the attempt.

| Notebook | Question / prediction before reveal | Checkpoint on the declared fixture | One intervention | Evidence boundary |
|---|---|---|---|---|
| [Group centering](../../notebooks/day-22/01_group_advantages.ipynb) | Predict: Will all-success and all-failure groups both supply reward gradient? | Both constant groups have zero relative advantage; the mixed group has both signs; estimator is detached. | Compare population/sample deviation or near-constant rewards. | Zero relative signal says nothing about absolute success. |
| [Ratios and KL](../../notebooks/day-22/02_token_ratios_kl.ipynb) | Predict: Will a sampled KL value identity establish its full gradient? | Exact forward-KL autograd matches its analytical gradient; positive/negative advantages flatten at the favorable upper/lower boundary. | Switch advantage sign or clipping epsilon. | Token surrogates are not trajectory importance ratios. |
| [Weighting and filtering · extension](../../notebooks/day-22/03_objective_weighting_and_filtering.ipynb) | Predict: Can filtering change a gradient while the policy remains fixed? | All retained policy hashes stay unchanged; analytical errors are below 1e-12; attempts exceed selections. | Change reduction/filtering on the same rollout pool. | Discarded attempts still count toward collection cost. |
| [Tiny decoder RLVR](../../notebooks/day-23/01_decoder_rlvr.ipynb) | Predict: Will doubling group size imply an equal-cost update comparison? | Both arms use twelve updates and seed 2223; retained response-token counts and every held-out row are shown. | Compare groups of four/eight with the same frozen task. | Warm-started symbolic performance is not pretrained transfer. |
| [Strict verifier](../../notebooks/day-23/02_verifier_contract.ipynb) | Predict: Will a plausible numeral with missing EOS satisfy the token task? | -35 is accepted for -35; ambiguous/full-width numerals and missing EOS refuse. | Try extra candidates or omit the ending marker. | Verified format is not proof of reasoning faithfulness. |
| [Positive controls · extension](../../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb) | Predict: Will higher average path probability improve every solvable item? | All declared seeds/updates survive; every positive control improves while failures and frozen initial weights remain. | Compare stopping/support faults and unknown lookup keys. | Tiny sequence learning and English-panel instruments have distinct evidence. |

For the actual native case, retain constant-group reward signal, KL pressure, generation caps and failed/absent responses. [The final campaign defense](../../experiments/reports/2026-10-05-final-native-campaign-defense.md) distinguishes held-out correctness from pooled diagnostics. Old policy produces the recorded rollouts; the reference supplies the anchor; collected targets and applied targets have separate denominators. [Operational exercises 22–25](../solutions/13-group-relative-policy-optimization.md#evidence-reading-extensions) retain their IDs and mappings.

Retain a short explanation of the intervention, observed change and claim it
does not establish. The deliverable should connect these explanations into one
defensible argument, with source/report links rather than an execution-only checklist.

<details>
<summary>Fresh CPU verification and operational reference</summary>

From the repository root, use the isolated environment/kernel described in
[Appendix D](../appendices/d-reproduction-and-environments.md):

```bash
.venv-course/bin/python scripts/verify_course_notebooks.py --days 22 23 --kernel dongxi-course --expected-prefix .venv-course
```

The verifier retains executed copies and identities in a new directory. Source
notebooks and learner attempts remain intact. Exact runner/replay/export commands
and their evidence boundaries are in the [runbook](../../docs/runbooks/rlvr_runner.md).

</details>
