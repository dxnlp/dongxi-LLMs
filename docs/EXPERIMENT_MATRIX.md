# Experiment Pathways and Evidence Matrix

CPU experiments inspect exact mechanisms and tiny models on authored fixtures.
Spark protocols address pretrained compatibility and behavior under separately
approved budgets. The completed TinyStories run remains actual GPU evidence,
not a new run in this build.

## Controlled questions

| Chapter | Intervention | Executable route | Evidence boundary |
|---:|---|---|---|
| 1 | identity and failed smoke criteria | [Lab route](../book/labs/01-evidence-before-optimization.md) | CPU claim audit; historical Qwen smoke |
| 2 | BPE corpus/merges and tied gradients | [Lab route](../book/labs/02-text-tokens-and-embeddings.md) | Educational encoding and tensor invariants |
| 3 | stable softmax, alignment and known target distribution | [Lab route](../book/labs/03-learning-the-next-token.md) | Measured tiny distribution; not language quality |
| 4 | causal masks, routing/content gradients and cache equivalence | [Lab route](../book/labs/04-attention-and-the-causal-information-boundary.md) | CPU invariants; not serving throughput |
| 5 | decoder components, costs and shared-block recurrence | [Lab route](../book/labs/05-building-a-modern-decoder.md) | Shape/gradient accounting; trained recurrence pending |
| 6 | valid targets, restart, masking and story diagnosis | [Lab route](../book/labs/06-pretraining-as-a-controlled-system.md) | Actual TinyStories run; coherence unresolved |
| 7 | frozen metrics, paired uncertainty, adversarial slices | [Lab route](../book/labs/07-evaluation-is-a-contract.md) | Fixture instruments; real model suite still required |
| 8 | templates, masks, packing and mixture exposure | [Lab route](../book/labs/08-instruction-data-as-an-interface.md) | Authored English interface data |
| 9 | full versus LoRA, accumulation and held-out behavior | [Lab route](../book/labs/09-supervised-fine-tuning.md) | Measured tiny copy task; Qwen route unexecuted |
| 10 | reward fitting and nuisance feature inversion | [Lab route](../book/labs/10-preferences-and-reward-models.md) | Authored preferences and adversarial slices |
| 11 | DPO versus chosen SFT and label flip | [Lab route](../book/labs/11-direct-preference-optimization.md) | Negative tiny-decoder outcome; Qwen route unexecuted |
| 12 | exact versus sampled gradients, baselines/RLOO/PPO | [Lab route](../book/labs/12-language-generation-as-a-policy.md) | Enumerated and procedural policies |
| 13 | group normalization, verifier errors and G4/G8 RLVR | [Lab route](../book/labs/13-group-relative-policy-optimization.md) | Real tiny autoregressive update; held-out failure retained |
| 14 | proxy/true divergence, repair and rollout versions | [Lab route](../book/labs/14-when-optimization-goes-wrong.md) | Measured proxy policy; labeled systems schematics |
| 15 | distillation KL/T², selection costs and genealogy | [Lab route](../book/labs/15-distill-evaluate-and-defend.md) | Tiny distillation/selection; Qwen capstone unexecuted |

Chapter6 also has a [saved-run analysis](../book/labs/06-reading-a-pretraining-run.md).
Every route links code, focused notebook sessions and interpretation questions.

## Reproduce bounded CPU references

From the repository root with the declared interpreter/kernel:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests
python scripts/verify_course_notebooks.py --kernel dongxi-course --export-figures
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode full
PYTHONPATH=src OMP_NUM_THREADS=1 python -m dongxi_llms.sft_lab --steps 100 --mode lora
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_preference_policy_cpu.py
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_grpo_capstone_experiments.py
```

These perform actual bounded computation. Experiment runners print measured JSON
rather than rewriting historical reports. The verifier saves new executed copies,
hashes, figures and failures, preserving source notebooks. Specs precede measurements.

Spark's verified interpreter is `/home/dongxi/dgx-spark-dongxi/.venv/bin/python`;
its kernel is `dgx-spark-native`. The portable environment setup is in
[AppendixD](../book/appendices/d-reproduction-and-environments.md).
Reading the course never launches an installer, model download or server.

## Pretrained checkpoint chain

1. Generate original instruction data and freeze source-group splits.
2. Profile Chapter9 base-model full SFT or LoRA under its explicit smoke cap.
3. Use the full HF policy and its saved audited template. A LoRA adapter must be
   explicitly merged with its exact pinned base and recorded as a derived checkpoint.
4. Chapter11 DPO requires full or merged weights, the parent's saved template,
   a frozen reference and independent evaluation. An adapter directory alone is
   not silently compatible.
5. Chapter13's optional RLVR runner validates local template/genealogy, performs
   grouped autoregressive updates and scores a fixed initial/final held-out panel.
   That small arithmetic pilot is not broad reasoning evidence.
6. Chapter15 compares actually obtained checkpoints under a capability-appropriate
   frozen contract. Unexecuted branches remain unexecuted; fixture outcomes are
   never relabeled pretrained-model results.

Inspect implemented entry points without loading weights:

```bash
PYTHONPATH=src python scripts/run_chapter09_spark_sft.py --help
PYTHONPATH=src python scripts/run_chapter11_spark_dpo.py --help
PYTHONPATH=src python -m dongxi_llms.qwen_rlvr_lab --help
```

Revision metadata, local-file hashes and parent genealogy establish distinct
parts of identity. Resource guards are sampled between operations; a blocking
kernel can overrun the sampling interval. A hard deadline requires an external
supervisor. Future learning-material creation does not approve a GPU campaign.

## Measured reports

- [Evaluation/SFT](../experiments/reports/2026-10-04-evaluation-and-sft-course.md)
- [Reward/DPO/policy](../experiments/reports/2026-10-04-preference-policy-cpu.md)
- [GRPO/failure/distillation](../experiments/reports/2026-10-04-grpo-diagnostics-distillation.md)

Each preserves negative observations and limits. Later integration revisions
are identified by the final course-build manifest; a historical hash remains the
identity of its own earlier revision. Complete material can coexist with deferred
capability or public-release claims.
