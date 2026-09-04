# Interactive Notebook Curriculum

This document maps the book's conceptual argument to executable learning
sessions. Every chapter receives a notebook pathway; notebooks are not optional
demonstrations added after the prose is finished.

The purpose is not to maximize notebook count. Each session should isolate one
important mathematical, architectural, optimization, data, or evaluation
mechanism that becomes substantially clearer when the learner can inspect and
change it.

## Chapter contract

Each chapter normally contains three focused notebook sessions:

1. **Mechanism microscope** — derive and implement the smallest transparent
   version; expose shapes, intermediate values, and invariants.
2. **Perturbation and failure** — violate one assumption, compare a broken
   variant, or change a controlled input; diagnose the consequence.
3. **Integration and evidence** — connect the mechanism to a small model,
   dataset, or evaluation contract; separate observation from interpretation
   and state what the result cannot prove.

A chapter may use two sessions when the mechanism is compact or four when an
important architecture and its experiment need separation. Sessions should be
small enough to complete interactively. Several focused notebooks are preferred
to one long notebook that mixes unrelated ideas.

Every exercise or prediction checkpoint is immediately followed by a clearly
labeled, runnable reference solution and a concise explanation. The learner
attempts the checkpoint first, but routine syntax lookup must not interrupt the
conceptual discussion.

Reusable computation belongs in `src/dongxi_llms/`. Notebook claims remain
exploratory until important results are reproduced by tests and, when empirical
claims matter, an experiment specification and report.

## Planned pathway by chapter

| Chapter | Mechanism microscope | Perturbation and failure | Integration and evidence |
|---:|---|---|---|
| 1. Evidence Before Optimization | Build a run identity and claim ledger | Change one uncontrolled variable and expose an invalid comparison | Reconstruct a smoke-test claim from manifest, metrics, and exit status |
| 2. Text, Tokens, and Embeddings | Trace Unicode → bytes → BPE merges → IDs | Compare multilingual compression and unknown/byte fallback behavior | Trace embedding lookup, contextualization, and both embedding-gradient paths |
| 3. Learning the Next Token | Logits → stable softmax → NLL and $p-q$ | Break causal shifting and loss-mask normalization | Learn a known conditional distribution from repeated one-hot targets |
| 4. Attention and the Causal Information Boundary | Build scaled causal attention and verify its invariants | Break mask placement and score scaling; inspect gradient paths | Prove cached and uncached decoding equivalence for an unchanged prefix |
| 5. Building a Modern Decoder | Assemble residual attention and feed-forward blocks | Compare normalization, position, activation, and attention variants | Build `DongxiGPT`, account for parameters/FLOPs/memory, then test optional recurrent depth |
| 6. Pretraining as a Controlled System | Trace batches, token loss, AdamW, schedules, and clipping | Trigger instability, overflow, or incorrect accumulation safely | Resume a bounded run and reconcile checkpoints, metrics, throughput, and memory |
| 7. Evaluation Is a Contract | Implement metrics, frozen splits, and uncertainty | Reveal sampling variance, leakage, and misleading aggregate scores | Compare fixed checkpoints with slices and qualitative error analysis |
| 8. Instruction Data as an Interface | Serialize roles and trace labels/loss masks | Break chat templates, packing boundaries, or assistant masking | Inspect a provenance-aware data mixture and its effective token weights |
| 9. Supervised Fine-Tuning | Trace SFT loss and gradient flow through one batch | Ablate masks, mixture weights, or adaptation choices | Compare base and SFT checkpoints for gains, regressions, and uncertainty |
| 10. Preferences and Reward Models | Derive Bradley–Terry probabilities and reward gradients | Explore disagreement, position bias, and reward miscalibration | Train and evaluate a tiny reward model with adversarial slices |
| 11. Direct Preference Optimization | Compute chosen/rejected sequence log-probabilities and derive DPO | Perturb $\beta$, references, masks, and length treatment | Run a controlled DPO comparison against SFT and frozen evaluation |
| 12. Language Generation as a Policy | Derive REINFORCE and inspect token/sequence credit | Compare no baseline, learned baselines, RLOO, and PPO clipping | Measure variance, KL, entropy, and reward under a small policy update |
| 13. Group-Relative Policy Optimization | Construct grouped rollouts, relative advantages, and GRPO loss | Expose zero-variance groups, verifier errors, and token-weighting choices | Run a small RLVR update and inspect reward, KL, entropy, and outputs |
| 14. When Optimization Goes Wrong | Simulate reward hacking, entropy collapse, and length bias | Break rollout freshness, synchronization, or monitoring assumptions | Diagnose a stored failure from telemetry without selecting only favorable evidence |
| 15. Distill, Evaluate, and Defend | Compare sampling, self-consistency, best-of-$N$, and distillation targets | Expose selection bias and capability regressions | Reconstruct checkpoint genealogy and defend the final comparison under uncertainty |

## Activation and maintenance

- At the start of a chapter, refine its three entries into named notebook
  sessions and add a local `README.md` under the relevant notebook directory.
- Create notebooks only as the corresponding learning material becomes active;
  do not commit empty placeholder notebooks for later chapters.
- Link finished sessions from the chapter, worked solutions, day artifact, and
  `notebooks/README.md`.
- Execute every reference path in the declared environment before marking a
  session ready.
- Preserve learner-entered predictions and outputs carefully; do not overwrite
  them during course maintenance without explicit permission.

## Day 4 activation

Day 4 activates three sessions under `notebooks/day-04/`:

1. `01_causal_attention_forward.ipynb` — score, scale, mask, softmax, value
   mixture, and prefix-invariance checks;
2. `02_attention_gradients_and_failures.ipynb` — autograd through Q/K/V,
   routing versus value branches, missing scaling, and incorrect mask placement;
3. `03_kv_cache_equivalence.ipynb` — prefill/decode state growth, cached versus
   uncached equality, request-local lifecycle, and the compute-memory trade.

Their detailed learning sequence is maintained in `notebooks/day-04/README.md`.
