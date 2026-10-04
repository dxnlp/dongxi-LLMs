# The 28-Day Learning Route

The complete material build was requested on2026-10-04. This route organizes
practice; the book's15 chapters organize the argument. All future sessions are
prepared ahead of live study under that explicit request. Readiness does not
change the learner's current Day 9 position or imply pretrained-model outcomes.

Use the chapter, the linked session index, its adjacent references, the worked
solutions and the experiment record together. A day's deliverable is an argument
supported by an intervention, not merely a notebook that ran.

| Day | Focus / chapter | Deep question | Concrete output |
|---:|---|---|---|
| 1 | Laboratory and evidence · Ch1 · [sessions](../notebooks/day-01/README.md) | What does a zero exit code prove? | Reconstruct a run identity and defend its acceptance criteria. |
| 2 | Tokenization and embeddings · Ch2 · [sessions](../notebooks/day-02/README.md) | Can an unseen character be representable but poorly compressed? | Trace bytes, merges, IDs and embedding gradient paths. |
| 3 | Next-token learning · Ch3 · [sessions](../notebooks/day-03/README.md) | How can one-hot observations teach an uncertain distribution? | Verify stable softmax, alignment and a known target distribution. |
| 4 | Causal attention · Ch4 · [sessions](../notebooks/day-04/README.md) | Which source can influence this prediction? | Verify Q/K/V, masks, gradient paths and cached equivalence. |
| 5 | Decoder components · Ch5 · [sessions](../notebooks/day-05/README.md) | How do several learned branches cooperate in one loss? | Assemble the decoder and inspect one-batch learning. |
| 6 | Modern architecture · Ch5 · [sessions](../notebooks/day-06/README.md) | Which design costs stored parameters, activation memory or compute? | Compare RMSNorm/RoPE/SwiGLU/GQA and account for costs. |
| 7 | Architecture defense · Ch5 · [sessions](../notebooks/day-07/README.md) | What makes a fair architectural comparison? | Trace tensors, explain trade-offs and defend an intervention. |
| 8 | Pretraining system · Ch6 · [sessions](../notebooks/day-08/README.md) | What state makes the next update reproducible? | Audit budgets, accumulation, AdamW and checkpoint recovery. |
| 9 | Run diagnosis · Ch6 · [sessions](../notebooks/day-09/README.md) | Can lower prediction loss coexist with inconsistent stories? | Read saved GPU evidence; verify packing boundaries and define the next comparison. |
| 10 | Evaluation · Ch7 · [sessions](../notebooks/day-10/README.md) | What exactly would make one model better? | Freeze metrics/splits; inspect uncertainty, slices and contamination. |
| 11 | Instruction data · Ch8 · [sessions](../notebooks/day-11/README.md) | Where does assistant supervision actually begin? | Trace template tokens, masks, document visibility and mixture exposure. |
| 12 | SFT mathematics · Ch9 · [sessions](../notebooks/day-12/README.md) | Can ignored prompt positions still learn from the answer? | Inspect loss gradients and valid-target accumulation. |
| 13 | Base-to-assistant experiment · Ch9 · [sessions](../notebooks/day-13/README.md) | Which behavioral changes survive a held-out evaluation? | Train the tiny assistant; prepare the bounded pretrained-model protocol. |
| 14 | SFT recipe defense · Ch9 · [sessions](../notebooks/day-14/README.md) | What is matched when full SFT and LoRA are compared? | Compare actual tiny adaptations and defend budgets/evaluation. |
| 15 | Pairwise preference · Ch10 · [sessions](../notebooks/day-15/README.md) | Which parts of a reward score can comparisons identify? | Derive and verify Bradley–Terry probabilities and gradients. |
| 16 | Reward models · Ch10 · [sessions](../notebooks/day-16/README.md) | Can a calibrated-looking score rely on the wrong feature? | Fit a reward model and test adversarial/length slices. |
| 17 | DPO derivation · Ch11 · [sessions](../notebooks/day-17/README.md) | Why does a fixed reference appear in a preference loss? | Derive KL-regularized optimum and inspect pair log-ratios. |
| 18 | DPO experiment · Ch11 · [sessions](../notebooks/day-18/README.md) | Can a better margin coexist with worse chosen-answer probability? | Train a tiny sequence policy and compare independent metrics/controls. |
| 19 | Policy gradients · Ch12 · [sessions](../notebooks/day-19/README.md) | How does a sampled token receive a sequence reward? | Compare exact enumeration, autograd and score-function estimates. |
| 20 | Baselines and PPO · Ch12 · [sessions](../notebooks/day-20/README.md) | What does clipping or a baseline really guarantee? | Inspect RLOO independence, estimator variance, ratios and KL. |
| 21 | Algorithm defense · Ch12 · [sessions](../notebooks/day-21/README.md) | Are equal rollout budgets equal optimization budgets? | Compare small policies with explicit sample/update clocks. |
| 22 | GRPO derivation · Ch13 · [sessions](../notebooks/day-22/README.md) | What information disappears in a constant-reward group? | Build group advantages, masks and clipped objectives. |
| 23 | RLVR experiment · Ch13 · [sessions](../notebooks/day-23/README.md) | What behavior does the verifier actually reward? | Run tiny autoregressive RLVR; inspect group-size costs and held-out outcomes. |
| 24 | Optimization failures · Ch14 · [sessions](../notebooks/day-24/README.md) | Can proxy reward rise while task success falls? | Create reward hacking and test a repair with independent metrics. |
| 25 | Rollout systems · Ch14 · [sessions](../notebooks/day-25/README.md) | When does a response belong to the wrong policy version? | Audit freshness, synchronization, memory and pipeline bottlenecks. |
| 26 | Distillation and selection · Ch15 · [sessions](../notebooks/day-26/README.md) | Who benefits from extra samples and at what token cost? | Compare distillation, majority voting, best-of-N and rejection. |
| 27 | Capstone evaluation · Ch15 · [sessions](../notebooks/day-27/README.md) | Which checkpoint gains survive a common evaluation contract? | Audit genealogy, model cards, paired errors and incomplete evidence. |
| 28 | Defense and release · Ch15 · [sessions](../notebooks/day-28/README.md) | Can another reader reconstruct and challenge the final claim? | Execute the release audit and defend decisions against failure cases. |

## Machine lane

All mechanism notebooks and saved-evidence analysis can run on CPU on Mac or
Spark. Mac is the default discussion/visual lane; Spark supplies separately
bounded model-scale execution. Days9,13–14,18 and23 have Spark experiment paths,
while their small references remain available on CPU. Days25–28 analyze costs
and evidence before deciding whether another GPU run is needed.

At the workload boundary, use **Switch to Spark** / **Continue on Spark**, or
**Switch to Mac** / **Continue on Mac**. The handoff preserves source, state and
pending questions. A directory index does not authorize launching its model-scale
candidate. Use the actual specification and resource profile for that decision.

## Completion and review

A day reaches the mastery rubric only after the learner can explain the mechanism,
interpret its perturbation and defend the evidence. A chapter's material can be
ready earlier. Preserve unanswered earlier practice as a review backlog rather
than automatically rewinding the lesson.

The final comparison uses a common frozen evaluation and complete checkpoint
genealogy. If a planned model branch has not run, the capstone reports it as
unexecuted. CPU demonstration outcomes remain bounded to their original fixtures.
No release decision can turn a proposed result into a measurement.
