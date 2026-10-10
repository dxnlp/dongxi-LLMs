# Dongxi LLMs — Reader's Guide

The complete teaching draft contains **15 chapters across 28 learning days**.
Read [the preface](front-matter/preface.md), [how to use the course](front-matter/how-to-use.md)
and [notation](front-matter/notation.md) first. The [day-by-day route](../docs/COURSE_SEQUENCE.md)
connects prose, visual notebooks, experiments and a defensible daily artifact.

The argument follows three connected problems: turn text into trainable
predictions, turn fluent generation into a useful assistant interface, and turn
judgments into policy improvements that survive evaluation. Worked examples
carry these problems through the equations. Predictions and controlled changes
help you explain the mechanism before judging a model result.

## Chapters and worked solutions

| Chapter | Learning days | Canonical chapter | Worked solutions |
|---:|---|---|---|
| 1 | 1 | [Evidence Before Optimization](chapters/01-evidence-before-optimization.md) | [Solutions](solutions/01-evidence-before-optimization.md) |
| 2 | 2 | [Text, Tokens, and Embeddings](chapters/02-text-tokens-and-embeddings.md) | [Solutions](solutions/02-text-tokens-and-embeddings.md) |
| 3 | 3 | [Learning the Next Token](chapters/03-learning-the-next-token.md) | [Solutions](solutions/03-learning-the-next-token.md) |
| 4 | 4 | [Attention and the Causal Information Boundary](chapters/04-attention-and-the-causal-information-boundary.md) | [Solutions](solutions/04-attention-and-the-causal-information-boundary.md) |
| 5 | 5–7 | [Building a Modern Decoder](chapters/05-building-a-modern-decoder.md) | [Solutions](solutions/05-decoder-notebook-solutions.md) |
| 6 | 8–9 | [Pretraining as a Controlled System](chapters/06-pretraining-as-a-controlled-system.md) | [Solutions](solutions/06-pretraining-as-a-controlled-system.md) |
| 7 | 10 | [Evaluation Is a Contract](chapters/07-evaluation-is-a-contract.md) | [Solutions](solutions/07-evaluation-is-a-contract.md) |
| 8 | 11 | [Instruction Data as an Interface](chapters/08-instruction-data-as-an-interface.md) | [Solutions](solutions/08-instruction-data-as-an-interface.md) |
| 9 | 12–14 | [Supervised Fine-Tuning](chapters/09-supervised-fine-tuning.md) | [Solutions](solutions/09-supervised-fine-tuning.md) |
| 10 | 15–16 | [Preferences and Reward Models](chapters/10-preferences-and-reward-models.md) | [Solutions](solutions/10-preferences-and-reward-models.md) |
| 11 | 17–18 | [Direct Preference Optimization](chapters/11-direct-preference-optimization.md) | [Solutions](solutions/11-direct-preference-optimization.md) |
| 12 | 19–21 | [Language Generation as a Policy](chapters/12-language-generation-as-a-policy.md) | [Solutions](solutions/12-language-generation-as-a-policy.md) |
| 13 | 22–23 | [Group-Relative Policy Optimization](chapters/13-group-relative-policy-optimization.md) | [Solutions](solutions/13-group-relative-policy-optimization.md) |
| 14 | 24–25 | [When Optimization Goes Wrong](chapters/14-when-optimization-goes-wrong.md) | [Solutions](solutions/14-when-optimization-goes-wrong.md) |
| 15 | 26–28 | [Distill, Evaluate, and Defend](chapters/15-distill-evaluate-and-defend.md) | [Solutions](solutions/15-distill-evaluate-and-defend.md) |

## How the companions fit

The chapter supplies the argument; the [76 visual notebooks](../notebooks/README.md)
make its mechanisms inspectable. Each exercise has adjacent runnable reference
answers, plots and interpretation boundaries. Reusable logic lives in `src/dongxi_llms/`,
not in copied notebook fragments. [Companion labs](labs/) provide reproducible routes
through the later experimental sections. Specifications, measured reports and
model/data cards live under `experiments/`.

Numerical teaching experiments are bounded CPU references. Separate Spark
evidence includes the original TinyStories run, a matched 400-update story
comparison, Qwen3 full/LoRA SFT, chosen-only/DPO comparisons, and bounded
reasoning/RLVR runs. The [experiment matrix](../docs/EXPERIMENT_MATRIX.md)
distinguishes these measured outcomes from prepared protocols and unrun stages.
An original arithmetic fixture is a mechanism microscope, not a broad assistant
benchmark; optimizer execution does not imply reward-driven learning.
Material preparation does not advance the learner automatically beyond Day 9
or establish mastery.

## Appendices

- [A — Laboratory setup](appendices/a-laboratory-setup.md)
- [B — Mathematical and tensor notation](appendices/b-mathematical-and-tensor-notation.md)
- [C — Evaluation and experiment templates](appendices/c-evaluation-and-experiment-templates.md)
- [D — Reproduction and environments](appendices/d-reproduction-and-environments.md)

## Quality and publication boundaries

The [blueprint](../docs/COURSE_BLUEPRINT.md) defines teaching depth, prerequisites
and evidence levels. [Release review](../docs/RELEASE_CHECKLIST.md) keeps material
readiness separate from model-quality claims, licensing and permission to publish.
Animation candidates have portable storyboards; production remains approval-gated
on Mac Studio. These do not count as rendered films.
