# Source map — X-POST-005

The public body has no links. Every factual claim maps to a source here. Paper facts were checked
against the primary text on 2026-10-30. Rows marked *secondary* were checked through papers that cite
them.

| Article claim | Source |
|---|---|
| Previous article: the judge gave *OpenAI is open. It's in the name.* 2 and *partly* answers 8 and 10; a reward model was named as the topic of the next article; the flattering answer as reward hacking | `../x-post-training-004/x-editor-draft-body-with-image-placeholders.md` |
| Figure 1 (reused): ten sampled answers to *Is OpenAI open?* at the judge's scores; *open* answers low, *partly* 10 and 8 | X-POST-004 GIF 05 (`visuals/animations/projects/post-training-rl-terms/rendered/05-value-q-advantage/05-value-q-advantage.mp4`), cut at 8.15 s before the V line, retitled "The judge's scores", DONGXI LABS mark added; illustrative numbers from that article |
| gpt-oss-120b and gpt-oss-20b are open-weight models under Apache 2.0; flagship models through API and apps | Hugging Face model cards `openai/gpt-oss-120b` and `openai/gpt-oss-20b`: "License: apache-2.0" (checked 2026-10-30; the repositories also include a usage policy). Stated as of the time of writing. |
| Preference judgments need an opinion about the result, not knowledge of how to produce it; forms: pair, best of N, ranking (N choose 2 pairs); sources: annotators, Reddit/StackExchange votes, LLM judges (UltraFeedback with GPT-4) | SLP3 (Jurafsky & Martin, draft of 2026-08-19), Chapter 8, §8.3 and §8.3.1 |
| InstructGPT labelers ranked 4 to 9 answers per prompt | Ouyang et al. (2022), arXiv:2203.02155v1, §3.5 (p. 8) |
| InstructGPT labelers agreed on about 73% of comparisons | Same, §3.4 (p. 8): training labelers 72.6 ± 1.5%, held-out labelers 77.3 ± 1.3% |
| Bradley–Terry (1952); `P = σ(z_i − z_j)`; the gap is the log-odds; loss `−log σ(r_w − r_l)` as binary cross-entropy; reward model from a pretrained LLM with the LM head replaced by one linear layer; best-of-N and data selection as uses | SLP3 Chapter 8, §8.3.2–8.3.3, eq. 8.2–8.4, Fig. 8.10 |
| Gap → probability values (0.12, 0.50, 0.73, 0.95); loss values (1.31, 0.69, 0.31, 0.05); 7 of 10 → gap log(7/3) ≈ 0.85 | Arithmetic. Learning note D4, D9, D41, D43 (`learning_artifacts/day-15-preference-data-and-bradley-terry/learning-from-preferences.md`) |
| Elo uses the same logistic curve on rating gaps | Learning note D4, D41 (Elo: base 10, scale 400). General knowledge, stated qualitatively. |
| Assumptions: one scale (no cycles), no ties, one standard | Learning note D41, D49 |
| Gradient descent; `∇L = −(1 − σ(gap)) · Δf`; two-step example (0.693 → 0.313, push 0.5 → 0.269) | Arithmetic. Learning note D52 (features recast to the running example) |
| Qwen3-0.6B: hidden size 1,024; 28 layers; 151,936 LM-head rows; about 596 million parameters | `../x-token-001/source-map.md` (config of snapshot c1899de…) |
| The head swap: LM head removed, a 1,024 → 1 score head added and randomly initialized; no instruction prompt; no generation | SLP3 §8.3.3 and Fig. 8.10; Hugging Face `AutoModelForSequenceClassification` with `num_labels=1` (learning note D30, D42) |
| Every logit is `h` · one head row; toy 4-number example (2.0, 1.0, −1.0; scores 2.0 and 1.0) | Arithmetic. Learning note D48 (labels recast: Part, Yes, No) |
| Score at every position; the last real token is used; causal attention; one-token change alters only later positions; padding | Hugging Face sequence-classification pooling (last non-pad token); learning note D28, D55. Token facts: Qwen3 tokenizer, both answers 8 tokens, ` Some` 4329 vs ` All` 2009 (checked 2026-10-30). |
| Bradley–Terry is a two-way softmax; softmax values 0.705/0.259/0.035 and 0.731/0.269; same loss `−log softmax(correct option)` | Arithmetic. Learning note D47 |
| Plackett–Luce generalizes to rankings and reduces to Bradley–Terry for two items | *Secondary:* PlackettLuce R package paper, arXiv:1810.12068 |
| Training with backpropagation and AdamW on all weights; supervised, no RL | Learning note D42, D52; SLP3 §8.3.3 (gradient descent on eq. 8.4) |
| InstructGPT trained the reward model for one epoch; more epochs overfit | Ouyang et al., App. C.2 (p. 41) and footnote 5 (p. 8) |
| InstructGPT reward model about 70% on held-out labelers | Same, §4.1 (p. 12): 72.4 ± 0.4% on training labelers, 69.6 ± 0.9% on held-out labeler groups |
| Llama 2 reward models 63% and 65% on their own test sets; 81% and 94% on clearly separable pairs | Touvron et al. (2023), arXiv:2307.09288v2, Tables 7–8 (p. 12): Helpfulness RM 63.2 on Meta Helpful; Safety RM 64.5 on Meta Safety; on "significantly better" pairs 80.7 and 94.3 |
| Random noise vs systematic bias: order kept and gaps shrink vs bias learned; more data does not remove bias | Learning note D3, D50, D53 (mechanism; the article quotes no simulation values) |
| In two of the settings studied, most of the reward gained during RLHF came from longer answers | Singhal et al., "A Long Way to Go: Investigating Length Correlations in RLHF", arXiv:2310.03716v2, §3.1: 70–90% of the reward improvement on WebGPT and RLCD explained by length |
| Humans and reward models sometimes prefer convincingly written sycophantic answers to correct ones | Sharma et al., "Towards Understanding Sycophancy in Language Models", arXiv:2310.13548v4, abstract |
| Overoptimization: the proxy reward keeps rising; the gold reward rises, peaks and falls; the gold is a larger reward model standing in for humans | Gao, Schulman & Hilton (2022), arXiv:2210.10760v1, abstract, §1, Fig. 1 |
| Llama 2 margin loss, larger for clearly better pairs | Touvron et al., §3.2.2, eq. 2 (p. 11) |
| Ties: Rao–Kupper (1967), Davidson (1970) | *Secondary:* arXiv:1909.07123, arXiv:2409.17431 |
| HelpSteer2: 0–4 ratings on helpfulness, correctness, coherence, complexity, verbosity; regression | Wang et al. (2024), arXiv:2406.08673v1, §2.1–2.2, §3 |
| ArmoRM: several ratings combined by a learned gating network | Wang et al. (2024), arXiv:2406.12845v1, abstract |
| PairRM (LLM-Blender) reads both candidates together | Jiang et al. (2023), arXiv:2306.02561v3, abstract (the paper's PairRanker; PairRM is the released checkpoint) |
| General preference models can represent non-transitive preferences; Nash learning from human feedback | Munos et al. (2023), arXiv:2312.00886v4, §3.1 |
| Generative judges keep the LM head (verdict text or token probability) | Zhang et al. (2024), "Generative Verifiers", arXiv:2408.15240v3; Mahan et al. (2024), arXiv:2410.12832 (*secondary* for the abstract) |
| Process reward models; PRM800K step-level labels | Lightman et al. (2023), arXiv:2305.20050v1, abstract (800,000 step-level labels) |
| Same-family base: Llama 2 initialized its reward models from chat-model checkpoints, so that an information mismatch would not favour hallucinations | Touvron et al., §3.2.2 (pp. 10–11) |
| Llama 2: separate helpfulness and safety reward models | Same, §3.2.2 (p. 10) |
| A larger reward model is harder to exploit | Gao et al.: the overoptimization coefficients scale smoothly with reward-model size, and larger reward models reach higher gold-reward peaks. Stated qualitatively. |
| InstructGPT: one 6B reward model for policies of every size; 175B reward-model training unstable | Ouyang et al., §3.5 (p. 8) and App. C.2 (p. 41) |
| Llama 2 collected new comparisons on the latest chat model in each round (RLHF-V1 to V5) | Touvron et al., §3.2.1 (p. 10), §3.2.3 (p. 12) |
| Uses: RLHF grader with a KL penalty; labeling new pairs; best-of-n; data selection | SLP3 §8.3.3 and §8.4 (eq. 8.6–8.7); learning note D8, D13, D40 |

Rows for the margin, ties, ratings, multi-objective, pairwise preference, generative judges, process
reward models, verifiers, same-family base, size and refresh rounds supported the section *Beyond
Bradley–Terry, and which base model*, removed at the learner's request (2026-10-30); kept for reuse.

## Evidence boundary

- **Published:** the InstructGPT, Llama 2, Singhal, Sharma and Gao figures and statements above.
- **Exact arithmetic on illustrative numbers:** every score, probability, loss, gradient step and
  toy vector.
- **Real token facts:** the token facts for the one-word pair come from the Qwen3 tokenizer.
- **Not reported:** no results from our own notebooks or runs, at the learner's request
  (2026-10-30).
