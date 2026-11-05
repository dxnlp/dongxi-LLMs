# Source map — KL article

The public body has no links. Every factual claim maps to a source here. Paper facts were checked
against the primary PDFs on 2026-11-04 unless marked *secondary*. All egg-example numbers are reproduced
by `check_numbers.py`.

| Article claim | Source |
|---|---|
| Egg question, five answers, reference shares, rewards, small move | S2 note D67 ★; illustrative |
| Ratio, log, weight, sum → 0.047; symbols of the formula; always-best 2.12; reverse 0.042 | S2 note D68 ★; `check_numbers.py` |
| Kullback and Leibler, 1951 | Kullback & Leibler, "On Information and Sufficiency", *Annals of Mathematical Statistics* 22 (1951), 79–86. *Secondary* for the wording: Wikipedia, "Kullback–Leibler divergence": mean information for discrimination; Kullback motivated it as an expected log-likelihood ratio |
| KL as average evidence per reply; evidence adds on the log scale; 64 replies to 20-to-1; 8.33-to-1 | Expected log-likelihood ratio (above); arithmetic in `check_numbers.py`. New in this article, not in the learning notes |
| Never negative; each term ≥ its gain; log x ≤ x − 1 | S2 note D68 ★, D69, D70 ★; Gibbs' inequality; book Chapter 3 |
| Extreme drop −3.689 × 0.01 = −0.037, KL 0.701 | D70 ★ |
| Other measures −1.569 and −0.126; the reference-weighted version is −KL(π_ref‖π), so negative for every change | D72 ★; identity by algebra |
| Reward, score, J values 0.485 / 0.638 / 0.209 / −0.120 | D67 ★, D70 ★ |
| Best answer's share 1.00 → 0.71 → 0.40 → 0.24 → 0.16 → 0.12 | D71 |
| Reward model reliable near its training data | X-POST-005; D56 ★ |
| Sixth answer: reference 0.005, 2-minute answer 0.075, reward 3, gold −2 | D79 (authored toy) |
| Same 0.10 move: ratios 1.4 and 21, KL 0.031 and 0.233, 7.4 times | `check_numbers.py`. Egg-example version of D57 ★ (three-answer toy: 0.023, 0.374) |
| KL not symmetric; dropping 0.102 vs infinite; inventing infinite in the RLHF order | D68 ★, D57 ★; arithmetic |
| Softmax probabilities positive; a small share has a high price | Book Chapter 11 §11.1 |
| Cross-entropy = entropy + KL(data‖model) | Day 3 note; book Chapter 3 |
| Sampled KL 0.096 / 0.061 / 0.048 | `check_numbers.py` (NumPy seed 0). One simulated run. Differs from the values in D68 ★ (0.055, 0.049), which came from another draw |
| Samples must come from the new model at its own settings | D35 ★; book Chapter 12 §12.8 |
| Reply log-ratio is the sum of token log-ratios; toy 0.223 + 0 + 0.182 = 0.405 | Chain rule; D35 ★, D57 ★; illustrative probabilities |
| InstructGPT: per-token KL penalty from the SFT model in the reward; β = 0.02 | Ouyang et al. (2022), arXiv:2203.02155v1, §3.5 and eq. 2 (p. 9); App. C.4 (p. 42) |
| GRPO adds the KL to the loss; estimator π_ref/π − log(π_ref/π) − 1, guaranteed positive; KL coefficient 0.04; PPO's per-token penalty in the reward as the standard approach | Shao et al. (2024), DeepSeekMath, arXiv:2402.03300, §4.1 (pp. 13–15) |
| Checks: 0 at step 0; negative batch mean as a bug; units | D59 |
| Gold signal independent of training | D60, D61 |
| Toy run: reference reward 0.505, gold 0.48; peak gold 1.10 at KL 0.81, reward 1.57; end (β = 0.25) KL 3.61, reward 2.67, gold −0.76, wrong answer 69% | `check_numbers.py`. Corrects D79: its row at KL 0.74 (gold 1.10) is near the peak, the exact peak is at KL 0.81; its first row (0.51, 0.49) is at large finite β, the reference itself has gold 0.48 |
| Gold rises, peaks and falls while the proxy keeps rising; KL distance is an inadequate metric for the quantity of over-optimization | Gao, Schulman & Hilton (2022), arXiv:2210.10760v1, abstract, Fig. 1 and p. 7 |
| Best compromise 0.178 / 0.303 / 0.110 / 0.395 / 0.013; log-ratios = R − 0.808; σ(1) = 0.731 both ways | D67 ★, D82 |
| The constant is β·log Z, a sum over all replies | D76 ★, D82 |
| DPO: β × log-ratio as the score in the Bradley–Terry loss; 0.693 → 0.313 | D83 ★; Rafailov et al. (2023), arXiv:2305.18290 (*secondary* here; to be checked for the KL/DPO article) |

## Evidence boundary

- **Published:** the InstructGPT, DeepSeekMath and Gao et al. statements above.
- **Exact arithmetic on illustrative numbers:** every share, reward, gold value, log-ratio and score.
- **Simulated:** the sampled KL values (one seeded run).
- **Not reported:** no results from our own notebooks or runs (series rule, X-POST-005).
