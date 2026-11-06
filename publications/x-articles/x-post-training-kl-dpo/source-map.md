# Source map — DPO article

The public body has no links. Every factual claim maps to a source here. Paper facts were checked against
the primary PDFs on 2026-11-06 (arXiv text extracted with pypdf). All egg-example numbers are reproduced by
`check_numbers.py`. D-numbers point to the S2 note unless marked.

| Article claim | Source |
|---|---|
| Subtitle "Your Language Model is Secretly a Reward Model"; DPO introduced by Rafailov and colleagues at Stanford, 2023 | Rafailov, Sharma, Mitchell, Ermon, Manning & Finn (2023), arXiv:2305.18290v3, title page; NeurIPS 2023 stated on p. 1; §5.1 uses the same line as a heading. D89, D90 |
| Separate reward model with a score head; two-stage RLHF | X-POST-005; DPO paper §3 (linear layer on the final transformer layer; Eq. 2–3) |
| Implicit reward = β × log-ratio; "the reward implicitly defined by the language model" | DPO paper §4 (r̂ below the gradient); D82, D87 ★ |
| Shortcut GIF: scores at the small move and the best compromise; RLHF stages | D82, D87 ★; DPO paper §3; `check_numbers.py` §1 |
| Egg question, five answers, reference shares, rewards | D67 ★; illustrative |
| Pair at start / small move / best compromise: gaps 0, 0.606, 1; chances 0.5, 0.647, 0.731; losses 0.693, 0.435, 0.313 | D83 ★, D85; `check_numbers.py` §1 |
| KL recap, 0.047; scores 0.485 / 0.638 / −0.120 / 0.808 | KL article; D73, D75 |
| Three signals: reward and KL in the update, gold evaluation only | D61, D62 ★. Gold terminology: D60; Gao et al. (2022) as in the KL article |
| Sixth answer: 0.005, 2-minute answer 0.075, reward 3, gold −2 | D79 (authored toy) |
| Reward alone: all on the sixth answer, reward 3, gold −2; β = 1: 4.3%, best 37.9%, reward 1.22, gold 1.00; reference gold 0.48 | `check_numbers.py` §3. Egg version of D56 ★ |
| Reasons for the price: stay where the reward model is accurate, keep diversity, prevent mode collapse | DPO paper §3 after Eq. 3 ("prevents the model from deviating too far from the distribution on which the reward model is accurate, as well as maintaining the generation diversity and preventing mode-collapse") |
| The reference knows things the reward does not measure | D56 ★ (reason 2) |
| Moving anchor: ten steps of e^(0.4·R), each KL ≤ 0.059; total 3.61, wrong answer 69% | `check_numbers.py` §4. New in this article; D56 ★ states the idea. Ten steps equal the KL article's β = 0.25 end point |
| Reference usually the SFT model, frozen, no gradient | DPO paper §3 (π_ref = initial SFT model); D83 ★, D92 |
| Net value; 0.288 / 1 / 0.5 / 1.394 / −1; chip test +0.011; 0.808; −0.0004 | D75, D76 ★; `check_numbers.py` §5. The omitted shared term is the "+1" of D76 ★ |
| Solving for the recipe; multiply, add, divide; Z = 2.243; π\* | D76 ★; `check_numbers.py` §5 |
| "Partition function" name; Z sums over all answers | DPO paper §4, Eq. 4; D76 ★ |
| β dial 71% and 24% | D71; KL article |
| One prompt on paper; π\* as a target for shared weights | D77, D78 |
| Log-ratios = reward − 0.808; β log Z; σ(1) = 0.731 both ways (section "From the model back to the reward") | D82; `check_numbers.py` §6 |
| "We start with the same RL objective as prior work" | DPO paper §4 |
| The formula was known; prior work cited | DPO paper §4 (cites refs. 31, 30, 19, 15: Peters & Schaal 2007; Peng et al. 2019; Korbak et al. 2022; Go et al. 2023; D89) |
| Rearrange (Eq. 5), partition function cancels (Eq. 6), loss (Eq. 7) | DPO paper §4 |
| "solve the standard RLHF problem with only a simple classification loss" | DPO paper, abstract |
| Sources of the loss's pieces; no KL computed; nothing to invert with no price | D88 |
| Assumptions: Bradley–Terry, flexible model | DPO paper §5.1; D88, D97 |
| Row format; loss steps; loss values | DPO paper Eq. 7; D83 ★, D85; `check_numbers.py` §7 |
| Nudge test; slopes −σ(−m) | D91; `check_numbers.py` §8 |
| Weight matters; naive version can degenerate | DPO paper §1 and §4 ("a naïve version of this method without the weighting coefficient can cause the language model to degenerate", App. Table 3) |
| Gradient as a list of derivatives; one per weight | D94 |
| Steps with learning rate 1: 0.693 → 0.219 | D91; `check_numbers.py` §8 |
| Equal and opposite push on the two log-probabilities, scaled by β; reference receives none | D92 |
| Learning rate and β table | D93; `check_numbers.py` §8 |
| Adam removes most of β's speed effect | D93; S2 notebook Exercise 6 explanation |
| Default β = 0.1, lr 1e-6, batch 64 | DPO paper App. B; D95 |
| One softmax step: (0.408, 0.155, 0.153, 0.202, 0.082); gap 1 | D96; `check_numbers.py` §9 |
| Ideal test: setup; DPO sees winners only | D97; S2 notebook Part E |
| 1,000 pairs, seed 0: (0.189, 0.325, 0.128, 0.343, 0.014), distance 0.05; reference 0.33 | `check_numbers.py` §10 (notebook code). Distance is total variation. One simulated run |
| Mean distance over eight datasets: 0.38, 0.15, 0.04, 0.01 | `check_numbers.py` §10 (notebook Predict 2 seeds). Authored toy simulation, labelled as such in the body |
| Ten pairs: one run with vague and wrong at 0% | `check_numbers.py` §10 (first N = 10 dataset: 0, 0.329, 0.197, 0.474, 0) |
| Three epochs in Tülu 2 | Ivison et al. (2023), arXiv:2311.10702, §2: "for three epochs" |
| Gap 1 in reward units for every β; log-ratio gap 1/β; KL 0.02, 0.33, 2.12 | `check_numbers.py` §10, exact from π\*. D98 ★ reports the fitted version (1.065 on 3,000 pairs) |
| Reply log-prob = sum of token log-probs; answer and end-of-turn tokens; prompt excluded | KL article; D98 ★; S2 notebook Part F |
| −40 and −60 for 40 and 60 tokens | Illustrative; D98 ★ gives ≈ −40 for 40 tokens |
| Four forward passes | Follows from Eq. 7 |
| Length-normalized DPO in Tülu 3, reduces length bias | Lambert et al. (2024), arXiv:2411.15124v5, §5.1, Eq. 6 |
| Loss starts at log 2, accuracy 50% | D98 ★; arithmetic (all gaps 0) |
| Training accuracy ahead of held-out | D98 ★ (shape only; our run's numbers are kept out of the body by series convention) |
| Constructed case: 6%, 2.5%, 58.1%, 21.8%, 11.6%; gap 1.609; loss 0.182 | `check_numbers.py` §12. Egg version of book Chapter 11 §11.4 |
| Likelihood displacement; *No* over *Never* raises *Yes*; Llama-3-8B-Instruct refusal rate 74.4% → 33.4%; similar embeddings | Razin et al. (2024), ICLR 2025, arXiv:2410.08847v4, abstract |
| Low learning rate 5e-7 required for stable training | Ivison et al. (2023), §2 |
| KL not computed or limited in training; measure from samples | D88; KL article |
| Pair accuracy is not gold; judge generations | D63, D79 |
| GPT-4 as judge; over-optimization left open | DPO paper §6 and "Limitations & Future Work" in its discussion |
| Offline: learns only from given pairs | Lambert et al. (2024) §5.1; main note D32–D34 |
| Tülu 2 DPO at 70 billion parameters; Tülu 3 stages SFT, DPO, RLVR | Ivison et al. (2023), abstract and §1; Lambert et al. (2024), §2.3 stages 2–4 |

## Evidence boundary

- **Published:** the DPO paper, Razin et al. and Tülu statements above.
- **Exact arithmetic on illustrative numbers:** every share, reward, gold value, log-ratio, loss and score.
- **Simulation on the authored toy, labelled in the body:** the ideal test. GIF 09's bars are the run's shares at 16 training steps (`check_numbers.py`, last section); the distance dips to 0.01 at step 25 and settles at 0.05.
- **Invented, labelled in the figure:** the curves of GIF 11; the token values and totals of GIF 10.
- **Left out of the public body by series convention:** the S2 notebook's Qwen3 results (D98 ★). §12 uses their shape only. `CAND-ANIM-078`'s storyboard quotes those numbers; GIF 11 uses invented curves instead, as recommended to the learner on 2026-11-06.
