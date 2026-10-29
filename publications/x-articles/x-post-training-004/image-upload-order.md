# Image Upload Order

Cover (upload separately, 5:2): `assets/cover-post-training-04.png`. All figures are English and shared by both versions.

1. `assets/01-one-score.gif` — DeepSeek-R1-Zero: a rule checks only the final answer, yet AIME 2024 pass@1 rose from 15.6% to 71.0% during RL (published). The token grid is schematic.
2. `assets/02-rl-loop.gif` — The RL loop: the environment gives a prompt, the model (the agent) writes an answer, the judge returns a score, and the score updates the policy.
3. `assets/03-state-action.gif` — The state is the prompt plus the tokens so far; the policy gives next-token probabilities; the sampled token is the action; the finished answer is a trajectory. Illustrative probabilities.
4. `assets/04-reward-return.gif` — The judge scores the finished answer: reward 0 on every token and 2 on the last. With γ = 1 the return at every position is 2.
5. `assets/05-value-q-advantage.gif` — Ten sampled answers land at their scores. All ten averaged give V = 4.3, each branch's average is its Q (2, 5, 9), and A = Q − V is −2.3, +0.7, +4.7. Illustrative numbers.
6. `assets/06-baseline.gif` — Measured from 0, every answer is pushed up. Measured from the baseline V = 4.3, only better-than-usual answers go up.
7. `assets/07-credit-assignment.gif` — V drops right after open, so the TD error of −2.3 puts the blame there. λ = 0 keeps it on open; λ = 1 spreads it to OpenAI and is.
8. `assets/08-policy-gradient.gif` — Two policy-gradient steps: the branches reshape (50/30/20% → 15/36/49% → 4/11/85%) and V rises from 4.3 to 8.3; closed turns from above to below average. Illustrative step size.
9. `assets/09-exploration.gif` — Greedy decoding only takes open: the other branches stay unknown, A = 0, and nothing changes. Sampling tries every branch and the policy moves.
10. `assets/10-ratio-clip.gif` — The batch was sampled from π_old. A = +4.7 pushes π(partly) from 20% to 24%; ρ = π / π_old reaches 1 + ε (ε = 0.2) and the push stops.
11. `assets/11-kl-penalty.gif` — A judge that likes flattery gives it 9.6. As the policy drifts from π_ref toward flattery, the judge's average rises to 9.54, β·KL to 2.38, and R − β·KL falls from 9.03 to 7.16 (β = 1, illustrative).
12. `assets/12-term-map.gif` — The whole loop on one picture: state, policy, action, reward, V, Q, A, and the update that reshapes the branches.
13. `assets/13-term-cards.gif` — Quick reference: every term in this article, grouped by step of the loop.
