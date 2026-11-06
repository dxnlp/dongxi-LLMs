# Image Upload Order

Cover: `assets/cover-post-training-dpo.png`. All figures are English, carry the DONGXI LABS
mark and are shared by both versions. File names keep their clip numbers: `00a` is image 1, `00b` is image 2, and each later file is two images later. `05-implicit-reward.gif` is KL GIF 12 re-rendered without its badge. All thirteen were
rendered 2026-11-06 from `visuals/animations/projects/post-training-dpo/` and copied into `assets/`. Candidate
IDs are in `visuals/animations/PROPOSALS.md`.

1. `assets/00a-two-routes.gif` — RLHF goes from human pairs through a reward model and RL to the chat model; DPO goes straight there with one loss on stored pairs. (`CAND-ANIM-079`)
2. `assets/00b-change-is-the-score.gif` — As the model's shares move away from the dashed reference, each answer's score moves with them: raised answers above 0, lowered answers below 0, unchanged answers 0. Illustrative numbers. (`CAND-ANIM-079`)
3. `assets/01-secretly-a-reward-model.gif` — Left, a separate reward model with a score head. Right, the trained model and the frozen reference read the same answer, and ratio, log and β give its score: 0.606 for the best answer after the small move. Illustrative probabilities. (`CAND-ANIM-067`)
4. `assets/02-three-signals.gif` — Reward and KL enter the update; gold only judges checkpoints. In DPO, stored pairs replace the reward model. (`CAND-ANIM-057`)
5. `assets/03-why-pay-for-distance.gif` — With reward alone, all share flows to the overrated answer: reward 3, gold −2. With β = 1 it gets 4.3%: reward 1.22, gold 1.00. Toy example. (`CAND-ANIM-052`, egg version)
6. `assets/04-no-move-helps.gif` — Five net-value bars level at 0.808, then the recipe: multiply by e^reward, add to Z = 2.243, divide. (`CAND-ANIM-069`, `053`)
7. `assets/05-implicit-reward.gif` — At the best model every log-ratio equals the reward minus 0.808. In a comparison the constant cancels: 0.731 either way. (`CAND-ANIM-066`, KL GIF 12)
8. `assets/06-loss-on-one-pair.gif` — The gap sweeps from −3 to 3 and the loss curve draws itself, with the egg pair marked at 0.693, 0.435 and 0.313. (`CAND-ANIM-070`)
9. `assets/07-the-push.gif` — A nudge test, then arrows sized by the push: 0.953, 0.500 and 0.047. One pair trained step by step, with steps that shrink and never stop. (`CAND-ANIM-071`)
10. `assets/08-where-probability-comes-from.gif` — One step moves only the pair's two logits; all five shares change. The wrong answer rises from 8.0% to 8.2%, and the bars differ from π*. (`CAND-ANIM-073`)
11. `assets/09-ideal-test.gif` — Who-won records arrive with the rewards hidden. Training on 1,000 of them moves the shares from the reference to the target, and the distance goes from 0.33 to 0.05. Simulated data; the bars are one run's shares. (`CAND-ANIM-074`)
12. `assets/10-answers-to-tokens.gif` — Answer tokens and the end-of-turn token add up; prompt tokens are not counted. A longer reply has a lower total, so each answer is compared with its own reference value. Illustrative values. (`CAND-ANIM-077`)
13. `assets/11-what-to-watch.gif` — Four things to watch: training and held-out accuracy from 50%, chosen and rejected probabilities, a run where both fall, and KL measured from samples. Invented curves, no measured values. (`CAND-ANIM-078`)
