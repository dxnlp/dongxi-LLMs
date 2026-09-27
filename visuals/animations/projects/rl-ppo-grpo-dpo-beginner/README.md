# How chatbots learn from feedback: PPO, GRPO and DPO for beginners

A 2:46 film at 1920×1080 and 30 fps, for viewers without a mathematics background. It
tells the same story as the detailed film in
[`../rl-ppo-grpo-dpo/`](../rl-ppo-grpo-dpo/README.md), but with characters and everyday
metaphors instead of equations. For each method it answers three questions, which an
on-screen **What · Why · How** stepper tracks: what it is, why it is needed, and how
the learning signal is calculated. The calculations use only simple arithmetic.

**Learning objective.** After watching, a viewer can explain in plain words:

- A language model picks each next word by "spinning a wheel of chances", and training
  changes the slice sizes.
- Feedback needs a judge. A judge can be fooled, which is why every method keeps the
  model on a leash to its original version.
- **PPO** uses a coach to decide which word deserves the blame or credit, and takes
  small, fenced steps.
- **GRPO** replaces the coach with a comparison against the group's average.
- **DPO** skips the judge and the practice loop entirely, and tips a balance between a
  preferred and a rejected answer.

## Cast and metaphors

| On screen | Technical term |
| --- | --- |
| Blue student robot with a mortarboard | the model being trained (the policy) |
| Prize wheel of next-word chances | the next-token probability distribution |
| Violet judge robot with score cards | the reward model |
| Rambling answer that gets a 10 | reward hacking (here, a length bias) |
| Photo of the "original model" | the frozen reference model |
| Orange leash from the photo to the student | the KL penalty |
| Teal coach robot with a headset | the critic (value model) |
| Coach's guess line and each word's credit | value estimates and advantages (one-step TD errors) |
| Fence on the wheel | PPO clipping, $\epsilon = 0.2$ |
| Several tries, answer key, group average | GRPO's group of samples, verifier reward and group baseline |
| Eye test (lens 1 or lens 2?) and the saved pair card | offline preference data (chosen and rejected answers) |
| Balance scale, and the "original" vs "student" bars with their ×/÷ multipliers | DPO's implicit rewards and their margin |
| Push arrow | DPO's gradient weight $\sigma(-m)$ |
| Memory box | the models held in memory during training: 4, 3 or 2 |

## The arithmetic shown on screen

- **PPO credit:** the coach's guesses before each word are 6, 6, 6.5, 6.5, 7, 7, and
  then the judge gives 2. Each word's credit is how much it changed the guess:
  0, +0.5, 0, +0.5, 0, and 7 → 2 = −5 for "Sydney".
- **PPO fence:** 40% × 0.8 = 32%. One round moves the *picked* word's chance by at most 20%.
  The other words shift only because all the chances must still add up to 100%.
- **GRPO:** scores 1, 0, 1, 0. Group average (1 + 0 + 1 + 0) ÷ 4 = 0.5. Credit = score − average = ±0.5, shared by every word of a try.
- **DPO:** A becomes ×3 as likely as under the original model, and B becomes ÷3. So A leads
  3 × 3 = 9 to 1, and the push is 1 ÷ (1 + 9) = 10%, down from 1 ÷ (1 + 1) = 50% at the start.
  This is exactly $\sigma(-m) = 1/(1+e^{m})$ with $e^{m} = 9$.

**Structure** (times from the final render's event log):

| Time | Section | Content |
| --- | --- | --- |
| 0:00 | Title | — |
| 0:05 | The basics | Word by word, the wheel of chances, a confident mistake. |
| 0:22 | Learning from feedback | Judging is easier than writing (the soup), people compare, a judge learns their taste. The judge is fooled by rambling, so a photo of the original model is kept on a leash. |
| 0:51 | Three ways | Three recipe cards: PPO, GRPO and DPO. |
| 0:57 | PPO: practice with a coach | Judge's score, the coach's guess line, word credits, the fence, rounds of practice, 4 models. |
| 1:38 | GRPO: practice and compare | Four tries, the answer key, the group average, shared credit, 3 models. |
| 2:06 | DPO: learn from comparisons | Eye test, a saved pair, no judge or coach, tipping the balance, the fading push, a built-in leash, 2 models. |
| 2:37 | The big picture | Each method asks "better than what?" |

## Render

```bash
cd visuals/animations/projects/rl-ppo-grpo-dpo-beginner
MPLCONFIGDIR=$PWD/media/mpl uv run --project ../.. python render.py --check    # story + checks, no video
MPLCONFIGDIR=$PWD/media/mpl uv run --project ../.. python render.py --preview  # 960x540 @ 15 fps -> media/preview/
MPLCONFIGDIR=$PWD/media/mpl uv run --project ../.. python render.py            # 1920x1080 @ 30 fps -> out/
```

`--only ppo,dpo` renders just the named sections, for quick iteration. The full render does the following:

1. Regenerates the fixture.
2. Renders `scene.py::BeginnerFeedback` with Manim, using the Cairo renderer.
3. Checks the scene's event log against the fixture. For example, it checks that the
   credits equal the changes in the coach's guesses, that the fence stops at 32%, and
   that the DPO push equals 1 ÷ (1 + lead).
4. Writes the contact sheets, the poster and `out/ppo-grpo-dpo-beginner-report.json`.

Every event also asserts that no object has left the frame. The MP4 is git-ignored;
regenerate it with the commands above.

| File | Role |
| --- | --- |
| `fixture.py` | Every number the film shows, with 16 self-checks. Writes `fixture.json`. |
| `scene.py` | The film. |
| `kit.py` | Paper palette, text helper, robots, photo frame, tiles, wheel, balance and other props. |
| `render.py` | Render, validation, review frames and report. |

This project is self-contained. It does not use the shared animation style guide, its
helpers, or the detailed film's code.

## Evidence boundary and simplifications

- Every number is an illustrative toy value from `fixture.py`, not a measurement from a trained model.
- "Words" stand for tokens. The wheel shows one next-token distribution. Training
  changes the model's weights, and the slices change as a consequence.
- **The judge and the leash:**
  - The judge is a reward model trained on human comparisons.
  - Rambling for a higher score is one well-documented form of reward hacking (length bias).
  - The leash is the KL penalty to the frozen reference. PPO for RLHF usually adds it to the reward; GRPO adds it to the loss.
- **PPO:**
  - Word credit is shown as the one-step change in the coach's guess: the TD error, which is GAE with $\lambda = 0$. Real PPO usually blends several steps.
  - The fence bounds the picked token's probability ratio to $[0.8, 1.2]$. Strictly, clipping removes the incentive to move further; it is not a hard wall.
  - The toy steps always land exactly on the fence.
- **GRPO:**
  - Real GRPO also divides each credit by the group's standard deviation; the film says so on screen. Here the population standard deviation is 0.5, so the credits
    would become ±1 (about ±0.87 with the sample standard deviation). Some variants skip this step.
  - The judge can be a rule-based answer checker, as in the film. In that case, only two neural models need memory.
- **DPO:**
  - "How much more likely than the original" is the implicit reward $\beta \log(\pi/\pi_{\mathrm{ref}})$, with $\beta = 1$ in the toy.
  - The toy moves A up and B down symmetrically. In practice, both likelihoods can fall while the gap still grows; the balance shows only the gap, which is what DPO optimizes.
  - "No judge, no coach" also means DPO cannot explore new answers. It learns only from the pairs it is given.
  - The reference model's log-probabilities can even be precomputed.

## Sources

- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman et al., 2016. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback.* arXiv:2203.02155.
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.
- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290.
