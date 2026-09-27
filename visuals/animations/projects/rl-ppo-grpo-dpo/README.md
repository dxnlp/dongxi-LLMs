# PPO, GRPO and DPO: three ways to learn from feedback

A 104-second 1080p film. It explains three families of methods for training a
language model from feedback, each shown through one representative algorithm.

| Family | Example | Where the per-token learning signal comes from |
| --- | --- | --- |
| Online RL with a critic | PPO | a learned value model \(V_\phi\) gives every token its own baseline |
| Online policy gradient without a critic | GRPO | the mean reward of a group of sampled answers is the baseline |
| Direct preference optimization | DPO | a fixed chosen/rejected pair, scored by the log-ratio to a frozen reference |

**Learning objective.** After watching, a reader can say for each family what is
sampled, what scores it, how credit reaches individual tokens, and what the
update does. They can also say which models must be held in memory: four for
PPO, three for GRPO, two for DPO.

**Structure.**

- **0:00, shared goal:** \(\max_\theta \mathbb{E}[r(x,y)] - \beta\,\mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})\), then the three families.
- **0:17, PPO:** sample, then score (one reward per answer), then credit (critic trace, \(\delta_t\), \(\hat{A}_t\)), then the clipped update.
- **0:49, GRPO:** the critic is removed. A group of answers is scored, the group mean becomes the baseline, and one advantage covers every token of an answer.
- **1:09, DPO:** the sampling loop breaks. A fixed pair gets an implicit reward, a margin and a loss, and the push fades as the margin grows.
- **1:34, summary:** one goal, three ways to estimate the learning signal.

## Render

```bash
cd visuals/animations/projects/rl-ppo-grpo-dpo
MPLCONFIGDIR=$PWD/media/mpl uv run --project ../.. python render.py --preview  # 960x540 @ 15 fps -> media/preview/
MPLCONFIGDIR=$PWD/media/mpl uv run --project ../.. python render.py            # 1920x1080 @ 30 fps -> out/
```

`render.py` does the following:

1. Regenerates the fixture.
2. Renders `scene.py::FeedbackFamilies` with Manim, using the Cairo renderer.
3. Checks the scene's event timeline against the fixture.
4. Writes the contact sheets, the poster and `out/ppo-grpo-dpo-report.json`. The report holds the checks, the hashes, ffprobe data and the environment.

The MP4 is git-ignored; regenerate it with the command above.

| File | Role |
| --- | --- |
| `fixture.py` | Toy numbers (GAE, PPO clip, GRPO group advantages, DPO margin and loss) and 14 self-checks, including finite-difference gradients. Writes `fixture.json`. |
| `scene.py` | The film. |
| `kit.py` | Palette, text and equation helpers, model blocks and chips. |
| `mathsvg.py` | Matplotlib mathtext to SVG, so no LaTeX install is needed. |
| `render.py` | Render, validation, review frames and report. |

This project is self-contained. It does not use the shared animation style guide
or its helpers.

## Evidence boundary and simplifications

- Every number is an illustrative toy value from `fixture.py`, not a measurement from a trained model.
- **PPO:**
  - The toy uses \(\gamma = 1\), \(\lambda = 0.95\) and one sparse reward at the end, so \(\hat{A}_t \approx r - V_\phi(s_t)\), as shown.
  - RLHF implementations usually add a per-token KL penalty to the reward. The film shows KL as the leash to the frozen reference instead.
  - The ratio motion is a schematic path, not an optimizer trace: each ratio moves in the direction of its advantage until it reaches \(1 \pm \epsilon\).
  - Clipping removes the gradient only in the direction the advantage favours.
- **GRPO:**
  - Outcome supervision: every token of an answer shares the normalized group advantage.
  - The KL term is added to the loss, not to the reward.
  - Some variants drop the standard-deviation normalization. The fixture checks that this choice only rescales the advantages.
- **DPO:**
  - The toy update moves \(\hat{r}_w\) up and \(\hat{r}_l\) down symmetrically, with weight \(\sigma(-m)\). In practice both log-likelihoods can fall while the margin grows.
  - DPO has no explicit KL term. \(\beta\) sets the strength of the KL-regularized objective that DPO solves in closed form, so the film keeps the leash.

## Sources

- Schulman et al., 2017. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- Schulman et al., 2016. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438.
- Ouyang et al., 2022. *Training language models to follow instructions with human feedback.* arXiv:2203.02155.
- Shao et al., 2024. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:2402.03300.
- Rafailov et al., 2023. *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv:2305.18290.
