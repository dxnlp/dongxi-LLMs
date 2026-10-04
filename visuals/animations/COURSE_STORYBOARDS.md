# Course Animation Storyboards and Portable Media Backlog

Captured automatically during the 2026-10-04full-course build. These are
**candidates and detailed production briefs**, not commissioned or rendered films.
Production belongs on Mac Studio after explicit approval. Preserve the existing
approved media and stable task IDs in [the proposal inbox](PROPOSALS.md).

## Course-wide coverage

| Chapter | Mathematical or architectural motion opportunity | Existing or new packet |
|---:|---|---|
| 1 | identity changes → acceptance criteria → claim boundary | Experiment-identity storyboard below |
| 2 | UTF-8 bytes → learned merges → IDs; tied embedding gradient branches | Existing BPE and embedding packets |
| 3 | logits → stable softmax → target NLL → p−q → update | Existing CE, decoding and next-token packets |
| 4 | Q/K/V projections → causal scores → value mixture → backward; growing KV | Existing attention/KV packets |
| 5 | residual additions, norm, gating, RoPE pairs, GQA reuse and recurrence | Existing architecture candidates012–017 and looped-depth packet |
| 6 | valid targets versus positions; accumulation → AdamW → recoverable state | Existing pretraining candidates018–021 |
| 7 | paired examples → sampling uncertainty → aggregate/slice contradiction | CAND-ANIM-022 |
| 8–9 | role serialization → masks → shift → answer loss → prompt gradients | CAND-ANIM-023 |
| 10 | reward margin → Bradley–Terry probability; nuisance feature counterexample | CAND-ANIM-024 |
| 11 | constrained optimum → reference ratios → DPO margin → failed held-out inference | CAND-ANIM-025 |
| 12 | sampled trajectory → reward → score-function gradient; baseline/RLOO/PPO | CAND-ANIM-026 |
| 13 | grouped responses → rewards → relative advantages → clipped token update | CAND-ANIM-027 |
| 14 | proxy optimization → true-task decline; policy versions and rollout age | CAND-ANIM-028 |
| 15 | softened teacher → KL and T² → student; selection under token budget | CAND-ANIM-029 |

Do not duplicate an earlier packet merely because its source chapter now has
more prose. Reuse tensor identities and visual grammar from the static reference
figures; do not turn illustrative animation timing into a throughput benchmark.

## Shared production contract

For each approved film, retain an editable mathematical implementation, an
event/state ledger, a source manifest and a narrative script. Verify numerical
events independently before rendering. Show one continuous example and persistent
object identities rather than a montage of disconnected equations. Meaning
must survive grayscale and paused frames. Use precise labels for observations,
synthetic fixtures and proposed workloads.

Deliverables, if approved: a readable16:9MP4, a compact loop orGIF where useful,
a static poster/fallback, captions/alt text and the return packet. Duration is a
production decision, not an automatic scope commitment. Never upload or publish
from the existence of this backlog.

## Experiment identity: what changed?

Source: Chapter 1 and Day 1notebook01. Begin with one ledger holding corpus
revision, tokenizer, model code, seed, environment and measurement contract.
Change the corpus revision while retaining a flattering loss number; the identity
hash changes and the comparison becomes an explicitly different experiment.
Then distinguish exit0, finite gradients, reserve safety and capability evidence.

Acceptance: canonical key order does not change the hash; a control mutation
does. The film must never imply that a successful smoke test proves story quality.
Machine: Mac. Status: candidate refinement of the existing evidence lesson.

## CAND-ANIM-022 — A paired comparison has several clocks

Source: [Chapter 7](../../book/chapters/07-evaluation-is-a-contract.md),
Day 10 notebooks and `evaluation_lab.py`.

Storyboard:

1. Keep identical prompt IDs under modelsAandB; reveal outcome pairs.
2. Separate concordant from discordant examples and show the paired difference.
3. Resample pairs together, not independent model rows; uncertainty moves while
   the frozen panel identity remains fixed.
4. Split the same rows by slice; an aggregate gain conceals a weaker slice.
5. Change a tokenizer or target weighting contract and visibly invalidate the
   old NLL comparison.

Acceptance: paired bootstrap indexes are shared; score denominators are visible;
a Wilson interval for a single proportion is not mislabeled an interval for a
paired difference. No causal attribution or population guarantee is inferred.
Status: proposed. Useful article angle: “Better according to which contract?”

## CAND-ANIM-023 — A zero-loss prompt can still learn

Source: [Chapters8–9](../../book/chapters/09-supervised-fine-tuning.md),
Days11–14 and `instruction_data_lab.py`/`sft_lab.py`.

Storyboard:

1. Convert a role-tagged conversation into the exact rendered token sequence.
2. Show separate attention and supervision masks; zero prompt loss does not
   remove prompt visibility.
3. Shift logits and labels once. Highlight supervised answer tokens and the
   valid-target denominator.
4. Trace one answer error backwards through the assistant state and its causal
   read into earlier prompt embeddings.
5. Compare unequal microbatch counts: naive mean-of-means changes the gradient;
   summing losses then dividing by total valid targets restores the reference.
6. Insert a new packed document; document-isolation closes cross-record edges.

Acceptance: target alignment, EOS treatment, mask indexing and accumulation must
match the actual notebook tensors. No arrows from future tokens in the forward
pass. Do not claim an embedding with zero local loss has zero upstream gradient.
Status: proposed. Article angle: “The prompt is not graded, but it is trained.”

## CAND-ANIM-024 — Preference is a margin, not a thermometer

Source: [Chapter 10](../../book/chapters/10-preferences-and-reward-models.md),
Days15–16 and `reward_model_lab.py`.

Storyboard:

1. Place two answer scores on one axis and map their difference through sigmoid.
2. Shift both scores equally; pair probability remains fixed.
3. Overlay repeated disagreements as a soft target, then show gradient
   `predicted_pair_probability − observed_preference_frequency`.
4. Add response length as a shortcut. A new adversarial slice reverses the
   correlation while the original training score still looks attractive.

Acceptance: gradient signs agree with autograd; reward scores are not probability
of correctness; absolute offset is non-identifiable from within-prompt pairs.
Calibration applies to the declared preference event, not all model capabilities.
Status: proposed. Article angle: “A reward model can learn what you measured,
not what you meant.”

## CAND-ANIM-025 — What DPO compares

Source: [Chapter 11](../../book/chapters/11-direct-preference-optimization.md),
Days17–18, `dpo_lab.py` and the measured preference-policy report.

Storyboard:

1. Accumulate response-token log-probabilities, excluding prompt/padding.
2. Pair chosen and rejected likelihoods for both current and frozen reference.
3. Subtract reference-relative changes and scale by beta; pass the margin
   into negative log-sigmoid.
4. Backpropagate: frozen reference stays still; chosen/rejected policy paths move.
5. Reuse the actual tiny-run curves: training margin improves while held-out
   chosen-answer probability declines. Do not visually turn this negative result
   into a capability win.

Acceptance: sum sequence likelihoods, not mean token likelihoods; chosen/rejected
share the prompt; beta has the declared convention; ref gradients are absent.
The closed-form KL optimum is not proof of success for finite-data neural DPO.
Status: proposed. Article angle: “Improved preference margin is not the same as
improved chosen-answer probability.”

## CAND-ANIM-026 — A reward becomes a gradient

Source: [Chapter 12](../../book/chapters/12-language-generation-as-a-policy.md),
Days19–21 and `policy_gradient_lab.py`.

Storyboard:

1. Enumerate a tiny action policy and its exact expected reward.
2. Sample actions and attach the score-function gradient weighted by reward.
3. Subtract a detached, action-independent baseline; expected gradient stays
   fixed while scatter changes.
4. Build RLOO from other group members. Contrast it with inclusive mean baseline
   and its explicit finite-group scaling.
5. Follow positive and negative advantages through PPO's two branches. Clipping
   blocks selected surrogate incentives, not every probability or KL change.
6. Separate current, behavior and reference policies using three fixed identities.

Acceptance: compare estimates with exact enumeration; do not call detached
action-dependent baselines automatically unbiased. Token ratios are not exact
trajectory importance ratios. Show the sampled-KL value and gradient distinction.
Status: proposed. Article angle: “Why clipping is not a trust-region guarantee.”

## CAND-ANIM-027 — Relative reward, shared context

Source: [Chapter 13](../../book/chapters/13-group-relative-policy-optimization.md),
Days22–23, `grpo_lab.py`, actual tiny decoder run.

Storyboard:

1. From one prompt sample G complete autoregressive responses with EOS masks.
2. Apply a strict verifier including malformed and truncated outputs.
3. Compute group mean and population standard deviation; center and scale rewards.
4. Collapse a constant-reward group: every task advantage becomes zero, even
   though a KL regularizer may still update the policy.
5. Broadcast each sequence advantage only to its response token positions.
6. Show the declared response-mean clipped objective and exact categorical KL.
7. Contrast measured G4/G8 token budgets and the unchanged failed held-out result.

Acceptance: population standard deviation and epsilon match this implementation;
this is one explicit GRPO variant, not every algorithm with that name. EOS is
scored, padding is not. More samples are not depicted as guaranteed reasoning.
Status: proposed. Article angle: “What a reward group cannot tell the optimizer.”

## CAND-ANIM-028 — The optimizer follows the proxy

Source: [Chapter 14](../../book/chapters/14-when-optimization-goes-wrong.md),
Days24–25 and `optimization_diagnostics_lab.py`.

Storyboard:

1. Give a policy three responses: correct, incorrect and proxy-exploiting.
2. Optimize the biased proxy while graphing independent true success alongside.
3. Correct the verifier/reward and rerun the same initial policy, keeping both
   traces visible.
4. In a second panel attach policy-version IDs to queued rollouts; an old response
   cannot become fresh by relabeling it.
5. Show rollout, verification and learning as pipeline stages with schematic
   rates; mark the bottleneck without claiming local measured throughput.

Acceptance: true/proxy metrics use the same underlying action outcomes;
failure labels are hypotheses until alternatives are tested. Freshness is a
version contract, not the wall-clock age alone. Queue traces are labeled synthetic.
Status: proposed. Article angle: “A healthy training loop can optimize a broken goal.”

## CAND-ANIM-029 — Transfer a distribution, not just its winner

Source: [Chapter 15](../../book/chapters/15-distill-evaluate-and-defend.md),
Day 26, `distillation_lab.py` and measured CPU reference.

Storyboard:

1. Soften teacher and student distributions at one declared temperature.
2. Move the student's logits under teacher-to-student KL, freezing the teacher.
3. Compare gradients with and without T²; connect high-temperature scaling to
   the implementation, without claiming exact equality at every finite T.
4. Sample N answers and compare majority voting, a biased selector and the
   oracle “any correct” diagnostic. Show token cost increasing with N.
5. End with genealogy arrows and one frozen evaluation panel; unknown result
   rows stay blank/unexecuted rather than being replaced by animation outcomes.

Acceptance: teacher probabilities detached; KL direction explicit; temperature
at training versus generation distinguished; schematic tokens/sample not labeled
measured model latency. Selection requires an adequate verifier/judge.
Status: proposed. Article angle: “Best-of-N depends on the selector, not just N.”
