# Preference and Policy CPU Mechanisms — Predeclared Specification

Mode: bounded mechanism/learning; authored synthetic fixtures; CPU only.
Created before the recorded evidence run. Reusable code: `reward_model_lab.py`,
`dpo_lab.py`, `policy_gradient_lab.py`. Execute the canonical runner from the
course root with an existing PyTorch environment, `OMP_NUM_THREADS=1`.

## Questions and hypotheses

1. Bradley–Terry gradients match probability minus soft target; an equal score
   offset changes neither loss nor gradients. Aggregated ambiguous votes have
   a finite optimum and nonzero residual entropy.
2. Confounding quality with length/format permits nuisance coefficients; a
   balanced fixture improves independent ranking/probability behavior on a
   distribution where nuisance features vary independently.
3. Finite DPO learns reference-relative pair preferences; flipping labels
   changes direction. Sequence DPO can improve pair margins without increasing
   absolute held-out desired-answer probability. No particular direction of
   that independent metric is assumed for the tiny decoder.
4. REINFORCE/autograd/exact finite gradients agree; action-independent baselines
   preserve expectation, RLOO is unbiased under iid sampling, and inclusive
   mean centering scales the gradient by `(G-1)/G`.
5. PPO clipping flattens only the already-favorable excessive movement. `k1`
   and `k3` share exact forward-KL expectation on common positive support;
   `k2` is generally biased. Variance ordering is measured, not guaranteed.

## Frozen controls

- Reward: 256train and256independent validation differences; seeds1516/1616;
  three float64 coordinates; label `sigmoid(2*quality)`; 240plain gradient steps,
  lr.2,L2coefficient.002. Confounded arm has nuisance=quality+.04noise;
  balanced arm has independent features. Adversary `[-1,3,3]` is declared here.
- Categorical DPO: two contexts/three answers; explicit reward/reference arrays
  in source; all three unordered pairs; exact soft labels; beta.5;240steps,lr1.
  Controls chosen-SFT and flipped-label DPO start from the same reference.
- Sequence DPO: original TinyDecoder12IDs,width16,one block,2heads,maxlength8;
  seed1718; six prompts, A/B plusEOS; held-out promptID8;80AdamWsteps,lr.008,
  decay.01,clip1,beta.5. Chosen-SFT control shares initialization and updates.
- Gradient/variance fixture: logits `[.3,-.2,.1]`, rewards `[0,1,3]`; all
  individual actions and all groupsG3 enumerated. No Monte Carlo tolerance.
- Policy comparison: six `(a,b)` prompts,a0..2,b0..1; three answers; reward1
  exactly when action=(a+b)%3; independent prompt logits;160rolloutupdates;
  G4 completions/prompt; seeds1921,1922,1923;lr1.5. REINFORCE,exact-value
  baseline,RLOO each1gradientstep/rollout;PPO3epochs,epsilon.2,exactoldvalue.
  Report sampled-completion and gradient-pass budgets separately.
- KL fixture:p`[.2,.3,.5]`,q`[.7,.2,.1]`; strictly positive normalizedsupport.

## Acceptance and failure

Objective/mask/expectation tests must pass with float64 close agreement, every
loss/metric must be finite, the reference must retain identical weights and no
gradients, and runtime must remain a bounded CPU exercise. Record all declared
arms/seeds, including regressions. Failure of a prediction is evidence to retain,
not a reason to select another seed. No language capability, human calibration,
hardware scalability, or unseen arithmetic generalization follows from these
fixtures. Large-model Spark DPO is separately proposed in the Chapter11lab and
must not be reported as executed by this CPU run.
