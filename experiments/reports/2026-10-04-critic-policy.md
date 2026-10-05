# Learned critics and frozen text rewards: a retained negative result

The bounded CPU package is reproducible, but this recipe did not learn reliable
color matching. Very small preference loss, an exact value oracle, and a learned
critic are each insufficient evidence that a generated response satisfies the
independent task rule. Several runs also failed to emit EOS within the delivery
cap. Those failures remain in the quality denominators and raw records.

## What was measured

The [premeasurement specification](../specs/2026-10-04-critic-policy.md) froze
inputs, seeds, grammar, budgets, scaling, evaluation panels, and quality rules
before either reward fit. The implementation trains a real one-layer causal
actor with 3,428 parameters and a separate causal value model with 3,105
parameters. Both consume the actual prompt and generated prefix. These are
randomly initialized tiny CPU models, not pretrained language models.

Two actual character-level reward models were fitted from the independently
authored balanced and confounded preference fixtures, saved, reloaded with
expected file digests, and frozen before policy updates. Only train pairs fit
the reward models. Train-only score centering/scaling and a fixed tanh transform
define the bounded terminal proxy; they do not calibrate a success probability.

Train objects are cup/key, calibration bag, test box, and withheld control mug,
each crossed with red/blue. The actor's actual word-ID encoding and the reward
character encoding are split-disjoint. The quality rule is a separately held
aside program: correct requested color, at most one style marker, and delivered
EOS. It supplies no reward, critic target, scaling choice, or checkpoint choice.
These authored judgments are not independent human feedback.

The conditional response grammar has fourteen terminal paths. Its environment
horizon is four actions, while the primary collector cap is three. An unfinished
prefix is censored, not terminal. Its continuation value is bootstrapped from the
declared environment, not from an EOS that the collector actually delivered.
The separate cap-four panel is a predeclared diagnostic; forced EOS at its final
grammar position is not a claim of spontaneous termination.

## Fixed work and evidence

Every arm used actor seeds 2001/2002/2003, forty fresh rollout batches, four train
prompts and eight sampled paths per prompt. The four arms are balanced/oracle,
balanced/learned, balanced/noisy, and confounded/learned. No best seed or
checkpoint was selected. Actor initialization is paired across arms.

Across all twelve runs, the measured work was:

- 15,360 training paths and 38,296 valid response tokens, including emitted EOS;
- 480 actor updates and 240 separate learned-critic updates;
- 460,800 full-prefix generation-forward positions, including post-stop dummy
  rows, not all actor/critic/reward computation or billed API units;
- 3,240 scheduled evaluation records at updates 0/20/40, 1,080 unchanged-policy
  control records, and 1,080 cap-four records: 5,400 evaluation records total.

Oracle and noisy arms take no learned-critic updates. Equal sampled-path budgets
therefore do not imply equal optimizer work or wall time. Batch wall time is
measured; it is not allocated into invented per-response latency.

The two preference fits each completed 120 updates at the frozen seed 1611:

| Reward arm | Initial pair loss | Final pair loss | Saved/live scores exactly equal |
|---|---:|---:|---|
| balanced | 0.823047 | 0.000005650 | yes |
| confounded | 0.822340 | 0.000006832 | yes |

The balanced comparisons match style on each side. The confounded arm associates
correctness with fancy style and reverses the same content contrast. Its two
orientations are duplicates, not two independent observations. Equal pair counts
and fit budgets do not equalize information. Bare color responses are outside
these styled preference comparisons; their reward is model extrapolation.

## All final sampled outcomes

Train quality uses 32 samples per seed; test and control each use 16. Every
unfinished, wrong-color, and repeated-style response stays in the denominator.
The proxy column scores the delivered prefix diagnostically. A high prefix score
is **not** terminal reward earned when no EOS was emitted.

| Reward / critic | Seed | Train quality | Test quality | Control quality | Train EOS fraction | Train prefix proxy |
|---|---:|---:|---:|---:|---:|---:|
| balanced / oracle | 2001 | 0.50000 | 0.50000 | 0.50000 | 1.00000 | 0.67109 |
| balanced / oracle | 2002 | 0.50000 | 0.50000 | 0.50000 | 1.00000 | 0.67109 |
| balanced / oracle | 2003 | 0.50000 | 0.50000 | 0.50000 | 1.00000 | 0.67109 |
| balanced / learned | 2001 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.01134 |
| balanced / learned | 2002 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.04136 |
| balanced / learned | 2003 | 0.50000 | 0.50000 | 0.50000 | 1.00000 | 0.67109 |
| balanced / noisy | 2001 | 0.03125 | 0.06250 | 0.00000 | 0.03125 | 0.76011 |
| balanced / noisy | 2002 | 0.53125 | 0.62500 | 0.50000 | 1.00000 | 0.67863 |
| balanced / noisy | 2003 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.07463 |
| confounded / learned | 2001 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.60392 |
| confounded / learned | 2002 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.59076 |
| confounded / learned | 2003 | 0.37500 | 0.56250 | 0.62500 | 1.00000 | 0.72926 |

All final greedy rows either delivered the constant blue answer, achieving 0.5
quality on each balanced-color slice, or failed to terminate and achieved zero.
The test/control splits contain only two prompts each: sampled fractions do not
establish broad held-out language transfer or a ranking of critic methods.

The exact oracle averages future proxy under the **current** actor. It does not
replace the proxy with independent quality. Learned critics sometimes became
badly inaccurate: final valid-state oracle MSE for balanced seeds 2001/2002/2003
was 9.790965 / 4.077184 / 0.007612; the confounded values were 12.974008 /
0.327242 / 0.000773. Censored-prefix bootstrapping and reward extrapolation are
plausible failure mechanisms exposed by this task, not a causal proof that one
uniquely caused every bad trajectory. No recipe was retuned after these results.

## Reproduction and source identity

The [full verification ledger](2026-10-04-critic-policy-verification.json)
preserves all training token paths, old log probabilities, masks, values,
bootstraps, TD residuals, advantages, detached targets, update metrics, and all
evaluation records. It was measured on Python 3.12.14, PyTorch 2.14.1+cpu,
Linux/aarch64, float64 CPU, one thread, in 8.009 seconds. The earlier
[first ledger](2026-10-04-critic-policy.json) is also preserved. Their source
hashes identify historical measurement snapshots, not the final source tree.

After adding the promised pre-fit actor-ID collision and duplicate/empty
vocabulary rejection, the
[compact current-source acceptance](2026-10-04-critic-policy-acceptance.json)
reran both fits and all twelve policy runs. Every update metric, raw training
path, evaluated response, frozen-policy control and cap-four panel matched the
verification ledger exactly. Only measured wall durations were excluded from
that numeric comparison. Twenty-one source/lesson/figure identities
remained unchanged during this run; both reloaded reward exports matched their
live fitted scores. The collector, tests, and replay took 10.067 seconds in all.

Sixteen focused tests passed. They independently enumerate GAE and lambda
endpoints, distinguish cap bootstrap from the deliberately broken cap-as-EOS
case, check padding and causal likelihood alignment, and establish separate
actor/critic/reward gradient boundaries. Old/current likelihood alignment was
exactly zero before each fresh actor update. The math check passed with 54
Markdown files, 1,169 expressions, and zero issues at collection time.

The [new notebook](../../notebooks/day-20/04_learned_critics_and_frozen_text_rewards.ipynb)
also passed a fresh isolated CPU kernel: seven code cells and five inspected
figures. Its schematic is labeled structural; trajectory and reward plots use
actual fitted models and fixed-budget records. Seed bands show all three seeds,
not confidence intervals or a selected winner. The acceptance JSON retains its
source hash and fresh-kernel verification row.

The saved verification rewards are
[balanced](../../fixtures/critic-policy/frozen-rewards-verification/frozen-balanced.json)
and [confounded](../../fixtures/critic-policy/frozen-rewards-verification/frozen-confounded.json).
Their file SHA-256 values are respectively
`75a6cfc5ee33267f92771347c6ce1c6e915b650bb709db7c878af0c768340fe5`
and `9d160cdf5fa47c1136e3b5c1e159eec3be91c2aaac2b3c8522fc4b4d59e0ea7a`.
Actor/critic in-memory parameter-byte identities are recorded, but no reloadable
actor/critic checkpoint export is claimed.

For a new raw run, use unused report/export paths as shown in the
[laboratory](../../book/labs/12-language-generation-as-a-policy.md). The compact
collector requires a newly verified notebook manifest and a historical raw
reference; it never overwrites either.

## What remains outside this evidence

This is a real neural actor/critic loop with a real fitted, frozen text reward,
within a finite conditional grammar. Its local clipped surrogate uses one
fresh update per rollout, so initial ratios equal one: it is not evidence for
multi-epoch or production PPO. CUDA, pretrained checkpoints, free-form language,
Mac execution, larger-data reward transfer, independent human review, crash
recovery, and exact resume have not been tested here. Positive mechanism tests
and negative policy outcomes coexist; passing this CPU package does not remove
those broader pending checks.
