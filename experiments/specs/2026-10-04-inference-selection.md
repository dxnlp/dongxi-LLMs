# Actual candidates and answer selection

Mode: original bounded CPU reference, frozen before fitting or collecting.
This is an evaluation microscope, not a pretrained reasoning benchmark.

## Question and predeclared model

Does a selector find a correct complete answer already present in a generated
pool, and what work was spent on unsuccessful candidates? Hold each pool fixed
while changing its selector. Keep oracle availability separate from selection.

Use seeds 10051, 10052 and 10053, all retained. Initialize the original shared
TinyDecoder: vocabulary10, width16, query heads4, KV heads2, head dimension4,
one modern block, hidden32, context8, untied output head, float64 CPU. Compare
its initial state and its final state after80 full-batch train-only supervised
updates, AdamW learning rate0.02, weight decay0, no dropout, clipping norm1.
Targets are two actual actions: the independently authored binary answer and
EOS. Both target positions use conditional support EOS,0,1. Freeze update80
regardless of any held-out result; never select a checkpoint or seed afterward.

The six train sources encode parity with operand pairs(0,0),(0,1),(1,0),
(1,1),(0,2),(1,2), balanced three answers each. Source-heldout parity pairs
are(2,0),(2,1),(2,2),(2,3), balanced two each. Their alias-instruction siblings
remain in the same source and test split. Four separately identified unseen
sum-positive-family items use(0,0),(0,1),(1,0),(1,1); their answer imbalance
is explicit. IDs, underlying family/operands, source groups and symbolic prompt
IDs cannot cross splits. English prompt strings are documentation: the model
receives an explicit four-ID interface, not an English parser.

## Actual collection and costs

Collect eight candidates for every item under each frozen initial/final policy.
Each candidate has a separate deterministic RNG seed derived from model seed,
policy state, stable item ID and ordinal. Sample temperature1 over EOS,0,1 at
every action; cap3. The grammar supplies no answer or forced EOS. Preserve
immediate EOS, repeated numerals, caps, all IDs and conditional likelihoods.
Raw text spells EOS; grading text removes only an actual final EOS.

Every attempt is persisted, including partial ordinary errors. Record prompt
IDs, conditional support, selected-action likelihoods, model/state/contract
identities, generated actions including EOS, attempted/completed full-prefix
forward positions/calls, first-forward prefill time, later decode time and
host wall time. Rescore every successful path under the same policy/support,
including EOS, in a separately counted forward: scoring actions, input positions
and wall time remain separate. Check recomputed log probabilities against the
collection values. No KV cache; CPU timings are not optimized serving latency.
Record process peak RSS and observed starting/ending MemAvailable, not CUDA
memory or a continuously monitored minimum. Replay-only unknown costs stay
null. No estimate from string length, no zero-cost claim for absent telemetry.

## Fixed pool selectors and budget controls

Reuse ordered prefixes N=1,2,4,8 of the same eight attempts. Never generate a
different pool for a selector. Compare:

- first attempt, even when invalid;
- majority frequency over supported binary, one-integer, naturally stopped
  candidates; tie: earliest eligible ordinal;
- unique-answer support voting, one vote per distinct parsed answer; ties:
  earliest eligible ordinal (an intentionally weak deduplication control);
- maximum mean conditional log probability over the same eligible candidates,
  EOS included, ties earliest; this uses train-only learned policy likelihood,
  not a trained correctness judge;
- a constructed longest-output preference over eligible candidates, ties earliest;
- any-correct complete-path availability, explicitly evaluator-only and undeployable.

Non-oracle selectors receive a strict projection with no reference, arithmetic
problem, correctness, format gold or graded status. Supported answer extraction
is an input-side declared rule, not comparison to gold. Invalid/error candidates
can abstain from voting but never disappear from costs or attempt denominators.
Independently sampled duplicate strings count as repeated votes; exact duplicate
sample IDs are rejected. Record raw-string and canonical-answer diversity,
duplicate fraction, valid-support size, invalid/error/truncation frequency,
pairwise wrong-answer agreement and correctness agreement without asserting IID.

Also evaluate token budgets3,6,12,24 on the same ordered pool. A candidate enters
the eligible prefix only if its full generation fits the remaining budget.
The first excluded boundary candidate is charged as attempted/generated work
but is not selectable; stop there, do not peek ahead for cheaper answers. Preserve
overshoot and the boundary rejection. Generation-token budgets are not total
compute budgets: prompt/rescoring positions and wall times remain visible.
Compare equal attempts across selectors and equal token-budget caps across
initial/final policies. Report cap versus actual consumed work; no exact-cost or
monotonic-improvement promise. A non-adaptive retrospective replay stops at a
whole-candidate boundary; it is not token-level interruptible generation.

## Evidence and acceptance

Retain all train histories, first-gradient reach, full numerical model state
hashes, every candidate, parser/error outcomes, selector decisions and budgets.
Freeze source/spec/fixture hashes before execution and check after. Persist
partial records and a failure event on an invocation failure; never overwrite
an existing run directory. Save an exclusive final summary only on completion.

Report every seed/policy/N/budget overall and by predeclared train, heldout-source,
heldout-template and heldout-family slice. Count source siblings as related;
descriptive pool metrics are not independent population confidence intervals.
Test gold isolation, stable ties, duplicate/invalid/empty handling, exact token
accounting and unknown replay telemetry, likelihood rescoring, source leakage,
causal generation, real decoder gradients and error preservation. Build one
visual notebook with predictions, adjacent solutions, fixed-pool perturbations
and inspected figures. Retain negative results; no tuning after held-out inspection.

No downloaded model/data, external API, GPU, Mac execution, installation, Git,
service launch or animation rendering. All task text and new logic are original.
