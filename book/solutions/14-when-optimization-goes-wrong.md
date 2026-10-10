# Chapter 14 — Worked solutions

Companion to the [chapter](../chapters/14-when-optimization-goes-wrong.md) and
[Day 24–25 notebook route](../labs/14-when-optimization-goes-wrong.md).

## 1. Correct optimization of a flawed proxy

The malformed action `35 0` has proxy reward 2 while the correct `35` has reward 1.
The exact expected-reward gradient increases probability of actions above the
current average. It therefore favors the malformed answer. Rising proxy and
falling strict accuracy are predicted consequences of the programmed objective.
The paired plot measures both; reporting only reward would hide the task failure.

## 2. Repairing a checker differs from repairing weights

A corrected verifier changes future updates. It does not restore diversity,
general skills or representations already lost during previous optimization.
Our repair experiment restarts from the same initial logits to isolate objective
quality. Repairing the actual hacked checkpoint would require another controlled
run, a recovery budget and external evaluation. Do not substitute one experiment's
result for the other's claim.

## 3. Low entropy has a context

A deterministic arithmetic task may warrant concentration on the correct
answer. On an unsolved prompt, concentration on one wrong answer destroys
exploration. Inspect correctness, valid fraction, output diversity and conditional
state identity alongside entropy. An entropy bonus can broaden outputs, but
cannot make the broadened candidates correct by itself.

## 4. The actual collection distribution is the behavior policy

Temperature scales logits; top-p removes outcomes and renormalizes the rest.
These operations change the sampling probabilities. Importance ratios must use
the probabilities of the distribution that generated the stored actions, not
the unmodified model distribution unless it was actually used. The basic course
runner avoids this ambiguity with temperature 1 and full-support sampling.

## 5. Average KL hides identity and tails

A KL value requires a named reference, state distribution, estimator and
reduction. A mean can hide extreme prompts and changes in response length.
Measure quantiles, fixed-prompt values and validity. First check identical-policy
KL, logit/label alignment and masks; then test the learning-rate or regularizer
hypothesis. A threshold breach is a reason to investigate, not a completed cause.

## 6. Denominator pressure

Response means spread equal response weight over variable token counts;
token means give long responses more total weight. With positive and negative
advantages this can change relative pressures on answer content and stopping.
Record the exact formula, EOS handling and token cap. A plot of nominal weights
isolates this mechanism, while a trained length-distribution comparison is
needed to establish the realized behavioral effect.

## 7. Synchronization does not rewrite history

A response collected under version 4 retains version 4 behavior probabilities
after the learner reaches version 6. Assigning version 6 to its denominator makes
the ratio an incorrect description of collection. Synchronization affects future
generation. Metadata checks reject unintended lag; intentional asynchronous
training needs an explicit lag/correction contract and engine integration tests.

## 8. Optimize the dominating cost

If generation takes 32 seconds, learning 6, verification 0.1 and synchronization 0.5,
halving learning time saves 3 seconds out of 38.6. Doubling useful generation
throughput saves 16 seconds under this simple model. Actual batching and memory
can change these numbers; the notebook labels them projected rather than Spark
measurements. Compare useful tokens, complete update time and safety together.

## 9. Resume the trajectory, not only its weights

Restore optimizer moments and step, scheduler, RNG, data/sampler position,
policy/reference identities and rollout progress. Reusing weights with a fresh
optimizer can be a valid new experiment, but should receive a new identity.
For replay checks, use a small fixed input and restored RNG to compare the next
sample and update under the same environment. Save recovery failures honestly.

## 10. Alerts versus explanations

An actionable alert names a failed configured condition and points to evidence:
nonfinite loss, reserve below 25 GiB, changed verifier, unexpected lag or reward
improvement alongside evaluation decline. A causal conclusion additionally
needs a mechanism and discriminating intervention. “Entropy fell” is a
measurement; “the model learned the correct deterministic action” needs
correctness evidence and alternate explanations checked.

## 11. Three probability distributions and a support boundary

Raw probabilities are 0.55,0.30,0.15. Behavior uses temperature 0.5, top-k 2 and
top-p 0.9, so it normalizes the squared first two probabilities: approximately
0.770701,0.229299,0. The declared temperature-one conditional target instead
normalizes 0.55,0.30:0.647059,0.352941,0. Its target/behavior ratios are about
0.839572 and 1.539216, not 1. The same source logits and retained mask do not
encode the same temperature. A matched temperature 0.5 target gives ratio 1.

For rewards 0,1,4, the conditional target's exact expectation is 0.352941.
Enumeration with the actual behavior denominator agrees, including the
gradient `[−0.228374,+0.228374,0]`. Substituting raw model denominators gives
0.269764 and gradient `[−0.174553,+0.174553,0]`: a different objective.
The analytical gradient is $t_i(R_i-J)/\tau_t$ on fixed support. The
[adjacent runnable reference](../../notebooks/day-20/03_behavior_probabilities_and_support.ipynb)
tests this identity before plotting the measured vectors.

The original full-support target requires action 2, which behavior never emits.
Its probability mass 0.15 contributes 0.60 to expected reward. No finite ratio
can recover that contribution from this collector. Preserve the raw objective
by changing collection to cover it, or explicitly change to a conditional
objective. Silently applying the old mask is not the former repair. This
single-state identity does not correct the state-distribution differences of
a general autoregressive off-policy objective.

## 12. Why a KL metric can disagree with a KL loss gradient

Under positive shared full support, $\mathbb{E}_p[k_1]$ and
$\mathbb{E}_p[k_3]$ equal $D_{\mathrm{KL}}(p\Vert q)$ because the extra
$q/p-1$ term has expectation zero. Freezing collected actions at $b=p_0$ while
differentiating only $k$ omits the derivative of their action distribution.
At the fresh point, frozen $k_1$ has expected gradient 0 and frozen $k_3$ has
gradient $p-q$. Exact forward KL has gradient
$p_i[\log(p_i/q_i)-D_{\mathrm{KL}}]$ instead.

In the finite fixture all five forward values are 0.299160737nats. Exact KL
gradient is approximately `[+0.391842,−0.242996,−0.148846]`; frozen $k_1$
is all zero and frozen $k_3$ gives `[+0.350000,−0.200000,−0.150000]`.
Differentiating the complete importance-weighted expectation
$\mathbb{E}_b[(p_\theta/b)k_\theta]$ reproduces exact KL for both statistics
in this one-state full-support example. Detaching the weight changes the
gradient again. An intended surrogate can differ from the full KL, but must be
labeled as such. This is not a universal trajectory-KL correction recipe.

## 13. Which stop permits a bootstrap?

Generated EOS is a scored response action; prompt EOS and EOS-valued padding
are not. Use valid lengths and prompt boundaries, then stop at the first
generated stop ID. A response ending only at the declared token cap has a
truncation boundary, not a sampled terminal decision. A continuing-task critic
can bootstrap there if its remaining state/value is available. A finite-horizon
task may instead define the cap terminal. The task convention decides; the
token value cannot.

The notebook shows both rows and rejects an unexplained early end. Its masks
index collated token positions and still need one shift/slice when used with
next-token logits. Recorded behavior likelihoods, advantages and reference
weights detach. Current target logits do not. Select valid positions before
arithmetic: zero times NaN remains NaN, and cannot safely exclude invalid
padding likelihoods. The tests assert finite current gradients, zero padded
gradient and absent gradients for old/reference/advantage tensors.

These are verified CPU mechanisms, not learned-value performance, model-scale
stability, a full PPO implementation or Mac execution evidence.

## 14. Distinguishability is not generalization

Removing an encoded collision means the model can now represent the two inputs
differently. It does not require their learned scores to rank them correctly.
The word experiment establishes a representation-level impossibility; the
character intervention removes it while preserving the raw fixtures and
predeclared test population. Its three held-out ranking accuracies remain 0.5.
The legitimate conclusion is that collisions were not the only limitation.
Training fit and calibration improvements do not supply the missing ranking
evidence. Sample size, shortcut learning and unseen patterns remain possible
explanations, not proven causes.

This matters before policy optimization: a frozen reward that fails independent
evaluation can still supply finite, consistent gradients. A learned critic can
then add a second measurement error at nonterminal caps. Inspect the reward,
the return target and independent task quality separately. Do not call an
unfinished prefix's displayed reward-model score a delivered terminal reward.

## 15. What exactly did rehearsal retain?

First measure the warm checkpoint on the unrelated task. If it solves only part
of that task, later rehearsal can both protect existing behavior and learn the
remaining examples. A final improvement alone cannot identify those effects.
Compare warm-to-final transitions on the same sources, preserve held-out cases
and label the initial capability honestly. The CPU experiment starts from
imperfect parity performance; it is not a pure forgetting test of a completed
skill.

Chosen-response likelihood anchors winners, not necessarily general knowledge.
Rehearsal anchors its particular demonstrations and adds forward/token cost.
The four-arm comparison separates these objectives, while clean/noisy and
length-matched pairs test different data hypotheses. A fixed strict evaluator
can legitimately disagree with a preference target containing extra style
tokens. Preserve that disagreement rather than changing the evaluator after
seeing the outputs. These controls do not establish a general pretrained-model
retention guarantee.

## 16. Saved work and elapsed time are different measurements

Caching avoids repeated prefix projections and attention work; batching adds
rectangular padding and changes kernel geometry. On the tiny CPU reference,
cached batching forwards 27 rather than 90 positions but its additional Python
bookkeeping and small kernels can outweigh that reduction in wall time. The
counts are not FLOPs and the timing is not a GPU serving benchmark. Measure
IDs, likelihoods, useful output, padding, tensor payload and actual time under
the same workload before describing a speedup.

## 17. Finish the preserved pending update first

Its old likelihoods, sampled actions, policy version and source cursor belong
to the policy that collected it. Restore those records together with weights,
Adam, reference and RNG, then consume the pending update before collecting new
data. Resampling changes both trajectory and cost; silently advancing the
cursor can skip data. The isolated CPU reference checks completed and pending
boundaries for tiny DPO/RLVR and preserves omitted-state divergences. It does
not establish resume in a pretrained Spark runner.

## Evidence-reading extensions

The chapter retains general systems questions 18–35. This guide preserves
their original detailed questions and numbering, so implementation-specific
checks remain reachable after narrative extraction.
[Appendix D](../appendices/d-reproduction-and-environments.md#d6-checkpoints-recovery-and-job-supervision)
provides the common checkpoint/supervision framework.

18. Why is a strong hash insufficient if two different states share its serialization?
19. Why can a leader exit 0 while the overall shutdown condition fails?
20. Why does a hard per-file limit fail to provide a total experiment disk quota?
21. Why can 204,800 positions of DPO update geometry exclude most logical forward work?
22. Why must an attempted update after a snapshot remain charged when that snapshot is restored?
23. Why can the final checkpoint fit an artifact cap while its safe publication cannot?
24. Why is validating a mock backend receipt different from proving real platform enforcement?
25. Why can checking a recovered pending pool spend work even without a new optimizer update?
26. Why do actual encoded lengths make a budget more useful without making it authorized?
27. How can cached generation use an uncached reservation without misreporting its work?
28. Why does budgeted snapshot publication not establish a whole-output disk quota?
29. How can replaying sampler draws validate a checkpoint without changing the training trajectory?
30. Why cannot a work prefix stored only inside a checkpoint authorize the work required to read it?
31. Why cannot a checkpoint payload contain its own completed save charge, and what should recovery do with later save failures?
32. Why does a guarded loader fail to bound pre-read work if identity collection already hashed the checkpoint? What should be bound instead?
33. Why do completed and pending RLVR snapshots with the same applied-update cursor require different first actions and save schedules? Can a pre-validation saved work prefix still retain later validation charges?
34. Why can an external runtime check still fail if its resource observer or incident logger blocks? What must remain uncertain after a logging failure?
35. How can bounded parallel file hashing preserve a recovery contract, and why are a completed snapshot, a whole-child exit and a three-way replay comparison still different claims?

## 18. Hash the state, not an ambiguous flattened description

Without dictionary boundaries, a key can appear to belong to an inner or outer
mapping while contributing the same byte stream. A cryptographic hash receives
identical bytes and must return identical digests; this is not a SHA256 collision.
Version 2 encodes types, container counts, byte lengths and tensor shape/dtype/
payload. Historical identities remain historical, and actual tensor-byte and
numerical comparisons bridge the migration. The digest still is not external
authentication, and trusted local checkpoint loading is not a hostile-input
sandbox.

## 19. A leader exit does not describe its workers

A leader may return successfully before an owned worker exits. Save the actual
leader code 0, but fail the shutdown criterion while a worker remains live or
uninspectable. A group signal reaches same-group processes; an independently-
sessioned worker requires its retained identity-safe handle. Never adopt an
unrelated PID merely because a conflict scanner reports it. The
[CPU guard controls](../../experiments/reports/2026-10-05-owned-workers-and-disk-guards.md)
test discovery and cleanup for fixed cooperative workers, not every daemonization
race or a production cgroup. Denied inspection must be preserved as uncertainty,
and cleanup/journal faults cannot justify declaring success.

## 20. Separate a file limit from a storage budget

Ten files can each satisfy a one-file limit while exceeding their intended total.
A per-file POSIX limit is a hard boundary on each child-written file, not a
directory quota or space reservation. An aggregate sample can trigger cleanup
after crossing, but the writer can grow between samples, and final evidence
writes add bytes afterward. Measure logical payload and available filesystem
space separately. A hard total guarantee needs a tested filesystem/platform
quota; sampled CPU fixture success does not supply one for Spark checkpoints.

## 21. One pair has four score branches

Each selected pair provides chosen and rejected sequences. Both the policy and
frozen reference score them, so one pair has four logical score forwards. Under
the current one-shift path, a length-512 branch supplies at most 511 input
positions. 100 updates times four pairs times two branches times two networks
times 511 gives 817,600 logical positions. 204,800 is only the declared single-branch
geometry before that shift. Baseline/final evaluation, generation and retries
are additional; backward activation recomputation is not included in this count.
This is a source-derived upper bound, not measured Spark FLOPs or latency.

## 22. Recovery must not erase a real attempt

The optimizer checkpoint describes a durable numerical boundary. The work journal
describes resource reservations and attempts, including later failures. They
need not have the same final cursor. Validate the snapshot's independently
retained ledger prefix, then replay all valid later charged entries. Restore the
optimizer but keep that spending. An interrupted library call can have unknown
partial work; its reserved allowance must not be silently refunded. A new output
directory or invocation ID cannot reset the immutable campaign allowance.

## 23. Publication has a peak

Old and new payloads coexist until publication finishes. Temporary and published
markers can also occupy separate entries, and failed writes leave partial bytes.
Reserve this complete bundle before opening the next artifact. Once completion
is observed, release only capacity whose absence or actual size is established
under the owned writer's contract. Never delete an old checkpoint merely to make
a proposed save fit. Cooperative accounting covers only routed writes; an
uncooperative export can still exceed it without a real physical quota.

## 24. A structural contract is not a platform observation

An authored fixture can claim an enforced timeout or empty GPU-owner list, and
the validator can correctly check its schema, freshness and binding. That proves
the source refusal rules, not the truth of the claim or a bounded real probe.
The current interface implements no production provider and refuses that mode.
A real readiness gate needs a separately authorized private backend, actual
ownership and membership observations, hard aggregate storage enforcement,
independent deadlines, bounded observer/cleanup behavior and GPU clearance.
Even that does not guarantee global host memory against unrelated jobs. Preserve
the 25GiB reserve and label samples rather than calling them continuous minima.

## 25. Verification has a computational path

A retained pending group must agree with the weights and collection RNG that
produced it. Rechecking selected likelihoods runs the collecting policy; replaying
the RNG-consumption trace can draw from a separate temporary generator. Neither
changes the restored optimizer cursor, but both execute real operations. Charge
them in separate validation dimensions, before they begin. Keep later failed
validation attempts when replaying an older trusted journal prefix. Otherwise
repeated reloads would create an unaccounted path around the job's allowance.

## 26. Geometry, integrity and authority are different

Encoded IDs and suffix masks reveal how many targets and scored positions the
actual branches require. Whole baseline/final panels and a declared attempt
schedule expose work that update geometry omits. But a supplied observation is
not authenticated tokenization; a live runner must reproduce it with the pinned
tokenizer/template/source. Even a reproduced requirement is not permission to
spend it. Require a separately retained full-vector budget decision and approved
stage. Do not silently interpret the old 204,800 geometry as permission for a
larger four-branch job plus evaluation. Fixture receipts remain fixture-only.

## 27. A conservative upper bound is not the measured path

For a prompt of length $P$ and a cap $C$, uncached decoding can forward at most
$CP+C(C-1)/2$ logical input positions. Cached decoding usually forwards the
prefix once, then one new input position per subsequent step. Early stops change
both totals. Reserving the uncached upper bound can protect an unchanged cached
observer, but report its actual executed calls/positions separately. Neither
number is FLOPs, cache memory, backward recomputation or GPU elapsed time. The
full/LoRA parity checks must retain the original cached dispatch, not replace it
with an easier-to-instrument algorithm.

## 28. Routed storage is not all storage

The shared writer can reserve snapshot payload, temporary/published markers,
entries and the retained earlier files. It cannot control an HF export, metrics
journal or child that writes outside that interface. Keep the snapshot claim
narrow and name those remaining paths. A physical aggregate guarantee requires
a separately verified quota/backend, not merely a counter class or free-space
sample. Preserve the failed file and its full reservation; a new output directory
does not erase the old root's storage history.

## 29. Temporary replay and live sampling have different roles

Replay the declared seeded stream using a separate generator and compare its
indices and resulting RNG state with the saved history. For $u$ completed DPO
updates with accumulation $a$, that check draws $ua$ times. Those draws are
verification work, not new supervision. Do not advance the live sampler while
checking. Apply the saved live RNG only after the complete state is accepted;
the next optimizer update must consume the original next indices. Count repeated
validation calls separately if the actual load and restore path both execute
them. Exact weights and a correct saved sampler do not imply zero replay cost.

## 30. Pre-read authority cannot depend only on post-read state

An existing journal opens unbound and cannot grant new reservations until its
trusted prefix is checked. A prefix stored only in the checkpoint is unavailable
before deserialization. Reserving in the semantic callback therefore covers later
checks, not the earlier byte verification, restricted load and generic finite/tree
scans. A separately retained bounded receipt can supply the pre-read expectation,
but its identity, same-journal binding and retained later charges need actual
ordering/failure tests. The separate shared I/O source and Day 25 microscope
exercise that boundary; physical and pretrained claims still require their own
evidence. Never create a new journal to bypass
the dependency, trust an unverified adjacent marker, or infer cross-host recovery
from copied receipts. File/resource envelopes and physical isolation remain
separate safeguards.

## 31. The save completion is later history

The payload must be fixed before its serializer, hashing and durable publication
can finish. It cannot contain a successful completion that has not happened yet.
Record the pre-save I/O prefix inside the payload, bind that exact prefix in the
separately retained receipt, and replay the physical journal's later entries on
resume. Those entries include the save reservation/completion and later failed
attempts. Keep the numerical checkpoint while retaining spending and partial
files. A readable marker after an uncertain fsync is not automatic proof that
the whole publication succeeded.

## 32. Admission must cover the first payload path

Generic identity hashing is itself a full payload read. If it runs before
admission, adding a reservation inside the loader cannot retroactively protect
it. Exclude the resume payload from that generic path. Read only bounded,
independently retained metadata, bind its expected payload SHA/size and journal
prefix, then let the admitted reader establish actual byte identity. Label the
pre-read digest an expectation until verification succeeds. Keep source/data
identity checks and every subsequent actual inspect/load separately visible;
changing the I/O path must not change the objective, sampler or reference.

## 33. One update cursor can name two different next actions

At $C_k$, no next pool has been collected; at $P_k$, the same $k$ updates have
been applied but the next group, source cursor and collection RNG are retained.
Apply that pool once without recollection. The full native lifecycle saves its
initial boundary and every pending/completed pair: $1+2U$ saves fresh,
$1+2(U-k)$ on completed resume, and $2+2(U-k-1)$ on pending resume. The pending
formula includes republishing the restored pending state and the subsequent
completed state. Add one admitted inspect and one load on each resume. These
are source-derived schedules, not measured FLOPs or a capacity approval.

The saved runner-work prefix precedes its semantic save check. The shared I/O
prefix precedes its whole save reservation. The external receipt pins those
exact saved prefixes; the physical journals retain the later validation,
publication and failed-read suffixes. A smaller prefix inside a valid payload
does not refund spending. Nor does recovery require rewriting the historical
payload to include a completion that happened after serialization.

The Exercise 8 microscope manually saves one boundary per arm, then actually
inspects and performs two loads, of which one deliberately rejects its callback.
It is not the full lifecycle. Compare its actual arrays/IDs and first resumed
collection count: completed adds one, pending adds zero; both recover the same
original four-update policy/reference/Adam/history/RNG. Logical source counts do
not establish language ability, hostile-payload safety, physical quotas or
bitwise cross-machine execution.

## 34. The watchdog must not inherit the failure it observes

An external check is useful only while its controller can keep checking time.
Calling a blocking probe or `fsync` from that loop merely moves the hang from
the model to the controller. Likewise, a readable pipe is not proof that an
entire response has arrived. Isolate those operations and bound response
assembly, logging and cleanup separately. Stop only the owned invocation;
discovered conflicting processes are evidence for refusal, not signal targets.

Retain the actual leader exit, worker-cleanup result and logger outcome as
separate facts. A stopped job with a failed final write is not a durable success
receipt. Keep available partial logs and label the unretained evidence unknown.
Inert failure controls test that mechanism; they do not demonstrate a successful
pretrained profile or continuous physical-resource enforcement.

The [first native DPO deadline case](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md)
is separate actual evidence. Its clean and source children exit 0, while fresh
resume reaches 600.683331 supervisor seconds with an unknown native exit and a
recorded reap timeout. Keep all three observations. The adapter returns 1;
substituting that for the missing native exit would falsify the receipt.
Above-reserve sampled memory does not establish continuous safety or diagnose
the deadline as memory exhaustion. A new configured retry must pass its own
recovery and shutdown checks without erasing the old failure or its spent work.

Run 02 also fails that gate despite actual CPU 8 witnesses and a durable update 2,
validation/generation and export. Its final result is missing. Later absent
processes or readable snapshots do not recover the original null native exit.
Run 03 supplies new completed exit-0 receipts for all three native children and
the CPU comparison under the unchanged 600-second guard per invocation; it does
not retroactively change either failure. Its resumed child time is 392.908058170
seconds, whereas its supervisor interval is 392.957157330 seconds. The outer
launcher interval is a separate owner-transcribed observation, not native time.

## 35. Preserve integrity while changing its scheduling

The serial default still performs each complete-file digest directly. The
explicit four-worker inventory schedules independent files concurrently, with
at most four outstanding tasks/file descriptors and 1-MiB buffers. It still
invokes every inventory, reads every body byte, applies the same no-follow,
regular-file, ownership, cap, missing-file, hardlink and race checks, reports
the original first failure and drains submitted workers. Metadata caching or
skipping a repeated inventory would weaken the contract; this change does
neither. The worker setting changes execution scheduling, not receipt identity,
the DPO objective, data, save cadence or work/I/O allowance.

Snapshot completion proves a durable numerical boundary. A whole-child exit
also requires later diagnostics, identity/integrity checks, final result and
supervisor cleanup to finish. The separate CPU comparison then admits the final
clean/source/resumed reads and compares all ten typed numerical components.
For actual run 03, those identities and the update 2 metric tail match, while
the frozen reference remains unchanged. This is stronger than matching three
export directories or finding update 2 in a journal.

Numerical equality does not require equal cost prefixes. The source's two
updates plus the resumed repetition of update 2 remain three updates in the
shared physical work journal, not two. CPU comparison inspect/load operations
also remain charged. Independent component hashing and artifact inventory
hashing are named exclusions from `DPO19`/I/O 9, protected by the bounded external
invocations rather than mislabelled as fully ledgered CPU work. Inventory hashes
remain within their invocation's time boundary; independent component hashing
remains within the separate CPU comparison boundary. No counters here measure
FLOPs or physical disk traffic.

The [actual report](../../experiments/reports/2026-10-05-native-dpo-recovery-deadline.md#run-03-whole-child-recovery-and-comparison-pass)
therefore supports local completed 1→2 recovery within these invocation bounds.
It does not isolate a causal speedup, supply a 100-update pilot result or prove
independent assistant/preference quality. Retain both earlier failures, their
null native exits, all spent work and partial/intermediate artifacts.
