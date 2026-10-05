# Student prefix distribution and teacher context

Premeasurement protocol: original bounded CPU finite-teacher/neural-student
experiment. Save before any fit or candidate collection. No full SDPO, self-teacher,
pretrained teacher, model-scale reasoning or sampling-gradient reproduction is
claimed. The earlier complete-response experiment remains byte-for-byte unchanged.

## Original task and exact teacher

Vocabulary8: PAD0, BOS1, EOS2, REVERSE3, HINT4, A5, B6, C7. Questions are
`BOS REVERSE x y`; correct responses are `y x EOS`. English descriptions are not
model inputs. Train pairs are AA, AB, BA, BB, CA, AC. Test pairs BC, CB, CC are
source-group/pair/encoded-question disjoint. Hints contain the correct reversed
pair and are authored supervision, not verified sibling samples or human feedback.

The authored finite teacher has no learned parameters. At every query it assigns
0.8 probability to one target token and 0.2/7 to every other token, conserving full
support and mass. With privileged training context, the target is the hint's token
at action positions0 and1, and EOS at position2. Without a hint, if the generated
prefix agrees exactly with the correct response prefix, its target is the same
correct next token. If the prefix has deviated, its target is the last generated
A/B/C token, or the question's first symbol if there is none. This explicit
history-sensitive recovery failure is a designed finite control, not observed
behavior of a real teacher. The hint therefore changes recovery targets on wrong
prefixes, rather than silently changing the student's input.

Hints are never passed into student inputs or evaluation generation. Student
input is only the question and its own generated history. A sampled HINT output
is an ordinary invalid token, not access to teacher-private data. The scorer may
use independently authored references after generation. Never evaluate with a
demonstration appended or rescue wrong/invalid completions with the hint.

## Fixed factorial comparison

Campaign seeds16021,16022,16023. Each student is an actual randomly initialized
TinyDecoder: vocabulary8, width16, heads4/KVheads4/headwidth4, one modern RoPE/
RMSNorm/SwiGLU block, hidden32, context8, untied head, float64 CPU, no dropout.
All eight arms start from identical numerical weights within a campaign:

- Prefix source: a fixed offline finite-teacher sampled pool, or fresh current-
  student autoregressive prefixes collected before every update.
- Conditional divergence: forward KL(teacher||student), or reverse KL(student||teacher).
- Training teacher context: no hint, or authored privileged hint.

Every arm uses30 AdamW updates, learning rate0.015, weight decay0, clip1 and a
global valid-state mean. No supervised prompt logits, post-EOS positions or
right-padding enter the KL objective. Each collected action supplies its preceding
question+history state, including the state that samples EOS. A cap does not
invent an EOS target; it contributes its actually visited three states.
Natural short stops supply fewer states and therefore less total loss weight.
Report actual state counts/forward work; equal updates do not mean equal state
exposure, teacher calls or generation cost. No checkpoint/seed selection or tuning
on held-out behavior is allowed.

Offline collection: once per campaign sample four trajectories for each train
source from the privileged authored finite teacher, all8-way temperature1, cap3,
natural EOS. Save every response and freeze those same prefixes for all offline
arms, regardless of the context used later to score them. The finite categorical
teacher incurs queries/actions but no neural forward positions. Its generation
is explicitly finite sampling, not a neural teacher response.

Student-prefix collection: before every update, each source supplies four actual
neural trajectories with the same full8-way temperature1, cap3 and natural EOS.
Sampling seeds derive from campaign, collection phase, update, item and ordinal;
the arm/context/direction is omitted from the random coordinate to pair the
streams across arms. The actual model state and arm remain in attempt identity.
Paired streams do not force identical outputs once weights differ.

Raw action records retain chosen probabilities, stop/cap/error, actual prefix
identities, model/interface/source identities and attempted/completed work. A
group with an ordinary generation error is skipped entirely for that update,
with all attempts and costs retained; no resampling. If no valid groups remain,
the update fails explicitly rather than performing a fabricated zero step.
The campaign saves partial journals on failure; it never overwrites an old run.
Separately audit whether each collected group contains a complete correct sibling.
Groups without one would be skipped by a sibling-demonstration rule, but this
diagnostic does not gate our authored-hint training. Its counts are labeled
`would_skip_without_sibling`, not actual fitted-prompt skips or SDPO execution.

Teacher target arrays are detached and retain the exact declared token mapping.
Sampling and state IDs are nondifferentiable observations. The actual update
is the gradient of conditional KL at sampled fixed states; it does not include
the derivative of the state-occupancy distribution or a REINFORCE term. Changing
prefix source and changing KL direction are independent experimental factors.

## Evaluation and exact controls

Evaluate each initial student and all eight final arms on the same nine items,
eight new full-support/cap3 coordinate-seeded actual generations. Use a separate
evaluation namespace, no hint or teacher query, and natural EOS with exact
two-symbol reversal scoring. Report train/test whole-response correctness,
format, EOS/cap/error, final-symbol correctness, and the full item-level pool.
Retain all arms/seeds/negative outcomes; a correct last symbol does not replace
whole-response correctness. Teacher/query, collection, KL scoring, neural
fitting, independent generation and fixed-pool diagnostics have separate costs.

At identical initial student states, independently compare both KL direction
gradients and both teacher contexts while holding states fixed. Save before/after
state hashes and first-backward embedding/Q/MLP/head gradient reach. Tests check
no teacher target gradients, same vocabulary mapping and hint exclusion,
right-padding invariance, EOS/cap boundaries, duplicate/source provenance,
actual student generation and retained ordinary failure costs.

Top-k microscope uses student-selected token indices at k1,2,4,7,8, then sums
all unselected probabilities into an exact tail bucket for both distributions.
Do not discard/renormalize the tail or insert an arbitrary probability floor.
Full and bucketed distributions must conserve mass. For either KL direction,
full loss equals bucket loss plus the appropriate tail-weighted conditional KL;
the bucket is a coarsening that loses within-tail token detail and need not have
the same gradient. k8 recovers the full objective with a zero tail.
The authored equal-top2/equal-tail-total control must show bucketed loss0 while
full KL remains positive. A separate zero-support control reports finite forward
KL and infinite reverse KL explicitly; infinities are not used to fit models or
written as invalid JSON numeric values. These are exact finite controls, not
measured pretrained approximation error or real large-vocabulary savings.

## Source and evidence boundary

The design follows the topic distinction in the pinned RLHF Book revision
eecc49e1e1daf1be670d7242eb090240b5bb0f04: chapter12's offline/on-policy
distillation discussion, code/distillation/README.md and loss.py. Those files
are inspected references; no code, prose, images or upstream results are copied.
The new implementation is original. Its explicit finite teacher, direct conditional
KL update, authored hints and no-resampling protocol do not reproduce that
repository's self-teacher/sibling/large-model training stack or a complete SDPO.

Bind actual source/spec/fixture/contract identity before and after the command;
record the approved isolated CPU interpreter/packages, actual wall time,
Linux lifetime RSS and bounded MemAvailable observations. No API, model download,
pretrained model, install, GPU, Mac, server, Git, publication or media production.
Build one Chapter15/Day26 extension notebook with adjacent deep exercises,
fresh kernel replay and inspected scientific figures. Root owns the chapter
integration and global registry/trackers; this package adds only its notebook index.
