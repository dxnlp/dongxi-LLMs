# Staged Spark comparison campaign and CPU supervisor contract

This specification is saved before implementation checks or fixture measurements.
It prepares the missing empirical campaign without authorizing or launching it.
The preparation program must never load a tokenizer or weights, acquire data,
profile a model, execute inference/training, start a service or manufacture a
checkpoint. Its optional verification mode may spawn only its own fixed,
standard-library CPU fixture children. The existing Day 9 result stays historical
evidence; it does not complete the new trained comparison.

## Separate scientific branches

The story branch compares two freshly initialized, matched 66.64M DongxiGPT
arms: the original peak learning rate 3e-4 versus a fixed half-rate 1.5e-4. Both
use seed 909, the same pinned prepared TinyStories bytes, GPT-2 tokenizer bytes,
document windows, training order, masks, effective batch 16, context 1024,
14,000-update schedule, 200-update warmup and proportional floor rates. This is
one learning-rate intervention, not a decoding intervention or an architecture
sweep. Candidate ceilings are four hours per learning arm, 50M valid training
targets and 229,376,000 padded training positions. These are proposed limits,
not approval, a fit/runtime forecast or measured exposure. Before pilot approval,
an actual profile must resolve the runnable target-count guard, local source/data
identities, safe geometry, save overhead and the whole-run shutdown contract.
The current story runner lacks an explicit valid-target cap; that is an unresolved
implementation gate, not a claim that the existing CLI enforces this proposal.

The assistant branch begins at pinned Qwen3-0.6B-Base and compares full SFT with
rank-8 Q/V LoRA on the existing source-disjoint interface generator. An adapter
must be explicitly merged with its exact base before downstream full-policy
evaluation or DPO. A declared SFT parent then anchors unchanged-parent,
chosen-only SFT and DPO controls. Additional chosen-NLL/rehearsal DPO arms remain
optional gated extensions; they consume extra supervision. They do not silently
become the original DPO baseline or a universal retention repair.

The reasoning branch separately evaluates genuine base and genuine post-trained
weights. Raw base, explicit-chat base, instruct chat with thinking disabled and
instruct chat with thinking enabled are distinct interface rows. A thinking
template toggle does not create a different checkpoint or establish rationale
faithfulness. Fixed-initialization G4/G8 RLVR pilots use the strict integer/EOS
reward, sixteen updates and a 64-token cap; their unequal attempted token budgets
must be recorded. Mathematical held-out evaluation remains separate from the
four-pair runner diagnostic and reward. No universal base→SFT→DPO→RLVR ladder
joins unrelated tasks or parents. Qwen3-1.7B validation and later scale/seed
confirmation require fresh justified approval, not automatic expansion.

## Planned rows must survive failure

The preparation output must retain every profile, smoke, recovery, pilot,
interface evaluation, adapter merge, controlled comparison, capstone and optional
confirmation row. Each row declares a branch, proposed parents, dependency
packages, evaluation contract and budget. Actual checkpoint/file/interface
identity, measurements, process exit, approval and genealogy remain null until
their evidence exists. Failed or denied rows remain present with an explicit
reason rather than being removed from a favorable checkpoint table.

The following ladder applies to each applicable branch:

1. Separate acquisition approval for each exact model, tokenizer and dataset;
   already cached bytes do not imply inference/training permission.
2. Actual local byte hashes, immutable upstream declaration, source/environment
   lock identity and observed tokenizer/template/stop compatibility.
3. Approved single-checkpoint profile with finite updates or inference, actual
   host/CUDA resources, valid/padded work and external exit/deadline evidence.
4. Branch smoke with gradients, masks, initial behavior likelihoods and immutable
   reference checks; then interface and completed/post-collection recovery.
5. Separate pilot approval naming the exact parent, data, command, target/position
   and output-token budgets, deadline, disk, evaluation and resource envelope.
6. Fixed initial/final comparison and all planned outputs, including caps,
   invalid formats, regressions, zero-signal groups and interrupted jobs.

Existing pretrained DPO/RLVR runners do not yet expose verified replay of both
completed and post-collection states. The CPU cache/recovery lab is not that
production validation. Such pilots must remain blocked until their own recovery
gate passes. A saved optimizer file alone does not satisfy it.

## Frozen evaluation contracts

The module must declare an original twelve-opening story publication panel,
separate from the three repeatedly inspected development openings. Freeze
greedy decoding and full-support temperature 0.8 sampling with seeds
909/1909/2909; four attempts per opening, output cap 256 and context 1024.
Compare predetermined updates 0/400/4000/8000/14000, retaining missing checkpoint
cells when interrupted. Publish only a matched achieved checkpoint comparison;
do not choose the most pleasing story or checkpoint. Corpus contamination and
local tokenizer/template binding remain pending audit before using this panel.

Score grammar, entity/object consistency, causal continuity, repetition and
ending independently on a frozen 0/1/2 rubric. Retain each raw completion,
token IDs, selected likelihoods, natural EOS versus cap, failures and all rater
scores. Blind checkpoint/arm labels and output order; use two independent raters,
record disagreement and adjudication, and resample at the source-opening group
for paired uncertainty. No human ratings are fabricated by the CPU fixtures.
Report fixed validation NLL and matched frozen-checkpoint train/development
NLL as separate metrics; repeated text alone is not proof of overfitting.

The assistant contract freezes generator value/source groups and task slices,
the audited saved template, complete answer/termination mask, unchanged decoding
and held-out publication items before fitting. The reasoning contract freezes
the original twenty-item arithmetic/algebra/multi-step panel and source/task-
family slices, exact bounded grading, invalid/unsupported/ambiguous answers,
natural stops versus truncation and attempted/accepted token costs. Publication
items cannot select coefficients, checkpoints, a preferred branch or decoding.
Regression and preference panels remain distinct from reward and story rubrics.
Logical panel hashes are not claimed to be observed model-interface hashes.

## Budget and evidence fields

Keep requested and measured values separate. Training budgets include valid
target presentations, padded physical positions, updates, microbatch,
accumulation, length, dtype and actual parameter/optimizer/reference costs.
Generation includes prompt/response valid tokens, padded/forwarded work, all
attempts including rejected retries, output cap, stop reason and runtime.
Null actual values are required for every unexecuted stage. Source identity,
declared parent and actual checkpoint genealogy are different records.

A host reserve of at least 25 GiB is a launch and sampled stop threshold, not
a continuous physical-memory guarantee. Preserve measured sampling intervals,
minimum sampled MemAvailable, actual CUDA peaks where available, deadline
signals, actual child exit and journal failures; never sum unrelated peak-memory
quantities. Known-service and large-RSS inspection is a conservative read-only
diagnostic, not proof that every GPU process is idle.

## Predeclared bounded supervisor fixtures

Use the approved isolated CPU Python, no packages or environment changes.
The only child command is the module's fixed standard-library fixture entry
point; it has no model/data/download/runner arguments and spawns no descendants.
The supervisor journals fsynced preflight/start/sample/signal/actual-exit/failure
events and signals only its own direct Popen child, never a discovered process.
It checks real Linux MemAvailable and sanitized process diagnostics before spawn
and periodically during supervision. Reserve cannot be lowered below 25 GiB.
No process environment or raw command line enters an artifact.

Freeze these eight collection rows: successful exit 0; deliberate exit 7;
self-signal SIGUSR1; child ignoring SIGTERM until the deadline sends SIGKILL;
injected 24 GiB preflight rejection; injected 24 GiB post-start reserve stop;
injected conflicting-service rejection; injected post-start probe failure.
Positive/nonzero/signal fixture deadlines are 2 seconds; the blocking fixture
deadline is 0.2 seconds, sampling 0.02 seconds, TERM grace 0.1 seconds and KILL
reap grace 1 second. Injection is clearly labelled a control, never a measured
host shortage or an actual discovered service. Every row and raw journal remains
in the report, including nonzero and refused launches. Independent tests add
invalid thresholds/unknown modes, observer-interruption cleanup, journal failure
and prevention of unrelated signals where practical.

This external direct-child deadline works independently of a child's Python
loop, but is not a realtime scheduling guarantee. It does not contain or kill
grandchildren, supervise a production launcher tree, prevent every memory spike,
authenticate approval or replace a cgroup/platform guard. Production integration
and external-stage execution remain pending. Source/spec hashes, actual invocation,
environment and independent CPU checks accompany the report. The learner stays
Day 9; no Git/global/chapter/registry changes, GPU/Mac/API work, publication,
service startup, animation rendering or shared-environment installation occurs.
