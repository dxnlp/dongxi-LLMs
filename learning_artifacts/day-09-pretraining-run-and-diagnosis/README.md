# Day 9 — Watch DongxiGPT learn English stories

Opened2026-09-13. Book placement: Chapter6's pretraining experiment and diagnosis
sections. Updated2026-09-14: the baseline run is complete and integrated into
the chapter; a controlled trained comparison and learner defense remain open.

## Completed evidence and chapter synthesis — 2026-09-14

- [Completed-run report](../../experiments/reports/2026-09-14-tinystories-learning-result.md):
  14,000updates,48.84M valid targets,3h22m,exit0,final development NLL1.674315.
  Adjacent JSON preserves the full validation curve and fixed-grid samples.
- [Chapter6](../../book/chapters/06-pretraining-as-a-controlled-system.md):
  sections6.14–6.20 integrate the actual run rather than a hypothetical future
  success. Eighteen worked answers and a standard-library evidence-reading lab
  are linked there. Dedicated Day9 analysis notebook remains planned.
- [Repetition diagnosis](repetition-versus-overfitting.md): distinguish observed
  repetition, decoding controls, incomplete generalization evidence and
  unresolved coherence. Lower loss does not independently score whole stories.
- Dashboard and playground were stopped on user request; port8765 verified
  closed. Reading the case does not require model loading or a server.
- Animation opportunity check extended existing budget/next-token/temperature
  candidates; production remains unapproved on Mac. No new article commissioned.
- Next: study the case, define a frozen coherence evaluation, and propose a
  bounded controlled trained comparison. Do not automatically launch it.

## Historical planning and launch record

The sections below record earlier decisions and limits at the time they were
made. Their pending/unauthorized-run statements are superseded by the completed
evidence above, not instructions to repeat preparation or restart training.

## Latest execution evidence — supersedes planning-only statements below

Learning run01 is now authorized and launched (2026-09-13T20:56:39 UTC), after
full data preparation and95 passing tests. Prepared1,792,647 training stories
and21,990 held-out validation stories; configured14,000 updates/four-hour cap,
with generated samples every400 updates. The
[launch ledger](../../experiments/reports/2026-09-13-tinystories-learning-launch.md)
distinguishes preparation, process startup and later learning evidence. Older
"no multi-hour run" notes below describe the prior smoke-only phase.

Monitoring follow-up: user accepted a local read-only training observatory;
implementation and use are recorded in
[the monitor guide](../../docs/TRAINING_MONITOR.md). The fixed-opening
checkpoint-comparison proposal is now implemented as a browser view of actual
logs, not an animation or a fabricated sequence of learning milestones.

The user authorized building and verifying the pipeline. See the
[smoke report](../../experiments/reports/2026-09-13-tinystories-pipeline-smoke.md)
and [reproduction guide](../../docs/TINYSTORIES_PIPELINE.md):82 CPU tests,
three full-size66.64M BF16 updates on Spark, finite gradients/activations,
safe sampled memory, and bitwise-identical GPU checkpoint recovery. Prepared
1,024 train and128 validation documents, not the full corpus. Gibberish samples
are retained as evidence of the initial learning state, not hidden as failures.
No multi-hour run, learner mastery, or coherent generation is claimed. Next:
full intended data audit and a sustained batch/throughput profile, then finalize
the learning budget. Earlier proposal history below remains useful context.

## Learner decision

The learner chose the easier first objective: watch a randomly initialized
DongxiGPT learn to continue coherent short English stories. Multilingual and
broad-domain training are deferred. For forward planning the learner asked us
to suppose the existing chapters are covered. Respect that premise and proceed
to Day 9; do not convert it into independently verified mastery or completed
GPU evidence, and do not force a return to older practice entries.

## Selected data and evidence boundary

User selected on2026-09-13 after reviewing alternatives:
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories),
with the [authors' paper](https://arxiv.org/abs/2305.07759v2).
Primary sources inspected 2026-09-13. The paper describes synthetic short stories
generated using GPT-3.5/GPT-4 and a restricted vocabulary, and reports coherent
generation with small models. These are source-reported results, not a guarantee
for our model, tokenizer or token budget. The dataset card lists
`cdla-sharing-1.0`; exact artifact revision, terms, content quality, duplicates,
held-out split and preprocessing still require inspection before use.

No corpus download, model training, profiling, installation, commit, push or
animation rendering has been performed by this discussion.

Working tokenizer/model baseline is now recorded in
[the design and parameter accounting note](tokenizer-and-model-baseline.md):
GPT-2 byte-BPE,50,257 IDs; our own12-layer,512-wide modern MHA/RMSNorm/RoPE/SwiGLU
decoder with tied embeddings and1024-token context. Shape-only counting verifies
66,638,848 parameters; GPU implementation and fit remain unverified. Tokenizer
and original TinyStories file-pair repository revisions are pinned in the note.

## Proposed observation protocol

Freeze a small authored story-opening panel and generation settings before
training. Include an untrained checkpoint and predetermined later checkpoints.
Compare grammar, character consistency, event continuity, repetition and endings
alongside fixed held-out token loss. Preserve all panel outputs, not only the
best story. Multiple fixed sampling seeds help distinguish sampling variation
from changes associated with training. Check for copied training passages.

Any anticipated transition from fragments to grammatical sentences to coherent
stories is a hypothesis, not a guaranteed sequence or timetable. This is a
completion model, not an instruction-following assistant. Reusing a tokenizer
would not reuse pretrained model weights.

## Next decision

### Making learning visible — proposed observation panel

After the GPT-2 speedrun discussion, resume our TinyStories experiment rather
than replacing it with a GPT-2 reproduction. Borrow the distinction between
throughput and time to a fixed quality target; FineWeb's speedrun loss threshold
does not transfer to TinyStories. No new run or data download is authorized.

Proposed authored openings (not yet audited against the corpus):

- Basic narrative: "Once upon a time, a little rabbit lived near a forest."
- Character and object continuity: "Lily put her red ball in a box. Then she went outside."
- Cause and effect: "Tom wanted to fly his kite, but there was no wind."

Use the same openings, generation limits, decoding settings and fixed seeds at
each selected checkpoint, including random initialization. Add a greedy view
alongside fixed-seed samples; do not tune decoding separately per checkpoint.
Finalize cadence and numerical decoding settings in the executable run spec.
Record grammar, entity/object consistency, causal continuity, repetition and
ending behavior separately. A generation cut off at the token limit is not the
same as a model choosing EOS. These are developmental probes, not a sufficient
general-capability benchmark or a pristine final test after repeated inspection.

Keep fixed held-out next-token loss alongside free-running completions: fluent
local transitions do not guarantee a globally consistent plot. If the panel is
used to change the recipe, treat it as development evidence and reserve unseen
openings for a later final check. Learner understanding remains unassessed.

Next discussion question: can held-out loss improve while a story still forgets
where Lily left her ball? Then settle document boundaries/long-story handling
and the bounded smoke specification. No animation production requested; the
existing checkpoint-comparison display opportunity covers this discussion.

### Dataset alternatives discussed — 2026-09-13

Historical comparison before the selection above: user asked for other datasets
appropriate to this easy English-story objective. At that point none was selected
or downloaded. Primary cards and
papers inspected; ranking below is an agent judgment, not measured Spark results.

- **SimpleStories English**: strongest alternative shortlist candidate. The
  [dataset card](https://huggingface.co/datasets/SimpleStories/SimpleStories)
  describes over two million synthetic stories and provides topic, theme, style,
  grammar and generation metadata. Card lists MIT. The
  [paper](https://arxiv.org/abs/2504.09184v3) reports parameterized generation and
  small-model experiments. Metadata could support balanced sampling or evaluation
  slices, but generation intent is not automatically a verified property of the
  resulting story. Use story text only initially; tags are not model conditioning
  unless deliberately included. Exact release and preprocessing remain unpinned.
- **Cosmopedia v0.1 stories configuration**: a later extension candidate. The
  [official card](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)
  describes Mixtral-generated content and a dedicated stories configuration;
  card lists Apache-2.0. Broader material warrants length/difficulty/content
  inspection before treating it as an easy replacement. Do not confuse the
  complete multi-format corpus with its stories subset or v0.2.
- **WritingPrompts**: a human-written contrast rather than the recommended first
  corpus. The [original paper](https://arxiv.org/abs/1805.04833) describes roughly
  300K prompt/story pairs. The
  [author repository](https://github.com/facebookresearch/fairseq/tree/main/examples/stories)
  supplies splits and notes its original first-1000-word modeling policy. More
  varied, longer narratives would likely complicate this first experiment;
  that is a design judgment. Data-use terms require separate review; code license
  or a third-party mirror label must not be assumed to settle corpus rights.

Earlier proposed step, superseded by the TinyStories selection: inspect matched samples and token-length distributions from
TinyStories and SimpleStories before choosing one baseline. No automatic mixture
or two-run campaign. If later comparing datasets, keep tokenizer/model and
token exposure controlled and evaluate both models on the same frozen panels;
raw validation losses on different datasets are not a fair model ranking.

Conceptual opportunity: dataset choice changes the prediction problem, not only
its volume. Simple language may make coherence observable within a small run;
that does not imply general-language competence. Learner interpretation of these
trade-offs remains pending. Article/animation production is not requested.

Discuss corpus/tokenizer choice, then profile a candidate before fixing parameter
count, context length, token budget and runtime. The roadmap's 50–150M range is
provisional, not a requirement to maximize model size. Keep Spark's memory
reserve, validation contract and checkpoint-recovery test in the future spec.
Retain the Day 9 requirement for one controlled comparison after a healthy
baseline. Heavy execution needs a bounded specification and explicit approval.

Reuse opportunity: a checkpoint-comparison view would make learning visible.
It is a proposed experiment display, not an approved animation production task.
