# Actual complete-response distillation CPU reference

Status: executed original bounded teacher → response → student experiment. The
[specification](../specs/2026-10-04-response-distillation.md) and original fixture
were saved before fitting or collection. No recipe, seed, checkpoint or held-out
selection was changed after inspecting results. This is a 15-token symbolic
sum/parity task, not pretrained English reasoning or proof of rationale faithfulness.

## Actual command and identity

Executed on 2026-10-04, exit 0:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  -m dongxi_llms.response_distillation_lab \
  --fixture fixtures/response-distillation/items.json \
  --spec experiments/specs/2026-10-04-response-distillation.md \
  --output experiments/reports/2026-10-04-response-distillation
~~~

The saved [input identity](2026-10-04-response-distillation/input-identity.json)
records original interpreter argv including `-m`, separate rewritten Python argv,
Git/dirty state, seven source/test digests, fixture/spec digests and actual packages.
The environment was Linux ARM64, Python 3.12.14, Torch 2.14.1+cpu, NumPy 2.5.3 and
Matplotlib 3.10.8. No environment lock was supplied to this invocation; its actual
package snapshot is recorded, not falsely described as a lock installation.
The interpreter was the existing isolated CPU environment. No downloads, GPU,
installation, notebook server or model service was used.

The immutable [raw result](2026-10-04-response-distillation/results.json) SHA256 is
`8eea7cb206d449d0d3f4297248621463df8452980768792bf155ea869f2c2aee`.
The [contract](2026-10-04-response-distillation/contract.json) freezes the actual
fixture/spec, vocabulary/template, budgets, decoding and selection identity.
The 1,872-row [response journal](2026-10-04-response-distillation/responses.jsonl)
and 841-event [fit/completion journal](2026-10-04-response-distillation/events.jsonl)
preserve individual attempts and all 840 update observations. There was no source
or input drift during the command. Existing output directories are refused.

## What was trained and generated

Campaign seeds were 26011, 26012 and 26013. Teachers used seeds 27011, 27012 and
27013, width 24, one modern causal block and 6,552 parameters. Students used width
16, one modern causal block and 3,088 parameters. Models had full 15-way output
support, context 12, untied heads and float64 CPU weights. The student's two arms
started from exactly the same numerical initialization within each campaign.

Six original parity training prompts were balanced by final label. Four unseen
source prompts, their four alias siblings and four unseen sum-positive prompts
formed the test diagnostics: 18 total items, 14 source groups. Source groups,
underlying family/operands and encoded prompts cannot cross train/test. Alias
siblings share test source groups; they are not independent evidence. The unseen
family's final labels have a 3:1 imbalance.

Each teacher fit 120 updates on original five-token `STEP total ANS binary EOS`
targets. It then sampled eight responses for every training source, at temperature
1 from all 15 IDs with a five-action cap. Syntax, digits and EOS were never forced.
The adapter selected the first compatible naturally ended fulltrace per source,
without testing arithmetic correctness. All six sources had coverage in each
campaign; all 144 teacher-data attempts happened to be well-formed and correct.
The 18 selected parent traces were also correct. Zero actual teacher-data rejections
is a result, not a missing failure filter; tests separately exercise malformed,
missing-coverage and ordinary exception cases.

Complete-response students learned those five-token actual bodies; answer-only
students learned three-token `ANS binary EOS` subsequences from the same parent
records. Both arms used 80 fixed updates and the prompt-excluded response/EOS NLL.
The output grammar parser supports either fulltrace or answer-only generations,
but generation itself is unrestricted over all 15 IDs. The notebook includes a
fresh seed-26011 teacher and both student fits, exact final-state reconstruction,
and independent coordinate-generated response equality.

| Campaign | Teacher NLL, first → last | Complete response NLL, first → last | Answer-only NLL, first → last |
|---|---|---|---|
| 26011 | 2.658411 → 0.000936 | 2.696378 → 0.001525 | 2.653267 → 0.002450 |
| 26012 | 2.721323 → 0.001085 | 2.699059 → 0.001238 | 2.654945 → 0.000991 |
| 26013 | 2.720685 → 0.001022 | 2.675130 → 0.001072 | 2.654755 → 0.003466 |

Each student arm succeeded on all six training items under the first-candidate
final-answer rule. Low train loss does not establish successful task transfer.

## Common independent evaluation and negatives

Original student, teacher and both fitted students each generated eight new
responses on all 18 items after training selection. The phase namespace differs
from teacher-data collection. All models use the same cap, support, temperature
and independent arithmetic scorer. Three seeds produce 1,728 evaluation attempts:
1,428 natural EOS stops and 300 cap stops; no ordinary generation exceptions
occurred. Every invalid output and capped attempt remains in the denominator.

The table gives first-candidate final-answer success, out of four items per held-out
slice. The entire raw pool and every seed remain available; this is not best-seed
or held-out checkpoint selection.

| Campaign / arm | New source | Alias template | Unseen family |
|---|---|---|---|
| 26011 original student | 0/4 | 0/4 | 0/4 |
| 26011 teacher | 3/4 | 3/4 | 3/4 |
| 26011 complete response | 2/4 | 3/4 | 1/4 |
| 26011 answer only | 3/4 | 1/4 | 2/4 |
| 26012 original student | 0/4 | 0/4 | 0/4 |
| 26012 teacher | 4/4 | 4/4 | 3/4 |
| 26012 complete response | 3/4 | 4/4 | 2/4 |
| 26012 answer only | 3/4 | 2/4 | 1/4 |
| 26013 original student | 0/4 | 0/4 | 0/4 |
| 26013 teacher | 4/4 | 4/4 | 3/4 |
| 26013 complete response | 2/4 | 2/4 | 3/4 |
| 26013 answer only | 3/4 | 3/4 | 3/4 |

No universal superiority of complete-response versus answer-only supervision is
established. Both learn the train sources, and held-out source/template/family
behavior varies. Four-item slices are too small for a general capability claim.
The original student has one correct short response among 432 attempts, but no
first-candidate success; availability and deployed selection are different.

Final answer and printed step are independently checked. Across all 432 attempts
per arm, the teacher has 80 wrong-step/right-answer and 24 valid-step/wrong-answer
outputs; complete-response students have 26 and 15. Teacher joint step+answer
validity is 308/432, compared with final correctness 388/432. Complete-response
student joint validity is 289/432, compared with final correctness 321/432.
Answer-only students have 318/432 correct finals and no usable printed steps;
their step correctness is null, not an inferred correct or wrong explanation.
Four authored truth-table probes and the notebook semantic-fault control are
explicitly separate from actual neural generations and training replacements.

First-candidate, majority-final and mean EOS-inclusive model log-probability
selectors replay the same ordered eight-pool. Selector views exclude attached
gold fields; majority ties and likelihood ties use earliest eligible ordinal.
Repeated independently sampled strings count as votes; duplicate attempt IDs fail.
Likelihood is a learned-model confidence ranker, not a preference/reward quality
judge. The oracle availability field is scorer-only. Selection does not repair
a chosen wrong step. For seed 26012 answer-only's unseen-family slice, first
selection succeeds on 1/4 while majority and mean logp succeed on 0/4 despite
one item having a correct candidate. That negative is retained.

## Cost and memory boundaries

| Work (all three campaigns) | Updates / attempts | Supervised target presentations | Padded fitting positions | Generated actions | Uncached attempted prefix positions |
|---|---|---|---|---|---|
| Teacher fitting | 360 updates | 10,800 | 19,440 | — | — |
| Complete-response fitting | 240 updates | 7,200 | 12,960 | — | — |
| Answer-only fitting | 240 updates | 4,320 | 10,080 | — | — |
| Teacher-data collection | 144 attempts | — | — | 720 | 4,320 |
| All common evaluations | 1,728 attempts | — | — | 7,365 | 42,575 |

EOS, caps, wrong and invalid outputs are charged. Attempted and completed forward
positions agree because no ordinary generation error occurred. Full response
supervision uses 30 targets/update against answer-only's 18; matching 80 updates
does not match token exposure or padded forward work.

Measured campaign-body wall time was 19.474911 s. Teacher fits totaled 2.337991 s;
complete-response fits 1.685550 s; answer-only fits 1.599649 s. Teacher-data inner
forward/sample timers totaled 0.106216 s and all evaluation inner timers 1.162858 s.
Those inner timers exclude checkpoint hashing, payload digest/serialization and
append-only journal overhead. Overall campaign-body wall includes that overhead,
but excludes import/startup and final result serialization. Selector replay timers
are recorded separately; replay does not rerun a teacher.

Linux lifetime peak process RSS was 754,740 KiB. `/proc/MemAvailable` observations
were 123,671,328 KiB before and 123,619,988 KiB after. Two observations do not
establish a continuously monitored minimum, allocator peak or device memory.
Parameter counts prove a smaller student architecture, not faster serving or
lower total distillation cost. This uncached tiny CPU timing is not a Spark profile.

## Verification and remaining scope

Seventeen focused CPU tests passed before the actual command. They check one
causal shift, prompt labels, EOS target gradients, padding invariance, no truncation,
exact IDs, source leakage, rehashed tokenizer/template mismatches, teacher provenance,
first-eligible semantic-wrong acceptance, missing coverage, gold isolation, same
initialization, actual backbone/head gradients, independent seeded neural sampling,
ordinary partial failure work and refusal to overwrite a run. A scripted second
forward failure retains 9 attempted versus 4 completed prefix positions; it is
not presented as an error that happened in the main experiment.

The [visual notebook](../../notebooks/day-26/03_response_level_distillation.ipynb)
has nine executable CPU reference cells, adjacent deep exercises/solutions and six
data-backed figures; its fresh execution/figure inspection are bound by the
[verification record](2026-10-04-response-distillation-verification.json).
Chapter 15 is its daily route; Chapter 9 is the sequence-objective prose/lab bridge.
The unchanged three-logit `distillation_lab.py` SHA256 was
`1bb80abc3b9135d92577dcf827c3f5237e23ceb45759d7fd155a103380bc7c8c`.

Source preparation for a [local-only Spark transfer](../specs/2026-10-04-response-distillation-spark-transfer.md)
is validated with temporary dummy directories, not executed checkpoints. Actual
pretrained/model-scale transfer, natural-language rationale review, causal
faithfulness, optimized deployment speed, GPU memory/throughput, cross-hardware
bitwise replay and learner mastery remain unmeasured. No smaller lookup model is
substituted for the teacher/student autoregressive sequence experiment.
