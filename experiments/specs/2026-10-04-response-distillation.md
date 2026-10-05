# Complete teacher responses and a generating student

Mode: original bounded CPU reference, frozen before any teacher/student fit or
candidate collection. This is response sequence supervision, not soft-logit
distillation or a pretrained reasoning experiment.

## Task model and fixed fits

Use campaigns26011,26012,26013. Teacher initialization seeds are27011,27012,27013;
student seeds are26011,26012,26013. Within a campaign, both student arms start
from exactly the same initial numeric weights. Every campaign remains visible.
Teacher width24/heads3/MLP48 and student width16/heads4/MLP32 each have one modern
decoder block, matching KV/query heads, untied heads, vocabulary15, context12,
float64 CPU, no dropout. Matching vocabulary/template does not imply matching
hidden dimensions. Record actual parameter counts, not an assumed speedup.

Vocabulary: PAD0,BOS1,EOS2,STEP3,ANS4,PARITY5,ALIAS6,POSITIVE7, digits0..6 atIDs8..14.
Input is BOS,instruction,a,b (fourIDs). English descriptions are not parsed.
Original train parity pairs are(0,0),(0,1),(1,0),(1,1),(0,2),(1,2), balanced.
Source-heldout parity pairs(2,0),(2,1),(2,2),(2,3) and their alias-template
siblings remain related within the test split. Four unseen-family sum-positive
items use(0,0),(0,1),(1,0),(1,1) and a previously unused instructionID. IDs,
source groups, underlying family/operands and encoded prompts cannot cross
train/test. References and explicit-step totals are independently checked from
integer arithmetic; a checked printed total is not causal reasoning faithfulness.

Teacher supervised targets are STEP,total,ANS,answer,EOS. Fit120 full-batch
updates on the six train sources only, AdamW lr0.01,weight decay0,clip1,full15-way
next-token cross-entropy. Select update120 regardless of held-out scores.
Student complete-response arm learns selected actual teacher five-action bodies.
The answer-only arm derives ANS,teacher-final,EOS from those exact same chosen
records. Both run80 full-batch updates, AdamW lr0.015,weight decay0,clip1,full
15-way objective. Prompt loss excluded, response including EOS included, exactly
one causal shift, padding ignored. Five versus three targets/source means unequal
token presentations and padded forward work; equal updates are not equal compute.
Wrong well-formed teacher steps/answers remain trainable; no semantic correction.

## Teacher adapter and eligibility

All teacher and student responses are actual neural autoregressive outputs.
At every step sample temperature1 from all15 output IDs, cap5; no forced STEP,
ANS, numeral or EOS. Prompt+response fits context12; never silently truncate.
Each attempt seed derives from campaign, model state, item ID and sample ordinal.
Collect exactly eight attempts per training source from the fixed teacher.
Choose the first eligible fulltrace response per source, never the best gold
answer: eligibility requires exactly STEP,digit,ANS,binarydigit,EOS, naturalstop,
no exception, compatible vocabulary/template/provenance. It does not require
the printed step or final answer to be correct. Retain all rejected candidates
and their costs. If a source has no eligible record, do not replace it by gold;
mark both student arms blocked for that campaign, preserve missing coverage and
continue other campaigns. Never tune the recipe to recover a blocked arm.

Each selected example retains source/item/split, teacher state/initialization/
training-seed/budget, exact full output IDs/text/stop, attempt identity/digest,
template/vocabulary identity, provenance and cost. The adapter rejects swapped
vocabularies, templates, changed payloads, duplicate attempts, test-source rows,
missing/nested EOS, unsupported special IDs, overlength/truncated responses and
empty supervision. Derived answer-only examples retain the same parent IDs and
are labeled transformations, not new teacher events. Shared09 digest/seed helpers
and shared SFT token_loss_sum are reused; its color-task journal is not relabeled
as a generic reasoning journal.

## Common independent evaluation and selectors

After all training selections/fits, evaluate original student, complete-response
student, answer-only student and the teacher on the same18 frozen items, eight
new coordinate-seeded generated responses per model. Training and evaluation
sample coordinates have separate namespaces. Evaluation has the same all15-way
sampling, cap5, EOS and scoring rules; no test fitting or checkpoint selection.
Accept either fulltrace STEP,total,ANS,binary,EOS or short ANS,binary,EOS as a
well-formed output. Report final correctness, explicit-step correctness (null
when absent), joint step+answer validity, format, naturaltermination, cap and
error separately. A correct final with wrong printed step remains distinguishable
from a wrong final with valid printed step; no rationale faithfulness claim.

Compare first candidate, majority-final voting and mean EOS-inclusive model
log-probability selection on the same ordered eight-pool. Gold-free selector
views include IDs/logp/text/format parsing only. Majority ties and exact rank
ties choose earliest eligible ordinal. Repeated independent strings count as
votes; duplicate attempt IDs fail. Any-correct availability is an evaluator-only
diagnostic. Selection may choose a wrong-step/correct-final response: preserve
its printed-step result instead of rescuing it through the answer score.

Report all campaigns/arms/source/template/family slices and source coverage.
Teacher data-generation tokens/attempts/forward positions/time and teacher fit
work remain separate from student fit/generation and selector replay costs.
All attempts/rejections count. Record valid generated actions including EOS,
uncached full-prefix attempted/completed positions/calls and wall time. Replay
does not rerun the teacher, and missing telemetry remains unknown. Linux CPU
peak RSS/observed MemAvailable have explicit measurement boundaries.

Four original authored probes fix the truth table of valid/invalid printed
step versus correct/wrong final answer. They are not neural generations or
training replacements. Preserve the existing three-logit forwardKL/T² module
unchanged; sampled-response NLL and teacher-distribution KL are different losses.

## Evidence limits and transfer gate

Freeze source/spec/input hashes before the command, check afterward, persist
append-only attempt/fit/selection/failure events and refuse overwrite. Save all
negative or blocked outcomes. Test masks/shift/padding/EOS, compatibility,
provenance and leakage, actual gradients, causal generation, semantic-fault
acceptance, selector gold isolation, independent coordinate sampling and partial
failure evidence. Add one focused Day26 visual notebook with adjacent reference
answers and inspected data-backed figures; integrate Chapter9 only.

Prepared Spark transfer source/spec accepts existing local full/merged teacher
and student checkpoints plus exact saved tokenizers and templates. It must not
download, install or launch a run; actual model-scale execution requires separate
approval/profile and report. Frozen training/test sources, capped teacher attempts,
response masks, trace reviews, separate teacher/student cost and25GiB reserve
remain mandatory. Do not translate a symbolic STEP checker into an unreviewed
natural-language rationale verifier. No API/GPU/Mac/install/Git/service/media
or publication is authorized by this CPU reference.
