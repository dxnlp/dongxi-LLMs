# Student-prefix API hardening: frozen follow-up

Status: specified on 2026-10-04 after independent review, before changing the
measured module or executing the follow-up. This is validation, not retuning.
The [original protocol](2026-10-04-student-prefix-distillation.md), raw results,
contract, input identity and journals remain historical and byte-for-byte intact.

## Two declared interventions

1. `topk_tail` will reject zero-support probability vectors before conditional
   tail decomposition. This finite-detail API is explicitly restricted to
   strictly positive full distributions. Full `kl_probabilities` continues to
   represent genuine zero-support mismatches as positive infinity. At k=V the
   coarsened zero tail remains valid; no floor or `infinity-infinity` subtraction
   will be introduced. Both p=(1,0), q=(0,1), directions and finite zero-support
   examples must fail explicitly at the detail boundary.
2. `states_from_pool` will require one collection cohort across all source groups:
   `(campaign, phase, update, arm, state_sha256)`. Offline pools must be update 0,
   arm `finite-offline`, and state `authored-finite-teacher-v1`. Student-prefix
   pools must identify one declared student-prefix arm and a SHA256 model state.
   Genuine individually digested records from different seeds, updates, arms or
   checkpoints will be rejected. Full original source coverage, four distinct
   ordinals per source, ordinary error retention and skip behavior remain intact.

## Fixed validation, with no selection

Run the expanded focused CPU suite using the existing isolated interpreter.
Replay all three seeds and all eight original arms with the frozen 30-update
recipe and same fixture/spec, with no new data, tuning or checkpoint selection.
Compare every numerical/non-timing field in `runs` and every response journal
record against the original. Exclude only keys ending in `seconds` and
`payload_sha256` values derived from those measured durations. Retain any failed
comparison, exception or partial replay; do not overwrite the original campaign.
The original report SHA256 is
`56ee9b50a319028716b395cf56d4130ced4377b3068c5418aa1124b414258f39`.

Use new follow-up artifacts under
`experiments/reports/2026-10-04-student-prefix-hardening/`; refuse an existing
directory. Bind original and current source/test hashes separately, original raw
results/journal hashes, actual argv/interpreter/package identity, expanded test
count, comparison rules, fresh notebook execution and inspected scientific
figures. Original observations are not silently relabeled as current-source data.

## Limits

No new teacher, task, seed, numerical recipe, privileged evaluation input,
occupancy gradient, pretrained model, API, GPU, installation or serving benchmark
is authorized. Passing these controls demonstrates rejection and preservation
within this bounded finite-support campaign, not general distillation quality.
