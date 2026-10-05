# Student-prefix hardening without changing the experiment

Status: executed follow-up, exit 0. The
[hardening protocol](../specs/2026-10-04-student-prefix-hardening.md) was saved
before source changes or replay. The original
[report](2026-10-04-student-prefix-distillation.md), results, contract, identities
and both journals remain historical. No seed, task, optimizer, budget, teacher
rule, prefix reduction or evaluation input was changed.

Independent review found two API edge cases, not contamination of the measured
campaign. `topk_tail` previously accepted zero-support vectors and could produce
undefined `infinity-infinity` detail. It now explicitly requires strictly positive
full distributions. Full KL still supports genuine zero-support infinities, and
positive full distributions at k=V still yield an exact zero coarsened tail. No
probability floor is used. `states_from_pool` now rejects mixed campaign, phase,
update, arm or checkpoint cohorts and enforces the finite offline collection role.

Exact original [module](2026-10-04-student-prefix-hardening/original-student_prefix_lab.py.txt)
and [test](2026-10-04-student-prefix-hardening/original-test_student_prefix_lab.py.txt)
snapshots retain their measurement hashes `ec6fef04e3914c53f3b1d969470e8b412f490d047428b77749e2973798915fb3`
and `7ff70c68d0217813d225edd5cb738a2983083b10560f60f7d4d7783a428103d5`.
The hardened module/test hashes are separately recorded in the
[acceptance record](2026-10-04-student-prefix-distillation-verification.json).

## Actual replay and regression checks

The existing isolated CPU interpreter executed:

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python \
  experiments/reports/2026-10-04-student-prefix-hardening/replay.py
~~~

The [comparison](2026-10-04-student-prefix-hardening/replay/comparison.json)
binds current source/input hashes and actual invocation identity. All three
campaigns, eight arms per campaign, 720 updates, exact distribution controls and
10,656 collected response records match the historical experiment. Only keys
ending in `seconds` and duration-derived `payload_sha256` were excluded from
those numerical/record comparisons. Top-level memory/provenance observations
are not compared as numerical trajectories. All original artifact bytes and all
current source/input bytes remained unchanged during replay. Replay wall time,
including fsynced retained journals and comparison, was 59.950321 s.

New [raw results](2026-10-04-student-prefix-hardening/replay/results.json),
[responses](2026-10-04-student-prefix-hardening/replay/responses.jsonl) and
[events](2026-10-04-student-prefix-hardening/replay/events.jsonl) are preserved in
a separately created directory. There was no failed full replay or retuning.
The final focused panel passed 27 tests in 0.618 s, including genuine mixed-seed,
phase, update, arm and checkpoint records plus zero-support adversarial controls.
The intentional existing-output argparse error is part of a passing refusal test.

Independent reviewers separately passed the hardened 27-test panel and audited
all original digests/cohorts, 1,728 final response grades and reported token work.
Root also replayed all 24 original arms before this hardening, matching every
non-timing numerical field. These are scoped code/provenance checks, not a claim
that tiny held-out reversal generalized well.

## Fresh notebook and visual inspection

The [Day 26 extension](../../notebooks/day-26/05_student_prefix_and_teacher_context.ipynb)
executed all nine code cells in the isolated `dongxi-course-clean` CPU kernel,
with no skipped references and seven figures. It verifies the historical source
snapshots and performs two fresh full-budget hardened arm replays. Every exported
figure was inspected for labels, overlaps, numerical correspondence and honest
scope. The first passing notebook had an overlong colorbar label; a separate
fresh pass shortened it and corrected a title spacing issue. Both execution
manifests and their earlier evidence are retained in the acceptance record.

No installation, external acquisition, API, pretrained model, GPU, Mac, service,
occupancy derivative, full SDPO implementation or serving-speed claim is involved.
The authored finite teacher and private-hint recovery rule remain explicit
limitations, and all original negative results remain unchanged.
