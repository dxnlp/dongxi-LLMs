# TinyStories learning run 01 — completed evidence

Recorded 2026-09-14. Mode: learning, not a benchmark or general-capability claim.
This completion report supersedes the pending outcome in the
[launch ledger](2026-09-13-tinystories-learning-launch.md). The
[specification](../specs/2026-09-13-tinystories-learning-01.md) preceded execution.

## Evidence and portability

Local raw evidence is under `outputs/day09-learning-01/`; the launcher's
`outputs/day09-learning-launch/status.json` records the actual child exit.
The adjacent [portable JSON](2026-09-14-tinystories-learning-result.json) contains
the model/recipe/data identity, completion status, memory summary, all recorded
development observations, aggregate training measurements, and selected complete
continuations. SHA256 hashes identify the raw files used. It includes no weights
and is not a complete copy of all raw telemetry.

Sample selection is retrospective and structural: initialization, first
observation (400), updates 4000 and 8000, and completion (14000); first configured
prompt, both greedy and temperature-0.8 sampling. The full run contains three
fixed prompts at every observation. This does not imply these five checkpoints
were a predeclared analysis subset; they were selected by update, not quality.
Original training observation cadence and prompts were fixed before the run.
Token IDs are omitted from this compact archive; text, settings, EOS flags and
generated-token counts are retained.

## Completed run

- Randomly initialized 66,638,848-parameter modern decoder; GPT-2 tokenizer only.
- Full prepared training data: 1,792,647 documents, 390,708,926 valid targets.
- Completed 14,000 updates and 48,839,975 valid target presentations.
- Processed 229,376,000 tensor positions; valid fraction 21.2925%.
- Recorded trainer duration: 12,102.9132 seconds; schedule complete, not stopped
  by the four-hour deadline. Actual child exit code: 0.
- Sum of update timers: 11,493.7445 seconds. Ratio-of-totals throughput:
  4,249.27 valid targets/s on update timers; 4,035.39/s on full trainer duration.
  Earlier data preparation is outside both timing boundaries.
- Minimum sampled host available memory: 104.7032 GiB; reserve: 25 GiB;
  sampling interval: 0.2 seconds; recorded abort reason: null.
- CUDA peak allocated: 11,293,046,784 bytes; reserved: 12,490,637,312 bytes.
  These are not independent physical RAM quantities to sum with host measures.
- Final checkpoint: `outputs/day09-learning-01/update-014000.pt`.
  Final observation file is `final.json`, not `update-014000.json`.

## Learning evidence

The fixed development selection has 512 windows and 107,264 valid targets,
selected with seed 909. It is a partial held-out split, not the full validation
corpus or an untouched publication test. Its NLL decreased at every recorded
observation: 10.904913 at initialization; 3.394123 at400; 2.086143 at4000;
1.828181 at8000; 1.674315 at14000.

The last500 online batch means average 1.661964. These changing-model,
equal-batch-weighted observations are not directly matched to the frozen,
valid-token-weighted final development measurement.

Archived samples show a qualitative transition from fragments to recognizable
story constructions, while retaining malformed language, repetition and
unexplained speaker/role changes. This is a qualitative inspection, not a
scored coherence evaluation. Final first-prompt greedy and sampled completions
both emitted EOS, demonstrating termination in those cases, not coherence.

The [repetition diagnosis](../../learning_artifacts/day-09-pretraining-run-and-diagnosis/repetition-versus-overfitting.md)
records later backend probes and the limits of the original user report. These
were exploratory inference checks, not another training run. Prompts were not
persisted by the playground; that note is not a full raw request archive.

## Boundaries and remaining work

- Exact-normalized deduplication is not a near-duplicate contamination audit.
- No blinded multi-prompt coherence scoring, training-seed replication, or
  matched frozen-checkpoint train/development gap was performed.
- Preflight next-update recovery was verified; no new recovery test at the
  final checkpoint is implied.
- Batch8/16 profiles informed resource selection; their under1% throughput
  difference is not a meaningful demonstrated speed advantage.
- No controlled trained scale/data/recipe comparison has yet been completed.
- More training, packing, dropout or decoding penalties remain hypotheses,
  not established fixes for the observed story errors.

The dashboard and CUDA playground were stopped at the user's request on
2026-09-14; port8765 was verified closed. No checkpoints or training evidence
were removed. The chapter and evidence-reading lab work without either service.
