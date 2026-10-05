# Checkpoint availability and portable verification

The planned Qwen3-0.6B-Base checkpoint was not found in the configured cache
or checked course/platform locations on Spark. The cached Qwen3-0.6B checkpoint
is a different, post-trained model; it is not substituted for the declared base
parent. This resolves the immediate next action: request acquisition and
tokenizer-only inspection before asking for a launch-ready profile.

This check also found and corrected a genuine portability mismatch in the new
matched-control test route. Source corrections and local CPU tests do not close
the actual Mac, hosted or pretrained experiment gates. The goal remains active
at 13 of 18 packages complete in bounded CPU scope, with the learner on Day 9.

## Observed checkpoint and data availability

The [inspection record](2026-10-05-checkpoint-availability-and-portability.json)
captures Linux/aarch64 host `spark-aa66`, the effective cache settings and exact
checked paths. No alternate cache environment setting was present. The ordinary
Hugging Face cache has Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`,
but no directory for the declared Qwen3-0.6B-Base revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Known course/platform output inventories
contain the existing story checkpoints and older platform SFT export, not the
declared base. This is a scoped inventory, not an exhaustive filesystem search.

The [official model card](https://huggingface.co/Qwen/Qwen3-0.6B)
identifies the non-Base model as pre- and post-trained. Its locally observed
weight-file size is 1,503,300,328 bytes; only its small config was hashed.
No safetensors payload was read and no pretrained model was loaded.

The [pinned official Base repository](https://huggingface.co/Qwen/Qwen3-0.6B-Base/tree/da87bfb608c14b7cf20ba1ce41287e8de496c0cd)
and its public API supplied a separately retained
[metadata projection](2026-10-05-checkpoint-inspection/upstream-manifest.json).
The ten repository files total 1,203,641,805 bytes. The weight file accounts for
1,192,135,096 bytes, with upstream-declared SHA-256
`cd2a512003e2f9f3cd3c32a9c3573f820bb28c940f73c57b1ddaa983d9223eba`.
These are remote declarations, not verified local weights or measured download
traffic. No model/tokenizer repository file was acquired. The metadata response
was bounded to 256 KiB; a projection, not the upstream chat-template body, is saved.

The native profile's `outputs/course-sft-interface-v1` directory is absent.
Replaying the original generator in memory reproduces all three saved data-card
hashes: 240/60/120 records with 80/20/40 mutually disjoint value groups.
That verifies the raw fixture declaration, not tokenizer-dependent labels.
Actual encoded targets and tokenizer/interface identity remain unknown.

## The profile is not yet launch ready

The existing proposed profile is full SFT: 20 updates, microbatch 1,
accumulation 4, context limit 256, seed 1212, learning rate 2e-5, a 900-second
external ceiling and at least 25 GiB host reserve. It is not the 400-update pilot.
The [native lab command](../../book/labs/09-supervised-fine-tuning.md)
still contains explicitly unresolved work and snapshot allowances.

Before launch, encode with the exact approved Base tokenizer and native mask
logic. Training selects the first 80 indices of the frozen seed-1212 shuffled
240-record order. Its target count is not the 20,480-position geometric bound.
The SFT16 `valid_targets` dimension must cover training **and** two complete
60-record development NLL passes. Baseline/final generation additionally reserves
16 attempts of at most 64 new tokens. Fresh execution saves at updates 0 and 20;
do not invent diagnostic checkpoint loads merely to use a schedule calculator.

Sizing the SFT16 contract, separate I/O9 contract, snapshot/journal envelopes
and privately owned output paths remains pending. The current campaign adapter
has no admitted production argv/execution API; its fixed CPU-fixture supervisor
does not externally supervise this native command. A genuine owned external
deadline/resource route is required before a model launch. Successful profiling
is not a prerequisite for the first profile; smoke/recovery gates precede pilots.
Supporting whole-job quotas and story I/O remain recorded in their own plans.

The platform environment's installed metadata matches the six inspected version
entries in its saved lock: Torch 2.13.0+cu130, Transformers 5.16.1, PEFT 0.20.0,
huggingface-hub 1.28.0, tokenizers 0.23.1 and NumPy 2.5.2. Neither Torch nor
Transformers was imported for this check. Version membership does not establish
complete environment equality, CUDA/BF16 compatibility or memory fit.

## Corrected portability scope

The chosen-control CLI deliberately requires actual Linux `MemAvailable` and
25 GiB reserve. Its success test previously expected that CLI to pass everywhere,
including the declared Mac lane and hosted CPU VMs using a separate 2 GiB teaching
policy. That is a static source mismatch, not an observed Mac/hosted failure.

Only that CLI integration test is now scoped to an eligible Linux host with an
observed reserve. All scientific/helper tests remain enabled; the production
CLI's guard is unchanged. Four injected selection controls cover non-Linux,
reserve boundaries and missing/malformed/failed probes without claiming another
host ran. The CPU orchestrator also now describes the recorded invocation's
platform, instead of unconditionally saying its evidence is not Mac evidence.
Its scope still excludes other-platform reproduction and model-scale outcomes.

Root verification retained two actual command records:

- [69 focused tests](2026-10-05-checkpoint-inspection/portability-verification.json)
  passed in 4.375 seconds of unittest time, exit 0, no skips on this Spark host.
  The actual tiny random local CLI test ran. Seven selected source hashes stayed
  unchanged during this command. Two host-memory samples remained above 25 GiB;
  this is not continuous supervision.
- [6 reproduction checks](2026-10-05-checkpoint-inspection/reproduction-scope-verification.json)
  passed in 1.073 seconds, exit 0. Four selected source hashes stayed unchanged.
  These check scope, locks, workflow source and owned-process failures/timeouts,
  not actual hosted execution.

This is not a full-suite or all-notebook rerun. The original 18-test matched
experiment and earlier 65-test follow-up remain historical, unmodified evidence.
All 45 campaign rows retain their 945 null actual fields, jobs started stays zero,
and actual genealogy remains empty. No pretrained/CUDA/Mac/hosted result follows.

## Next authority request

Authorize only acquisition of the exact pinned Base model/tokenizer files and
CPU tokenizer-only inspection/sizing, or provide the exact existing local path.
The anticipated repository payload is about 1.20 GB; this is not a training or
model-inference approval. Do not install an environment, start a service, launch
GPU work, execute another campaign row, push Git or dispatch hosted CI under
that request. Present the fully sized profile separately before launch.
