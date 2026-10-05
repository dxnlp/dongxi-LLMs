# DPO work limits reject blocking and symlinked inputs

The current frozen DPO source passes 63 CPU tests and the separate actual CLI
FIFO control rejects with process exit 1 in 0.6725 seconds, without creating an
output directory. That nonzero exit is the intended negative result, not a
training failure. The numerical recovery collector still passes: a fresh process
restores update 3 and completes update 6 exactly while retaining work charged
after the checkpoint.

The [original 61-test report](2026-10-05-cumulative-work-budgets.md) and its
run-01 output remain unchanged, with their historical source identities. They
did not test this initial cap-file input gate. This report is the current
source-hardening acceptance, under the separately saved
[premeasurement protocol](../specs/2026-10-05-work-limits-input-hardening.md).

## Defect and narrow correction

The production CLI originally called `Path.open('rb')` on `--work-limits` before
checking the input type. A FIFO with no peer could block at open, before a bounded
read or the run's guard existed. A byte-count limit did not make that open safe.

The current reader walks ancestors with retained no-follow directory descriptors
and opens the final component using `O_NONBLOCK | O_NOFOLLOW`. It requires a
regular file before reading, checks its size, then reads at most 64 KiB plus one
refusal byte. Size and modification observations must remain consistent during
the read, and descriptors close on success or rejection. Normal regular JSON
still follows the existing strict caps/schema parser. This changes neither the
DPO objective nor the declared work dimensions, reservation policy or training
recipe.

Two additive tests cover regular and oversized files, directories, final and
ancestor symlinks, and the actual CLI path for a no-peer FIFO. The CLI control
uses a separate process with a predeclared ten-second test deadline. Its complete
command, stdout, stderr, exit and observed output absence are retained in
[limits-fifo-cli.json](2026-10-05-cumulative-work-budgets/run-02/limits-fifo-cli.json).
The traceback establishes rejection inside the new initial reader, before
hardware checking or model work. The temporary FIFO was retained through the
observation, then removed with its owned temporary directory; no special file
was left in the repository.

## Current measured verification

The [current verification JSON](2026-10-05-cumulative-work-budgets/run-02/verification.json)
contains 63 passing tests, actual test-process exit 0, unchanged before/after
source hashes, raw failure controls, full-panel accounting and a separately
launched exact recovery. Unittest reports 13.358 seconds; its process took
15.3390 seconds. The fresh recovery process exited 0 in 2.3063 seconds.
The two focused new controls separately passed in 0.665 seconds before the
frozen collector.

The model remains the original randomly initialized CPU FP32 seed-1818 Qwen3
fixture with six updates and actual activation checkpointing. Original-reference
weights, Adam, RNG, history and masks are unchanged. The exact numerical-state
digest is still
`7ed7c945c62c1fd26ae2edee889eb41362468bbeee9ae26e0061883999aff255`.
The recovered ledger still charges seven update reservations and marks six
updates completed; a new output cannot refill the later failed attempt.

Current frozen identities are:

- DPO runner: `227e9ab352779d8bc3a08ec7d0d62d34933b8f0bf9bddf3b35a8d4594ab4d0a8`.
- DPO budget tests: `ca15302b2d6b4b9d6b4e0970eb53647212f54b551cc2e8ae5f4d7720f3562c47`.
- Hardening protocol: `86386a8706056d90079df1d28852480daaf9100cd8d5d009511672e89912cd3a`.
- Unchanged ledger: `7483a6e20fa3a48fb39502862fd1fad45b14ea183ae47ba1f4aea49d7d60d297`.
- Verification JSON: `30177bbb6392fadf04331e4f1c9952e15907b41c7eeba3e767a064ad43060a3e`.
- Raw FIFO CLI record: `b367364b68fe5128c0b1d94343a8cc88a0fe8b38de864f41124b49a07001cdf4`.

The changed source produces a new scientific contract digest,
`1e4982d819b215734955dc4c5b195ef4ab1befb40885d6f271bb69760b3c04bd`;
it does not pretend to be the historical contract.

## Scope remains bounded

This is an initial trusted-local CLI input gate, not hostile filesystem
containment for every later provenance-file access. The work ledger remains
cooperative logical accounting rather than a physical quota. No pretrained
weights, GPU/CUDA run, installation, service, Git action or external campaign
occurred. Other-runner integration, physical enforcement, pretrained/BF16
compatibility, model-scale overhead and quality evidence remain pending. Current
hardening is measured without retroactively changing historical claims.
