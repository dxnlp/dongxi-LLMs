# Cumulative DPO work limits survive numerical recovery

The frozen CPU controls pass: 61 tests, actual process exit 0, and an independently
launched process resumes update 3 to update 6 with exactly the same numerical
state as the uninterrupted run. The recovered budget includes a deliberately
failed forward charged after the checkpoint. Changing output directories does
not provide new capacity. This establishes cooperative logical-work accounting
in this DPO source, not a GPU, pretrained, physical-quota or whole-campaign result.

The [premeasurement protocol](../specs/2026-10-05-cumulative-work-budgets.md)
declared the recipe, dimensions, normal caps, failure controls and evidence scope.
The exact collector output is [run-01/verification.json](2026-10-05-cumulative-work-budgets/run-01/verification.json).
Its raw [test output](2026-10-05-cumulative-work-budgets/run-01/tests.json),
[fresh-process execution](2026-10-05-cumulative-work-budgets/run-01/fresh-execution.json),
and [retained failures](2026-10-05-cumulative-work-budgets/run-01/retained-failures.json)
are available separately. The initial development panels passed 16 journal,
13 DPO, 32 combined, and 28 unchanged earlier DPO/checkpoint controls; these are
not substituted for the final frozen 61-test collection.

## What the budget means

Before drawing a training example, the runner reserves the complete accumulated
update using componentwise maxima over the frozen encoded branches. It does not
sample first, truncate a completion or reduce accumulation to fit the remaining
allowance. Validation reserves a whole panel before either network runs. Greedy,
single-sequence, uncached generation reserves the declared cap and growing-prefix
positions before its first forward. A forward wrapper refuses an unexpected
logical input geometry before the backend call.

Reservations permanently consume capacity, including unused early-stopping
allowance. Completed work, successful partial work in failed operations, entered
work and uncertain entered-call upper bounds are separate. Selected supervision
counts identify the actual masks encountered; they do not claim a failed forward
successfully scored those targets. Logical calls and positions are not FLOPs,
backward/checkpoint recomputation, GPU time or a physical memory quota.

The uninterrupted training-only control records the following difference between
conservative capacity and measured successful work:

| Dimension | Reserved | Successfully completed |
| --- | ---: | ---: |
| Updates | 6 | 6 |
| Draws / sampled examples | 12 | 12 |
| Valid completion targets | 72 | 62 |
| Policy positions | 96 | 86 |
| Original-reference positions | 96 | 86 |

The separate full collector also charges baseline generation, final validation
and final generation. Its totals therefore include 76 reserved versus 66
completed targets and 130 reserved versus 111 completed policy positions. These
different scopes are explicitly retained in the JSON, not compared as the same
training-only denominator. Its two generation requests reserve eight output
tokens and return six, with stop IDs included.

## Recovery, failed work and raw records

The model checkpoint binds the ledger UUID, physical file identity, immutable
limits and an exact hash-chained prefix. Before model state application, recovery
validates that prefix against the active journal, then reads every later charge.
It does not rewind the budget to the model cursor. The deliberate post-checkpoint
failure entered one policy call at four logical input positions but returned no
output. One draw and five selected targets are known; internal failed-call FLOPs
are not. After successful fresh-process recovery, seven update allowances remain
charged while six updates are completed. The original reference, Adam moments,
sampler/Torch RNG, complete history and numerical state match the uninterrupted
six-update result exactly.

The negative controls also establish refusal before active sampling or a network
call when the whole operation cannot fit, including a new-output recovery with
exhausted update allowance. A new/copied journal or changed caps is rejected
before state application. Baselines are charged again in new invocations.
Validation or export failure cannot remove the final committed model snapshot or
the later spent work. Reservation fsync failure prevents work and poisons the
live journal instead of allowing an unrecorded operation.

Generation records preserve actual returned IDs and stop metadata even when text
decoding fails. When the model forward itself fails, raw output IDs remain
unknown; the entered and successful input geometry is retained separately. The
declared greedy uncached dispatch is an operational contract, not a claim of
unchanged arbitrary cached decoding or measured recomputation cost.

## Filesystem and production interface

The retained journal is a single-writer flocked, fsynced, bounded JSONL file. It
requires owner UID, mode 0600, a single link and a private owned final directory.
Ancestors are opened no-follow using retained directory descriptors; FIFO
targets are rejected nonblocking. Every append/snapshot checks live ancestry,
pathname, mode, inode, size, same-inode raw bytes and authoritative accounting
state. Unlink/replacement, copied journals and accidental public-map mutation are
refused. Invalid or incomplete tails stay intact; there is no destructive repair.

Production CLI requires `--work-limits PATH` and
`--work-journal-max-bytes INT`. New runs default to their new private output
directory's `work-ledger.jsonl`; `--work-journal PATH` can explicitly select an
owned private location. Resume additionally requires the same physical journal
through `--work-journal`, alongside the existing independent snapshot digest,
byte size and scientific contract. Old unbudgeted CPU references remain an
explicit optional API, not the production bounded-job claim.

This is trusted-local cooperative accounting. It cannot defeat privileged
rollback, arbitrary malicious code or a hostile rename/check/syscall race. It is
not portable by copying a journal to another filesystem/machine. Artifact-byte
reservation is a separate package; this runner does not claim that all outputs,
other runners or a campaign are physically contained.

## Frozen identities and remaining gates

The final test process took 15.0263 seconds; unittest reports 12.964 seconds.
The separately measured fresh process exited 0 in 2.3394 seconds. Execution used
Python 3.12.14, Torch 2.14.1+cpu and Transformers 5.18.0, with offline environment
flags and no visible CUDA devices. The fixed original random seed-1818 model,
six updates, masks, beta, optimizer and actual activation-checkpointing recipe
were unchanged. No model acquisition, GPU run, installation, service, Git or
external campaign occurred.

Source SHA256 values are:

- DPO runner: `22856562cf8242e5c027f672df6f1ecb44466550c5ff2e184fa102fc8022e805`.
- Work ledger: `7483a6e20fa3a48fb39502862fd1fad45b14ea183ae47ba1f4aea49d7d60d297`.
- Journal tests: `0843e8b6b256be4708cd5495d9a544b9d3bd47fbbb05de85b2aff574032cb68f`.
- DPO work tests: `1cabfc9caba4f4c1a66a9e0a06ab9072d3052b2c931c0b85737aec1c6d5b313d`.
- Protocol: `4d7ed13787b2457d0059a6530629e3d632845fd5cade311eaa21fa6ed1b545a5`.

Before/after identities match. The scientific contract digest is
`0770828180f07ba90384e0c71b0f1fcd4b50d05de83364ba3ec554a72738efe8`;
the exact numerical-state digest is
`7ed7c945c62c1fd26ae2edee889eb41362468bbeee9ae26e0061883999aff255`.
The verification JSON digest is
`419cc5e368beced211c2409d76745ad5bf7109e0e32ffae86e7f70bf163518d6`.
Historical DPO/checkpoint reports keep their earlier source identities rather
than being rewritten as current-source measurements.

CUDA/BF16 and pretrained compatibility, other-runner integration, independent
physical enforcement, model-scale accounting overhead and language quality remain
pending. The bounded journal is rescanned and accounting state rehashed for
cooperative tamper detection; production-scale cost has not been profiled. No
external campaign row gains an outcome from these CPU controls.
