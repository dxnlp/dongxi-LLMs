# SFT work limits remain spent after numerical recovery

The current frozen CPU collection passes 64 controls, with test-process exit 0,
and four separately launched full or LoRA processes reproduce the uninterrupted
four-update numerical state exactly. Each recovery retains a failed forward
charged after the update 2 checkpoint. Five update allowances remain reserved
while four updates complete. These results establish cooperative logical work
accounting in the actual SFT source, not pretrained, CUDA, BF16, physical quota,
whole-campaign or language-quality evidence.

The [premeasurement protocol](../specs/2026-10-05-sft-cumulative-work-budgets.md)
declared seeds, modes, objective, caps and controls. Two narrow addenda bind the
[exact parsed cap bytes](../specs/2026-10-05-sft-cap-byte-binding.md) and their
[bounded identity and closing reader](../specs/2026-10-05-sft-bounded-cap-identity.md).
The current acceptance is [run-02/verification.json](2026-10-05-sft-cumulative-work-budgets/run-02/verification.json),
with [raw test output](2026-10-05-sft-cumulative-work-budgets/run-02/tests.json).
The test process took 19.2710 seconds; unittest reports 18.405 seconds.

## What is reserved and measured

Before advancing the seeded cyclic cursor, the SFT loop previews the entire next
accumulation window. It reserves its exact shifted valid-target count, selected
examples, unpadded logical tokens and padded forward geometry. These cyclic
selections are not new random sampler draws. The original full and LoRA losses,
single label shift, assistant and ending-token masks, Adam updates, dropout,
activation checkpointing and clipping are unchanged. Refusal does not shorten
data, supervision, accumulation or the objective.

Whole NLL and generation panels are reserved before their first network call.
SFT generation preserves its original cached greedy dispatch with
`use_cache=True`, including the LoRA base-model dispatch. Its reservation uses a
conservative uncached growing-prefix upper bound; successful work records actual
cached input geometry. For seed 1212, uninterrupted training plus baseline and
final observers reserve 302 policy positions but successfully use 200. The two
generation panels reserve 144 growing-prefix positions but use 42 cached input
positions. This difference is retained allowance, not a refund or a FLOP saving
measurement. Stop IDs remain included in returned output-token counts.

Each of the twelve dimensions is independent and overlapping. Logical calls and
positions exclude backward and activation-checkpoint recomputation; they are not
physical GPU cost. Completed work, successful partial work, entered-call work
and uncertain entered-call upper bounds remain distinct. A failed backend call
does not count selected labels as successfully scored model targets.

## Recovery and negative controls

Before model, Adam or RNG state application, a budgeted snapshot validates the
scientific contract, immutable caps, same physical journal and retained exact
prefix. Later failed spending is read from that journal rather than rewound to
the numerical cursor. A new output directory cannot refill an exhausted ledger;
new or copied journals and changed caps cannot apply the old snapshot. Baselines
are charged again for each invocation.

| Seed | Mode | Fresh exit | Fresh seconds | Reserved updates | Completed updates |
| --- | --- | ---: | ---: | ---: | ---: |
| 1212 | Full | 0 | 2.1116 | 5 | 4 |
| 1212 | LoRA | 0 | 2.2472 | 5 | 4 |
| 1213 | Full | 0 | 2.2227 | 5 | 4 |
| 1213 | LoRA | 0 | 2.2958 | 5 | 4 |

All four arms match weights, original frozen LoRA weights, Adam, order and cursor,
Python and Torch RNG, masks and full numerical history exactly. Their raw
receipts, journals, interruption errors and subsequent entered-forward errors
are retained under each arm directory. The failed calls have 19 entered logical
positions for seed 1212 and 11 for seed 1213, but no returned output. Internal
failed-call computation remains unknown.

Focused controls cover refusal before cursor/collation/forward; poisoned partial
updates; whole observer-panel refusal; reservation persistence failure; and
snapshot, metric, baseline, final-observer and export failure. Completed
snapshots and already-spent work survive these failures. Raw generation records
retain returned IDs and stopping metadata when decoding fails; when the backend
itself fails, unknown output IDs remain unknown instead of fabricated empty text.

This is not a quality result: all eight final authored development generations
reach the four-token cap and fail exact match. Their NLLs are preserved, but
slightly lower losses do not establish instruction following or language transfer.

## Cap files and production interface

Production requires `--work-limits PATH` and `--work-journal-max-bytes INT`.
New runs use a private owned journal, optionally selected with `--work-journal`;
resumes require the same physical journal through that flag, plus the existing
independent numerical receipt, byte bound and scientific contract. The optional
old unbudgeted CPU API remains backward compatible. Two earlier CLI test fixtures
were updated only to supply valid required cap arguments; their prior assertions
and numerical recipes remain intact.

Cap reads are bounded to 64 KiB, nonblocking, no-follow and regular-file checked,
with fd-relative ancestor traversal. Their initial exact bytes enter the science
contract. Caps are excluded from generic input hashing and instead recorded as a
`bounded_work_limits` role at both identity observations, followed by an enclosing
digest recomputation. The same bounded reader verifies the role before export
and completion. A no-peer FIFO CLI control exits 1 in 0.6616 seconds without
creating output. Deadline-bounded regular/FIFO replacement controls before
identity collection and at closing retain their actual exceptions; no tokenizer
or model loader is called in the pre-load controls. See the
[raw identity controls](2026-10-05-sft-cumulative-work-budgets/run-02/bounded-cap-identity-controls.json).

The shared ledger's owner/private-mode, no-follow ancestry, single-writer lock,
hash-chain and retained-tail rules are reused without changes. This is
trusted-local cooperative protection, not resistance to privileged rollback or
an arbitrary hostile check-to-syscall race. Copying a journal to another machine
does not preserve its physical identity. Artifact, log and export byte
containment remains a separate unimplemented SFT integration.

## Current and historical identities

Execution used the isolated offline CPU environment: Python 3.12.14,
Torch 2.14.1+cpu, Transformers 5.18.0 and PEFT 0.20.0. The original randomly
initialized tiny Qwen3 recipe uses seeds 1212 and 1213, full and rank 2 LoRA,
four updates, accumulation 2, learning rate 0.003 and attention dropout 0.1.
No pretrained acquisition, GPU, installation, service or Git mutation occurred.
Existing read-only Git identity probes were retained in the older preflight
tests with the root agent's explicit approval.

Current SHA256 identities are:

- SFT runner: `046c94d2a2c7ec0b80614a53a1525154e25185accad8090dacbbe7e1cae18c26`.
- SFT work tests: `bf87f36d764261ed82fac8d596bf0b66e558903b3a1be251baba668beeba1689`.
- Shared work ledger: `7483a6e20fa3a48fb39502862fd1fad45b14ea183ae47ba1f4aea49d7d60d297`.
- Current verification: `a8e438654c4223cf5de57f474eea80ddfeffadfff9f8464dbab74bddbba1c8b1`.

All recorded source hashes agree before and after `run-02`. The earlier
[run-01](2026-10-05-sft-cumulative-work-budgets/run-01/verification.json) passed
61 controls and four replay arms under runner digest
`2986884c63425a0edd9dd43f7d8be0fbe9e809a65ca453f0a02576c0985e0cd4`;
it remains historical and does not claim the later generic-reader identity fix.
The four numerical digests match across the two collections; only the scientific
source contracts change with the hardening. Earlier SFT recovery reports keep
their own historical identities.

Pretrained and CUDA/BF16 compatibility, model-scale overhead, physical quotas,
SFT artifact/export containment, external campaign outcomes and independent
quality evidence remain pending. The bounded journal is rescanned for cooperative
tamper checks; its overhead has not been profiled at model scale.
