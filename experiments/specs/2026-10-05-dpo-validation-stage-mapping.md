# DPO fixture validation-stage mapping v2

Freeze this protocol before new tests or numerical controls. Preserve the v1
protocol, source captures, failures, raw results and approvals byte-for-byte.
This continuation changes only the fixture mapper, its tests and manifest-only
source pins. It does not authorize a production run or authenticate a supplier.

## Contract and fixed recipe

Requirements, approvals and mappings explicitly use v2. The consumable cap file
contains the actual runner's nineteen exact dimensions: the original fourteen
remain unchanged, followed by `recovery_validation_operations`,
`recovery_history_rows`, `recovery_tensor_elements`, `recovery_rng_states` and
`recovery_sampler_draws`. Old schemas, missing/extra dimensions, bool aliases,
zero/insufficient allowances and individual extra-cap overrides refuse.

Keep the existing original sixteen-token WordLevel tokenizer, byte-bound chat
template and split-disjoint two training pairs, one validation pair and one
generation prompt. Keep seed 1818, one-layer Qwen width 16/intermediate 32,
two query heads/one KV head/head dimension eight, context 512, FP32 CPU, zero
dropout, activation checkpointing on, frozen reference/cache off, two updates,
accumulation four, beta 0.1, AdamW learning rate 5e-7/decay 0.01 and clip one.
Baseline/final generation each has cap 64, with natural stops and cap-truncated
or incorrect responses retained. No filtering/truncation fits an allowance.
Retain original-equation comparison with absolute and relative tolerance 1e-6.

## Observed architecture and one panel

Before fitting, observe the actual randomly initialized local model's policy
state-dict shapes/dtypes, ordered optimizer parameter shapes/names, effective
policy/reference configuration hashes and CPU/CUDA RNG layouts. Strictly bind
these observations and their canonical hash alongside actual encoded records,
input bytes, tokenizer interface and encoder source. The pure mapper imports no
model, tokenizer or Torch; its supplied metadata is not authenticated. Unknown,
oversized or malformed layouts refuse before multiplication/traversal.

Use the separately specified runner estimator exactly. P is state-dict elements
including aliases; O is ordered optimizer parameter elements; n is parameter
count; R is CPU RNG bytes; C is CUDA RNG bytes. At completed cursor k one semantic
panel reserves one operation, k history rows, k times accumulation sampler
draws, two plus CUDA count RNG states, and
`3P + (3O+n if k>0 else 0) + 4R + C` logical tensor-element units. These are
verifier-envelope units, not FLOPs, bytes, CPU instructions or physical quotas.
Generic shared-loader deserialization/tree/finite scans before its callback,
metadata traversal, serialization and ledger integrity remain excluded.

## Exact bounded schedule

Require explicit positive bounded `checkpoint_every` and
`diagnostic_loads_per_attempt` (one for this fixture, at most sixteen). Let U be
the independently prepared maximum updates and A the explicit complete-attempt
capacity (one or two in numerical controls; at most four). Commit cursors are
sorted unique `{0, positive cadence multiples through U, U}`: a periodic final
cursor is not committed twice. A fresh attempt reserves these commit panels plus
the declared final diagnostic load callbacks. A resumed attempt at durable k
reserves a callback load and duplicate restore validation at k, a new invocation
commit at k, all future commits and its final diagnostic callbacks. Componentwise
maximum across declared durable cursors gives a conservative resumed envelope;
different dimensions can have different worst-case cursors. This is capacity,
not a measured trajectory, retry permission or invocation-count enforcement.

The original fourteen whole-attempt requirements are still multiplied by A
without adding training work. New validation requirements are fresh envelope
plus `(A-1)` times resumed envelope. Numerical primary cadence remains two;
a separately labeled cadence-one control proves initial, periodic and final
panels without changing training. The actual final diagnostic load occurs before
capturing the final journal summary. Restore charges do not refund spending.

## Controls and evidence

Test independent arithmetic against the actual runner estimator, all nineteen
insufficient/zero dimensions, old-schema rejection, metadata/layout/type/shape
refusal, schedule bounds and whole-panel pre-work refusal. Test actual consumable
reader plus recovery contract, full schedule, original equations, same-journal
completed and interrupted continuation, retained failures and no added training
capacity. Numerical policy/reference/Adam/history/RNG replay remains unchanged.
Each diagnostic load count is explicit, rather than a universal default count.
Retain the actual 512-position pilot encoding boundary: old single-branch
204800 geometry cannot approve 817600 four-branch training positions. No pilot fit.

Use the existing isolated CPU interpreter, hidden CUDA, offline mode and one
thread. After owner/source coordination freeze, collect focused mapper and
compiler tests and the bounded actual reference in a new exclusive directory,
with before/after source hashes, actual commands/exits, full generated responses,
snapshots, work journals, architecture metadata and refused cap vectors. Preserve
any initial failing attempt separately. No downloads, pretrained model, GPU,
Mac, installation, services, Git, external outcomes or learner advancement.
