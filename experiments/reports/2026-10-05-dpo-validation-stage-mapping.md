# DPO fixture mapper: model work and recovery checks have separate budgets

The current-source bounded CPU reference passes. Its nineteen-dimensional cap
file is consumed by the actual DPO reader and recovery contract. The original
fourteen model-work limits are identical to the historical v1 fixture; five
explicit semantic-validation dimensions now cover the declared checkpoint and
inspection schedule. This is fixture evidence, not production approval.

## Evidence and history

The [premeasurement protocol](../specs/2026-10-05-dpo-validation-stage-mapping.md)
fixes the recipe, layout observation, cadence and diagnostic-load count. The
current [run-03 raw reference](2026-10-05-dpo-validation-stage-mapping/run-03/verification.json)
has SHA256 `04c99671be0592abc9bc89fc2f99c8ceab8520609ebc9d43d408ead2e1e7e8ab`.
It records 23 mapper tests and 19 preparation/compiler tests, both exit zero
(unittest 8.761 and 4.922 seconds; child times 10.858 and 4.971 seconds).
The separately retained actual fresh-process continuation exits zero in 2.743
seconds. Collection takes 20.771 seconds, and all 50 pinned source bindings are
unchanged before/after. Raw logs, input bytes, tokenizer/interface, model-layout
observations, snapshots, journals and full generated responses remain in its
exclusive directory.

Earlier passing candidate collections remain untouched:
[run-01](2026-10-05-dpo-validation-stage-mapping/run-01/verification.json), SHA256
`13e289b5971929527f9d9585c8046001899c9bcd1f81bb7cb22249b86c18b880`, and
[run-02](2026-10-05-dpo-validation-stage-mapping/run-02/verification.json), SHA256
`13c9aa690283cdb6f3ae2134c5596bab2b8c47cb53c0ee1fb566dcb6dd27f7f1`.
They are historical because later runner interface/layout/metadata refusals
changed source identities, not because their measured trajectories were repaired.
The [v1 report](2026-10-05-dpo-stage-budget-mapping.md) and its artifacts remain
historical and byte-preserved. Three development failures are retained separately:
[live/JSON comparison](2026-10-05-dpo-validation-mapping-first-diagnostic.json),
[typed-container contract refusal](2026-10-05-dpo-validation-mapping-container-diagnostic.json)
and [capacity versus actual cursor](2026-10-05-dpo-validation-mapping-cursor-diagnostic.json).

## What the mapper derives

The pure module imports no model, tokenizer or Torch. It binds supplied actual
encoded records and observed architecture metadata, without authenticating their
supplier. The fixture independently reloads its saved tokenizer/template,
re-encodes real inputs and compares actual model/optimizer/configuration metadata
before consuming the mapped caps. Unknown shapes/dtypes, unsupported schemas,
stale inputs/source/interface, split collisions, invalid masks and altered
schedule vectors refuse. Rehashing an altered requirement is not enough to hide
an omitted validation panel. Approval requires a new exact full-vector fixture
receipt; extra per-dimension allowances are not accepted.

The actual random local model has P=2880 state-dict elements, O=2880 optimizer
parameter elements, n=14 parameters and R=5056 CPU RNG bytes. No CUDA RNG states
are present. One semantic panel reserves `3P + (3O+n if k>0 else 0) + 4R`,
or 28864 tensor-element units at cursor zero and 37518 at a positive cursor.
These are declared verifier-envelope units, not FLOPs, byte counts or measured
CPU instructions.

For the primary two-update fixture with cadence two, commit cursors are zero and
two. One independently declared final diagnostic callback adds the third panel.
The fresh envelope is therefore 3 operations, 4 history rows, 103900 tensor
units, 6 RNG states and 16 sampler replay draws. All other limits match the
historical fourteen-dimensional fixture exactly.

For capacity two, a resumed attempt includes load callback, restore validation,
a new invocation's commit at its retained boundary, later commits and its final
diagnostic callback. Componentwise maxima over the possible declared durable
boundaries give the conservative envelope. Combined capacity is 8 operations,
12 history rows, 265528 tensor units, 16 RNG states and 48 replay draws.
Different dimensions may have different worst-case boundaries; this is not one
measured trajectory, retry permission or an invocation-count enforcement rule.
The original training-update allowance remains four, not a new larger recipe.

## Actual numerical and recovery controls

The fixed scientific recipe remains seed 1818, two updates, accumulation four,
beta 0.1, learning rate 5e-7, decay 0.01 and clip one. It uses the original
sixteen-token local WordLevel interface and random one-layer Qwen FP32 CPU
model, with activation checkpointing enabled, frozen reference and cache-off
training. Independent original equations reproduce sampler choices exactly;
maximum absolute differences are policy `4.657e-10`, Adam `3.725e-9`, gradient
norm `8.941e-8`, loss zero and reference zero, all within 1e-6 tolerance.

A separate cadence-one control commits zero, one and two and includes the final
diagnostic callback. Its actual uninterrupted schedule exactly fits its mapped
limits. In the retained interruption, update one is durable and the next forward
fails. A fresh process reopens the same physical journal, charges both callback
and restore checks, continues only the remaining update, and reaches the exact
policy/reference/Adam/history/counter/Torch/sampler-RNG endpoint SHA256
`58cac828d6197548917dad9c7465e22af8dea19d3e482b44a6e1f3092be5761f`.
The failed attempt remains charged; three training updates were attempted,
although independently declared full-attempt capacity was four.

The separate zero-cursor recovery retains its later failed update and reserves
all four allowed training updates. A further whole update refuses before sampler
or forward work. Actual generated samples, including incorrect or cap-truncated
ones, are retained without gold repair. Their content is not an English-quality
or preference-learning result.

The real 512-position pilot encoding still requires 817600 policy-plus-reference
training positions rather than the historical single-branch geometry 204800.
The old coarse vector still refuses; no pilot fit or production launch occurs.

## Scope and remaining gates

Execution is Linux ARM64 CPU, Python 3.12.14, Torch 2.14.1+cpu, Transformers
5.18.0 and tokenizers 0.23.2, offline with hidden CUDA and one numerical thread.
The final mapper source is `6897c8d3663666750fafb77a41bda90a40c6ba2a032b7469ef43e518da517e1c`;
tests are `5283e0962206becbb1963fb374fafbb62f457aa9c3c8953cfca0ea324d52f45a`.
The source manifest changed only by pinning new protocols/tests. The raw record
binds actual DPO `ef6e366d…` and SFT `c0a5287f…`, not earlier candidates.

Generic shared-loader deserialization/tree/finite scans before the runner
callback remain excluded. Serialization, arbitrary reports/export/child writes,
full-process compute and physical containment are not claimed bounded here.
Live production tokenizer/architecture reproduction, supplier authentication,
authorization/backend/compiler integration, pretrained/CUDA/BF16, Mac and hosted
execution remain pending. All external campaign outcomes remain unfilled; no
model-scale job, installation, service, Git action or learner advancement occurs.
