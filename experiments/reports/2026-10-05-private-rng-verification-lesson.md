# Private sampler checks without changing the training stream

This extends the existing Day25 recovery notebook, not the course inventory or
learner day. The [protocol](../specs/2026-10-05-private-rng-verification-lesson.md)
was saved before implementation. The helper is an original bounded CPU generator
microscope; it does not validate a real checkpoint or establish model quality.

Five helper tests pass, actual exit0, unittest0.002s: exact private replay,
unchanged live/global RNG, intentionally consumed live state, zero history,
determinism/independent returns and bounded exact integer inputs. The fresh
selected [notebook manifest](2026-10-05-private-rng-verification-lesson/notebook-run-01/manifest.json)
records eight executed code cells, six images, no skipped cells or failures,
2.549956s total and117.804573GiB minimum **pre-execution sampled** MemAvailable.
Its actual kernel is the existing isolated Linux ARM64 Python3.12.14 environment,
Torch2.14.1+cpu, CUDA unavailable. The verifier process exited0. This is one
selected notebook, not an all76-notebook run or a Mac execution.

For seed1818, three completed updates and two scalar draws per update, observed
history and private replay are both `[4,0,1,0,0,2]`. The correct next source IDs
are `[1,2,4,1,3,4]`. Checking with a separate generator preserves that future.
The intentionally broken checker consumes those six live draws and then returns
`[1,2,2,4,3,4]` as the future. State equality distinguishes true replay from
individual sampled IDs that match by chance. The live state hashes before and
after private verification are both
`a7f4420780e3b60b9298044a823d34a44e8954286bcef423488313cfa645fd8f`;
after broken verification the hash is
`5c465c6aad85582423b3030e2f999cc202281a732242b3322df6180a2ec084e6`.

Each private verification still performs six real scalar draws. These counts are
not tokens, FLOPs, CPU instructions or a physical work quota. The deterministic
SFT selector does not receive invented RNG draws. The shared reader's earlier
deserialization/tree/hash operations still require the
[separate pre-read integration](../../docs/SNAPSHOT_INSPECTION_BUDGET_PLAN.md).

Only the sixth image was extracted from this actual fresh execution into the
new reference asset; the five existing preview files were not overwritten.
The image was visually inspected. Its SHA256 is
`8b8b2351379083a11a876fc3c8a3d4ffe2a6a5452dc7970de02da13b240b1b33`
(100507bytes). Chapter14 questions/solutions29–30 and the lab connect the RNG
example to the trust dependency that precedes a runner callback. The animation
candidate remains an uncommissioned Mac-side proposal, not a rendered output.

Source identities for this run:

- helper: `d2d14c47aa458a3ad8a735202fc509f081971c876abe152fbb00d2de258af739`;
- tests: `f993b041db93868d6f9a5bc1ef72ac8711d390c3a8a970cb2d1b3fb210a573db`;
- protocol: `66e9c4ab0f0f2e2b5441475b8e3ae6c53b628d5e3ebd8ad74ad964fbed9102f8`;
- notebook: `8384c132dc4c26eb18b3c42afc8a95ab3fce06c1ae5786f1420db5b0e5eb582f`;
- manifest: `b6b8dcce2103f83fc9b14d7c0eb6a5fe98ee7960b5a3dd0fcd6f5bc0874d3838`.

No acquisition, pretrained/GPU job, installation, persistent service, platform
write, Git write, media rendering or publication. Package status13of18, active
learnerDay9 and all45 null external model outcomes are unchanged.
