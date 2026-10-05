# Complete draw accounting for the private-replay lesson

The [first lesson run](2026-10-05-private-rng-verification-lesson.md) remains
unchanged at its original helper revision. Independent review found an omitted
comparison cost: the expected-future generator also makes six actual draws.
The pre-rerun protocol addendum adds that bucket and a total without changing
the seed, any returned source-ID sequence, state hash, figure or assertion.

The corrected helper records36 scalar `torch.randint` calls: six each for
retained training history, private validation, safe future training,
independent expected-future comparison, intentionally broken live verification
and its later training. A new wrapper spy directly compares the returned total
and sum of buckets with all actual calls. Six helper tests pass, exit0,
unittest0.002s. This complete **microscope** count is not the DPO runner's
validation count: the teaching control deliberately includes independent and
broken branches that a valid production check would not execute.

The separate corrected [fresh notebook run](2026-10-05-private-rng-verification-lesson/notebook-run-02/manifest.json)
exited0, eight executed cells/six images/no skips/no failures,2.551446s total,
117.235222GiB minimum pre-execution sampled MemAvailable. The actual isolated
CPU kernel/environment is unchanged. The existing sixth preview remains the
actual identical plot extracted from run01; it was not regenerated or relabelled
as a new image. Its displayed draw values remain unchanged.

Corrected helper SHA256:
`2f572e033259a3b592b0a8972a86ea7861eb72a644dbe14c1838021b8d15f8af`.
Tests SHA256:
`8b7eb474baa39c6b8c8d02b7f24d031fb2678283195517b4653dce7bf8e6d795`.
Run02 manifest SHA256:
`fcd2e436aea7e9de94c534142e68fa579dfe1334011ed0a1fc795ab1d9b85700`.
The notebook source is unchanged from run01; the imported helper is the
corrected dependency. The final integration panel independently reruns this
same selected notebook and rehashes its executable dependencies.

The reader's pre-callback work, physical resources, pretrained outcomes and
actual Mac/media/hosted execution remain outside this CPU lesson. No course
status, active day or external result was advanced by the correction.
