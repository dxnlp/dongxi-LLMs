# Foundation and Saved-Run Companion Verification

Executed on 2026-10-04 under the [predeclared specification](../specs/2026-10-04-course-foundations.md).
Scope: nine new CPU/offline lessons for Days 1, 2, 9, plus independent reusable
function checks. No pretrained weights, credentials, downloads, GPU computation
or service startup were required.

## Declared references and observed outcomes

The Day1 pathway checks canonical identity and independent smoke acceptance
criteria, then analyzes the actual historical TinyStories JSON. Key-order changes
leave a canonical identity unchanged; changing a declared control changes it.
Exit0 does not override a failed memory reserve or missing/nonfinite observed loss.

English-only learned BPE merges retained `数` as UTF-8 IDs
`[230,149,176]`, and decoding reconstructed the original character. A 20-copy
`数据库` corpus with eight learned merges encoded the nine input bytes as one
new token 263. All 256 base byte pieces remained available for unseen Unicode.
The frequency tie rule is deterministic. This educational whole-document encoder
does not claim production regex/special-token/normalization equivalence.

The embedding lesson reproduces repeated-row lookup gradients, tied versus untied
output paths, and positive prompt-embedding gradients from response-only loss.
Those are graph properties of the implemented CPU fixture, not a semantic claim
about particular vectors in a pretrained model.

Document-isolated causal masking for IDs `[0,0,1,1]` permits the rows
`[1,0,0,0]`, `[1,1,0,0]`, `[0,0,1,0]`, `[0,0,1,1]`.
Future sources and earlier independent documents are excluded. An EOS alone
does not enforce this boundary.

The saved-run budget analysis independently recomputes:

| Quantity | Observed/recomputed value |
|---|---:|
| Valid targets / processed positions | 0.2129253932 |
| Valid targets / summed update seconds | 4249.2658 targets/s |
| Valid targets / completed run seconds | 4035.3900 targets/s |
| Target presentations / prepared training targets | 0.1250034789 |

These clocks use actual stored 2026-09-14 measurements. The two throughput values
have different timing boundaries; neither is a new GPU throughput benchmark.
The exposure fraction is not an assertion that a random fraction of the corpus
was uniformly covered.

## Verification route

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_course_foundations.py
python scripts/verify_course_notebooks.py --days 1 2 9 --kernel dgx-spark-native --export-figures
```

Five foundation tests passed. All nine notebook references passed fresh-kernel
execution, producing nine meaningful figures and preserving source cells.
The final course-wide manifest identifies final source and figure hashes;
earlier temporary manifests describe their exact earlier revisions. Source:
[course_foundations.py](../../src/dongxi_llms/course_foundations.py);
[Day1](../../notebooks/day-01/README.md), [Day2](../../notebooks/day-02/README.md),
[Day 9](../../notebooks/day-09/README.md).

The course verifier samples host `MemAvailable` before each notebook against a
25 GiB floor. This is not a continuously sampled memory peak or proof of no
between-sample spikes. Reference readiness does not certify learner mastery.

## Limits and preserved evidence

TinyStories fixed-development prediction improved in the historical run; complete
samples still contain malformed language, repetition and role inconsistency.
The notebook uses a declared structural checkpoint grid and marks retrospective
selection. It does not establish reliable story coherence, a unique cause for
repetition, a memorization audit, replication across seeds or a controlled
trained comparison. No new model-quality result was produced in this build.
