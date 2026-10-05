# Final current-source course verification

Actual Linux ARM64 CPU verification passed on October 5, 2026. This is local
execution evidence, not a Mac run, hosted CI result, learner assessment or new
model experiment.

| Check | Actual result | Retained receipt |
| --- | --- | --- |
| Complete test suite | 1,533 tests, `OK`; test-runner interval 215.829 seconds, enclosing command 219.122586 seconds | [CPU manifest](2026-10-05-final-goal-cpu/run-01/cpu-verification.json), [test log](2026-10-05-final-goal-cpu/run-01/tests.log) |
| Full notebook reference set | 76 passed, no failures; 539 source code cells, 535 executed reference cells, 210 image outputs | [Full76 manifest](2026-10-05-final-goal-notebooks/run-01/manifest.json) |
| Preserved learner work | Four unfinished Day3 exercise cells, indices 6, 17, 24 and 31, intentionally skipped | Full76 manifest and original notebook |
| Final selected references | Five fresh passes after model work completed | [Selected manifest](2026-10-05-final-goal-cpu/run-01/notebooks/manifest.json) |
| Book math | 54 Markdown files, 1,294 expressions, zero issues | [Math log](2026-10-05-final-goal-cpu/run-01/math.log) |
| Navigation/schema | 15 chapters, 15 solutions, four appendices, 76 notebooks; zero problems | [Route log](2026-10-05-final-goal-cpu/run-01/routes.log) |
| Executable source stability | All 238 Python/workflow identities unchanged; no added or removed Python sources | CPU manifest and independent final original-criteria review |

The isolated CPU Python is
`/tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python`, version 3.12.14;
the temporary-prefix kernel is `dongxi-course-clean`. The dependency lock and
course-manifest identities are retained in both manifests. CUDA was hidden,
Hugging Face network access disabled and CPU thread counts set to one.
No package installation or shared GPU-environment modification was performed.

The full76 reference set was deliberately run separately from the final test
orchestrator. Some references overlapped bounded G8 evaluation; this is not an
isolated timing benchmark. The test suite waited for model processes to finish
because its subprocess boundary tests use model-runner names. Full76 verifier
time was 176.065534 seconds. The final CPU orchestrator then passed tests, math,
routes and five selected references; it is not falsely labeled a second all76
execution. Both executions use the same frozen executable and notebook sources.

The full reference invocation selected every route through the existing course
manifest, including optional and extension lanes:

```python
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, "src")
from dongxi_llms.course_manifest import load_manifest, select_notebooks

manifest = load_manifest(Path.cwd())
paths = [row["path"] for row in select_notebooks(manifest)]
subprocess.run([
    sys.executable, "-B", "scripts/verify_course_notebooks.py",
    "--kernel", "dongxi-course-clean",
    "--output", "experiments/reports/2026-10-05-final-goal-notebooks/run-01",
    "--expected-prefix", sys.prefix,
    "--host-reserve-gib", "25",
    "--notebooks", *paths,
], check=True)
```

The recorded output path is historical evidence; use a fresh output path for a
future independent run. These are reference executions, not completed learner
exercises. The learner remains on Day9. Mac portability, hosted CI, actual
inline-host use, media production and publication remain separately unverified
or unexecuted.
