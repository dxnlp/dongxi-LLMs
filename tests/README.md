# Tests

Tests verify reusable course implementations, numerical examples, tensor shapes,
and failure conditions independently of notebook execution.

Use the isolated `course` extra and `uv.lock` described in
[AppendixD](../book/appendices/d-reproduction-and-environments.md). The test
suite includes tiny random local HF/tokenizer/adapter fixtures, so Torch alone
is not sufficient; no pretrained weights or live API calls are acquired.
The CPU acceptance orchestrator keeps actual exits/logs even after a failure
and separately runs the five-notebook panel. A passed test suite does not imply
every notebook, Mac kernel or pretrained campaign passed.
The [historical runner-boundary report](../experiments/reports/2026-10-05-runner-boundaries-readiness.md)
records838 passing tests, including actual SFT/RLVR work, DPO snapshot reservation
and actual-encoding fixture DPO caps at its own revision. The
[historical semantic-validation report](../experiments/reports/2026-10-05-semantic-validation-readiness.md)
records884 passing tests,1260math expressions/76routes and six selected fresh
notebooks43cells/19images/four preserved skips. SFT/DPO strict16/19-dimensional
budgets charge each runner-semantic check; generic pre-callback reading and
save work were pending at that revision.161 executable hashes stayed unchanged.
Not all76 rerun.
The [historical shared-I/O report](../experiments/reports/2026-10-05-snapshot-io-readiness.md)
records 971 tests, 1269 math expressions/76 routes and six selected fresh
references with 44 executed cells/20 images/four preserved skips. 170 executable
hashes stay stable. Separate I/O9 preserves SFT16/DPO19; legacy teaching APIs
stay explicit v1, accounted snapshots v2. The initial missing-kernel-path run is
retained. Supply `JUPYTER_PATH` pointing to the existing temporary kernel prefix's
`share/jupyter` directory when invoking the collector; do not reinstall a kernel
to repair discovery. This is neither all76 rerun nor physical/external evidence.
The [historical native-RLVR report](../experiments/reports/2026-10-05-rlvr-io-readiness.md)
records998tests/1291math/76routes, six selected fresh references45executed cells/
21images/four preserved skips and174 stable sources. Its21new/102existing-shared/
6lesson panels overlap the full suite. Original23 caps and saved pre-semantic
prefix remain unchanged; pending fresh replay never recollects before applying.
Native publication failures and earlier developmental candidates remain retained.
Day25 now has10source/reference cells8images, plus an execution-only identity
preamble. Root rechecks663files/774relationships/88receipts; no all76current,
Mac/hosted, physical or pretrained result is inferred.
The [current story-work report](../experiments/reports/2026-10-05-story-work-readiness.md)
records1037tests/1294math/76routes and179 stable executable hashes. Seven selected
fresh references execute50cells/23figures, preserving four unfinished skips;
this includes the separate Day9 lesson, not all76 notebooks. Its33new/47existing/
6lesson component counts overlap the suite; four actual fresh-process continuations
preserve original equations and failed spending. Story byte-I/O and physical/
external gates remain pending. LearnerDay9 and13of18 completion are unchanged.
The existing isolated-CPU [acceptance orchestrator](../scripts/verify_semantic_validation_integration.py)
retains actual exits/logs/memory samples, never launches pretrained work.
Cooperative logical accounting
is not an OS quota or a real launch backend. Earlier failed checks and their
source identities are retained rather than relabeled as current passes.

Book math regression checks run with the ordinary test suite. They scan every
Markdown file under `book/` for the project's known GitHub rendering hazards.
For the standalone source check and optional MathJax rendering check, see
[math formatting](../docs/MATH_FORMATTING.md).
