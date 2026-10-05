# Original DXI-01–08 acceptance review

Date:2026-10-05. Scope: read-only acceptance, source and course-material review,
not a new numerical verification panel or a package-status change.

## Conclusion

No concrete unmet original acceptance criterion was found in the completed
DXI-01,02 and04–08 packages. This is a scoped conclusion about their named
deliverables, not a claim that the course has completed every planned training
maximum, demonstrated general model improvement or assessed learner mastery.
DXI-03 remains in progress: it has substantial actual branch evidence, but
successful mechanisms, partial campaigns and evaluation instruments do not
substitute for its remaining native comparisons and final integration.

The eight live acceptance arrays and eight dependency arrays exactly equal the
[retained original baseline](2026-10-05-native-base-profile/original-acceptance-before-integration.json).
The observed [ledger](../../docs/course_improvements.json) statuses were
complete for01,02,04,05,06,07,08 and in progress for03. No criterion or dependency
was removed or weakened during this review.

| Package | Original criteria | Original dependencies | Review conclusion |
| --- | ---: | --- | --- |
| DXI-01 |4 | none | Original interface/provenance scope supported |
| DXI-02 |5 |01 | Frozen grading/replay and actual behavioral evidence supported |
| DXI-03 |6 |01,02,04,17 | In progress; remaining native comparison/integration gates retained |
| DXI-04 |4 |02 | Bounded constructive learning and capability controls supported |
| DXI-05 |4 |02,04 | Selection and actual candidate-cost controls supported |
| DXI-06 |4 |02,05 | Refinement state machine and controlled failures supported |
| DXI-07 |5 |01,08 | Text/outcome/process reward mechanisms supported |
| DXI-08 |5 | none | Offline preference/judge audit supported |

## DXI-01: identity is an interface, not a tensor shape

Criteria1–3 are supported by the [identity implementation](../../src/dongxi_llms/run_identity.py),
actual SFT/DPO/RLVR integrations and the retained positive/negative identity
checks. The interface binds token-to-ID mapping, encoding rules, special and
stop IDs and the template; same-sized permutations and changed rules are not
accepted merely because embedding shapes match. Declared revisions remain
separate from observed file hashes. The journal retains partial preflight
identity and failure stages. Legacy adoption is an explicit migration rather
than invented historical provenance.

Criterion4 has actual native evidence, not only a categorical or random-model
fixture. The [independent criterion map](2026-10-05-native-acceptance-criteria-independent.json)
and [review](2026-10-05-native-acceptance-independent.md) retain the full and
LoRA20-update checkpoint/recovery identities and exact fresh10-to20 replay.
The original BF16 merge check failed its declared tolerance. A separately
specified FP32 merge and fresh reload passed; that does not retroactively make
the BF16 check pass. The criterion asks for explicit full/merged compatibility,
not universal equivalence in every precision.

The mechanism is integrated into [Chapter1](../../book/chapters/01-evidence-before-optimization.md),
the SFT/DPO/GRPO chapters and labs, and the
[checkpoint-interface notebook](../../notebooks/day-01/04_checkpoint_interface.ipynb).
The notebook supplies adjacent answers and a same-size broken interface, not
merely a manifest-count demonstration.

## DXI-02: a reproducible score still has a behavioral boundary

Criteria1–2 are supported by the frozen item/source/split/parser/decoding/stop
contract, raw response ledger, error and truncation denominators, slice reports
and source-group paired uncertainty. Criteria3–4 are supported by the
[bounded mathematical grader](../../src/dongxi_llms/reasoning_evaluation.py):
extraction, equivalence and task validity are distinct; fraction/decimal,
boxed-answer, set/interval, unit, ambiguity and unsupported fixtures retain their
declared behavior. It does not execute model text or silently invoke an
unbounded symbolic solver. The original strict integer task remains separate.

Criterion5 has both authored positive/negative controls and actual pretrained
responses. The [actual Base/short-SFT replay](2026-10-05-pretrained-evaluation-replay.md)
retains30 outputs from the frozen15-item panel. Base has0/15 rubric successes;
the disposable full-SFT20 checkpoint has1/15, with both models capped on every
response. The independently reviewed phrase-rubric pass is visibly not a clean,
finished helpful answer. The review is an unblinded AI review, not human
agreement or a broad safety benchmark. The exporter consumes existing evidence;
its own non-generation boundary does not erase the producer's actual model
events.

These distinctions are developed in [Chapter7](../../book/chapters/07-evaluation-is-a-contract.md),
[worked answers21–22](../../book/solutions/07-evaluation-is-a-contract.md),
the [lab](../../book/labs/07-evaluation-is-a-contract.md) and the
[grading/replay notebook](../../notebooks/day-10/04_mathematical_grading_and_response_replay.ipynb).
The later120-item assistant publication and story ratings are different
instruments; this review does not pool their scores with the15-item panel.

## DXI-04–08: substantive mechanisms and preserved counterexamples

- **DXI-04, criteria1–4:** the [constructive-control report](2026-10-04-reasoning-controls.md)
  retains actual unsaturated sampled autoregressive learning across three
  predeclared seeds, fixed oracle/frozen/constant controls, original task-family
  and held-out source/template slices, invalid/cap observations and protocol
  distinctions. Two seeds still fail the known-solvable zero-sum item;
  apparently perfect held-out slices match a constant1 policy. The prior negative
  GRPO result remains unchanged. [Chapter13](../../book/chapters/13-group-relative-policy-optimization.md)
  and its [visual notebook](../../notebooks/day-23/03_reasoning_tasks_and_positive_controls.ipynb)
  distinguish base/instruct weights from thinking-mode presentation and do not
  claim faithful rationales or family transfer.

- **DXI-05, criteria1–4:** the [selection report](2026-10-04-inference-selection.md)
  and [implementation](../../src/dongxi_llms/inference_selection_lab.py) retain
  nested N=1,2,4,8 pools, gold-blind selection, ties, duplicate and invalid rules,
  oracle separation, generation/rescoring work, measured bounded CPU costs and
  equal-attempt/token comparisons. A five-wrong-versus-three-correct majority
  and worsening success at higher caps contradict monotonic-benefit claims.
  [Chapter7](../../book/chapters/07-evaluation-is-a-contract.md) and the
  [candidate notebook](../../notebooks/day-10/05_actual_candidates_and_answer_selection.ipynb)
  explain availability versus the one answer actually selected.

- **DXI-06, criteria1–4:** the [refinement report](2026-10-04-critique-revision.md)
  retains drafts, critiques, proposals, decisions, delivered states and failures,
  including harmful and confidence-only transitions. Exact programmatic
  controls match serialized budgets in162/162 cases; actual-pool replay retains
  only202/324 exact matches,135 overrun and165 underfill tokens. Those are not
  relabeled compute equality. The [independent replay](2026-10-04-critique-revision-root-acceptance.json)
  and [notebook](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb)
  explain no-critique and matched-token controls. Neural self-critique remains an
  optional protocol, not a missing literal acceptance requirement.

- **DXI-07, criteria1–5:** the text reward mechanism has retained endpoint,
  EOS/padding, pair-swap, gradient and reload checks. Its separate
  [character intervention](2026-10-04-text-reward-character.md) detects encoded
  overlap before fitting without replacing the historical word-arm failures.
  All three character seeds report held-out ranking0.5 and poor terminal/process
  transfer. Crossed final-answer/step labels separate outcome from process
  supervision, and complete frozen exports are linked to the fitted-reward
  policy control. [Chapter10](../../book/chapters/10-preferences-and-reward-models.md),
  [solutions](../../book/solutions/10-preferences-and-reward-models.md) and the
  [text-reward notebook](../../notebooks/day-16/03_text_reward_and_process_labels.ipynb)
  retain the synthetic-versus-human boundary and linear shortcut baseline.

- **DXI-08, criteria1–5:** the [preference audit](2026-10-04-preference-audit.md)
  retains484 simulated judgments, separately authored reviewed labels, raw
  verdicts/errors, source identities, swaps, repeated order effects, verbosity
  and injection controls, disagreement and explicit sibling/source weighting.
  Authored reviewed labels are not called human feedback. The
  [hardening report](2026-10-04-preference-audit-hardening.md) preserves adversarial
  parser failures and exactly replays the original ledger. [Chapter10](../../book/chapters/10-preferences-and-reward-models.md)
  and the [judge notebook](../../notebooks/day-15/03_preference_collection_and_judges.ipynb)
  supply adjacent explained references without paid/live/private collection.

## DXI-03 remains a separate empirical obligation

Actual evidence now includes native full/LoRA profiling and replay, first400
story training and its192-continuation rating panel, and the distinct assistant
400-update publication. The [story comparison](2026-10-05-native-story-first400-comparison.md)
does not claim completion of14,000 updates or the missing later checkpoints.
The [DPO deadline report](2026-10-05-native-dpo-recovery-deadline.md) preserves
the failed fresh-resume invocations, durable intermediate outputs and unknown
native exit rather than calling them accepted recovery.

The currently inspected [campaign assembler](../../scripts/assemble_native_campaign_evidence.py)
pins the original45 declarations, retains DPO replay01/02 as unselected
histories and selects the predeclared03 path. Missing actual receipts remain
missing; execution-only CPU/hash-worker settings are not a scientific
intervention or an increased quota. This is a source review, not a run of the
assembler or proof that replay03 has passed. Base→SFT→DPO/chosen and
Instruct→RLVR remain different branches. Their outstanding comparison and final
integration gates cannot be closed by this acceptance review.

The original DXI-17 dependency is preserved. Its actual Linux execution gate is
distinct from honest unverified Mac/hosted scope; an unavailable Mac is not a
reason to claim an otherwise permitted Spark experiment did not occur.

## Review identities and limits

The point-in-time ledger SHA256 was
`e6c770026614edbd88b155da248a342f4b4d20f99924bd0630f74dd41052ecf3`;
the original baseline SHA256 was
`6cb9caee1d87b36f4a1ae6a9fbb5ef36889ad9f6e198cf822aeb11ec5bf2be2b`.
The independently read native criterion map SHA256 was
`2c0a004f45f07b776f0ffb89163d40160f14622469a2e9525138c184f5d2b5bd`;
the actual behavioral-review JSON SHA256 was
`f5aea1502b52f8c812d1bedf29026be0c9e3bbd2e705cc4453c91dcb27b8866e`.
The inspected assembler SHA256 was
`e9262b247415cc5a55d030038ad92588caf9aa640f42ee23d0e85d89819cad0a`.

The04–08 reviewer checked applicable retained module/notebook hashes as well as
the implementations, chapters, worked answers and failure interpretations.
Historical numerical reports establish their recorded source revisions;
subsequent source edits do not automatically inherit those measurements.
This review performed only file reading, small-file hashing and literal JSON
comparison. No tests, model loading, fits, forwards, downloads, GPU jobs,
assembly, Git writes or hosted runs were performed. It does not reassess09–18,
advance the learner from Day9, invent human/Mac evidence or declare the entire
eighteen-package goal complete.
