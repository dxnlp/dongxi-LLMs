# Book Editorial Improvement Plan

Status: revised proposal; editorial implementation not started. Prepared and
reviewed 2026-10-10 from a read-only audit of all 15 chapters, 16 labs,
15 solution guides, front matter, appendices and the
76-notebook inventory at commit `7f5175b`. Line numbers below refer to that
revision and will drift as edits land; re-locate by section heading. This revision
resolves review findings about extraction boundaries, phase gates, evidence
provenance, notebook contracts and checkpoint ancestry.

This plan is editorial. It changes how the completed teaching draft reads; it
does not add chapters, change the 28-day route, launch Spark work, alter any
measured report, rewrite historical evidence, advance the learner beyond Day 9,
or authorize publication. It is separate from the completed eighteen-package
reference-audit upgrade in [COURSE_IMPROVEMENT_PLAN.md](COURSE_IMPROVEMENT_PLAN.md).

## 1. Diagnosis

Coverage is complete and the derivation cores are strong. The weaknesses are
editorial and were introduced mainly while production evidence was being
integrated during October 2026.

| # | Finding | Evidence |
|---|---|---|
| F1 | Operations/log prose interrupts the teaching narrative | Largest candidates: Ch 14 §14.8 and Ch 13 §13.8.3; also Ch 9, Ch 6 §6.5/6.12/6.18/6.20–6.21, Ch 7 §7.10–7.17, Ch 5 §5.15 tail, Ch 11 §11.7 tail + §11.8.3, Ch 15 §15.10, Ch 1 l.234–250 and l.457–494. These passages mix operational detail with essential concepts; review before extracting |
| F2 | Glued number/word tokens in both directions (`all20`, `in61.636`, `0.45correct`, `11controlled`) | Confirmed examples across chapters, labs and solutions; no existing prose lint catches them. The original total of 481 conflicts with its listed file counts summing to 602. No validated total is claimed; Phase 0 establishes the scanner and baseline |
| F3 | Labs are link lists or runner manuals rather than learner routes | Thin: Labs 1–4, 8, 10, 12 (30–80 lines, no checkpoints). Manuals: Lab 9 (253 lines; sixteen SFT caps, `--resume-io-receipt`), Lab 11 (299; budget keys), Lab 7 (254; CLI/settings JSON). Lab 12 is titled "Chapter 12 — Policy Gradient Laboratories" |
| F4 | SFT/RL chapter presentation underserves decision D005 (SFT/RL deepest) | Ch 12 §12.9 lacks its existing measured REINFORCE/baseline/RLOO/PPO table, figure and update code. Ch 9 lacks a loop listing and representative Base/Full400/LoRA400 answers. Ch 8 refers to visualizations without displaying them. Implementations and evidence already exist in modules, notebooks and reports; $\beta$ is defined in Ch 11 and needs restating in Ch 12 |
| F5 | Stale present-tense status and repeated evidence obscure scope | Ch 11 l.281–283 says the executed preference comparison remains pending; Ch 13 l.501–502 still gates its now-measured baseline. Ch 9 l.322 needs historical context for its proposed campaign. Ch 6 and 7 story tables overlap but contain different measures; Ch 15 §15.10 repeats campaign numbers. Distillation overlaps §9.12–9.13 and Ch 15. Ch 1 l.41 says three Day 1 notebooks; four exist |
| F6 | Notation drift | See §4 below |
| F7 | Missing chapter scaffolding | No outcomes/prerequisites block in Ch 6, 7, 8, 9 (Ch 4 l.25–51 is the model); no H3 structure in Ch 7, 8; Ch 8 transition at l.127 precedes four sections; Ch 5 organized by Day labels (l.11–19, table l.615, l.1239); "teacher forcing" used at Ch 3 l.128, defined at l.761; "validation" vs "development" split naming inconsistent inside Ch 6 |
| F8 | Uneven notebook contract visibility | Heading scans flag Days 2, 8, 24, 26–28 for predictions/checkpoints and Days 17–19, 21, 26–28 for reference solutions. Some already use prose "Prediction", "Reference explanation", bold "Solution", or explanations in code. Normalize visibility and inspect actual adjacency; missing headings do not establish missing teaching |

First extraction targets: Ch 13, 14, 9 and 6; then Ch 7, 11, 15, 5 and 1.
This is a qualitative editorial priority, not a measured contamination score.
Phase 0 records counts using one reproducible definition. Counts locate review
work; chapter clarity and preserved reasoning determine whether an edit succeeds.

## 2. Phases

All implementation and verification use the isolated CPU course environment and
local checks. Preserve measured reports, specifications, archives, model cards
and learner work. Reuse current implementations and verification tools before
adding a narrow missing check. Time estimates below are planning estimates;
reassess them after Phase 0 inventories the actual work.

Common gates: math and navigation/integrity checks pass; mechanical prose checks
remain clean after Phase 0; narrative findings outside the phase's declared
targets do not regress; the diff stays within its scope.
Run focused checker tests when checker behavior changes and fresh notebook
references when notebook content or its dependencies change. A prose-only move
does not require another full numerical run. Phase 5 performs the complete final
local verification; intermediate zero targets apply only where stated below.

Implementation surfaces are `book/`, editorial checks in `scripts/` and `tests/`,
notebook Markdown, and necessary runbook/navigation/tracking documents in `docs/`,
`BOOK.md` and `PROGRESS.md`. Conditional index/candidate updates are limited to
`LEARNING_MEMORY.md` and `visuals/animations/PROPOSALS.md` when the tracking rule
below applies. New verification receipts may be written to a new
editorial report directory under `experiments/reports/`; old evidence is immutable.
Reusable model/training modules remain canonical and outside this editorial edit.

### Phase 0 — Guardrails (≈0.5 day)

- Add a narrow `scripts/check_book_prose.py` with two rule groups:
  - **Mechanical prose checks:** `book/chapters`, `book/labs`, `book/solutions`,
    `book/front-matter` and `book/appendices`. Detect both
    letter-to-digit and digit-to-letter joins, including decimal cases and
    `exit0`. Exempt valid names, units and identifiers such as `Qwen3`, `GPT-2`,
    `BF16`, `FP32`, `sm_121`, `utf8` and version strings. Exclude fenced/inline
    code, math and link targets from automatic spacing repair; inspect visible
    link labels and ordinary table prose. Literal commands, configuration keys,
    tensor identifiers and intentionally broken examples must remain exact.
  - **Narrative review checks:** conceptual chapter prose. Locate operational
    task/run IDs, policy dates, session-status language ("the learner", "does
    not advance", authorization/approval), and machine-specific absolute paths.
    These are review candidates, not proof that a passage should be removed.
    Permit necessary evidence captions, quoted examples and defined identifiers
    through documented contextual exemptions. Appendices and runbooks retain
    legitimate provenance, commands and operational boundaries.
- Record scanner version/source hash, paths, exclusions and counting units:
  number of matches and number of affected lines separately, both overall and
  per file. Preserve the before-repair baseline in the new editorial report.
  Maintain a small explicit review/exemption list with a reason for each item;
  do not clear findings through blanket per-file exclusions.
- Register it beside `check_book_math.py` in Appendix D's verification list
  and in `scripts/run_cpu_verification.py` if that panel exists for prose.
- Repair F2 using a reviewed prose-only pass; inspect every changed number,
  table and identifier. Keep literal code and historical evidence unchanged.
- Acceptance: mechanical prose findings are zero after justified exemptions;
  narrative findings have a recorded baseline and no regressions. Focused
  checker tests cover true joins, decimals and legitimate exclusions. Math and
  integrity checks pass. Phase 0 implementation edits stay in `book/`,
  `scripts/`, `tests/`, plus the new editorial verification report/receipts and
  narrow `BOOK.md`/`PROGRESS.md` milestone pointers.

### Phase 1 — Extract operations from narrative (≈2–3 days)

Extend **Appendix D — Reproduction Commands and Environment Locks** with a
"Checkpoints, Recovery and Job Supervision" section. Keep conceptual explanation
there and put exact runner commands in the existing guides or Phase 3 runbooks.
At each origin retain the mechanism, a useful example and an evidence link;
the passage length follows the teaching need rather than a fixed sentence quota.

Appendix E is a fallback only if D becomes unwieldy. Adding it requires deliberate
updates to `scripts/check_course_integrity.py` (currently four appendices),
`BOOK.md`, `docs/COURSE_BLUEPRINT.md` and navigation links. Appendices are not
registered in `docs/course_manifest.json`; no schema change is needed solely
for E. Historical four-appendix reports stay unchanged.

| Origin | Keep in chapter | Move |
|---|---|---|
| Ch 1 l.234–250, l.457–494 | all required success criteria; eight-ID checkpoint example; token semantics versus shape; what hashes can prove | detailed recovery incidents, manifest fields and genealogy changelog |
| Ch 5 §5.15 l.844–933; ex. 31–34 | ragged logical positions, validity/causal masks, EOS/PAD distinction, compact K/V shapes, row compaction, cache identity and work-versus-latency example; saved weights versus saved continuation | loader/digest and training-recovery mechanics, mainly l.914–933; retain ex.31–32, split 33, relocate operational 34 with a Ch 13 pointer |
| Ch 6 l.157–199, l.461–508, §6.21 l.835–898 | update-boundary budget enforcement, refusal without state change, logical versus physical work; exposure/schedule/development-NLL evidence for the matched 400 intervention; concise stopping/ending contrast | source-guard, journal prefix and tranche/receipt implementation |
| Ch 7 §7.10–7.17 | evaluation contracts, oracle versus selected accuracy, likelihood versus truth, paired/blinded comparisons and distinct real-model cases in 7.16 and 7.17 | ledger fields, CLI and invocation detail |
| Ch 8 §8.12 | 42-vs-46-target exposure example | collector lock/crash recovery |
| Ch 9 §9.6 tail l.174–186, l.287–305, §9.11 | checkpoint identity table l.153–158; one sentence on BF16 merge rounding | snapshot format, fsync, merge receipts, runner description |
| Ch 11 §11.7 l.223–249, §11.8.3 l.425–469 | why original-reference identity defines the DPO objective; truly matched draws/exposure; §11.8.4 table, figure and `purple pouch` example | persistence, recovery chronology and worker-witness telemetry |
| Ch 13 §13.7.1 l.331–380, §13.8.3 l.504–725 | pending rollout as an existing observation, old-policy/reference distinction, stop-inclusive masks and collected/applied work; interface/cap tables; unsupported-extraction example; §13.8.4 result and zero-advantage/KL lesson; six completed/two failed initial conditions, 680/800 observed and 120 missing | supervisor, cleanup-gate, exit-code and timing receipts; shorten substantially without a hard 60-line cap |
| Ch 14 §14.8 l.463–868 | incident-versus-cause reasoning, generation cost, replay of the pending update, logical versus spent work, bounded-update-versus-bounded-job distinction; aim for roughly 80–120 lines if these remain clear | state-digest versions, hash workers, buffers and save-schedule implementation |
| Ch 15 §15.10 l.500–621, §15.11 l.651–668 | genealogy diagram and a worked defense (Phase 2); material failure/coverage distinctions | redundant campaign summaries and logging receipts |

Also in this phase:

- Screen exercise ranges Ch 1 ex.8–9, Ch 5 ex.31–34, Ch 6 ex.19–21,
  Ch 7 ex.14–23, Ch 9 ex.17–19, Ch 13 ex.18–25 and Ch 14 ex.18–35.
  Retain questions about mechanisms, equations, masks, objectives, sampling,
  controls, coverage and warranted conclusions. Move only tasks requiring
  particular CLI commands, journals, receipts or runtime gates to an
  evidence-reading section in the appendix or matching solutions guide.
  Split mixed questions and preserve exercise IDs, solution mappings and links.
  In particular, Ch 7 ex.14–20/22, Ch 11 ex.14–15 and Ch 13 ex.18–21 stay core;
  split Ch 7 ex.21/23 and Ch 13 ex.22–25 rather than relocating them wholesale.
- Fix actual stale status: Ch 11 l.281–283, Ch 13 l.501–502 and Ch 1 l.41;
  contextualize Ch 9's proposed-campaign wording at l.322. Retain the valid
  principle at Ch 9 l.324–325 and Ch 13 l.401–405 that tooling is not execution.
  Also review Ch 5's model-scale recovery status at l.928–930 against later
  measured recovery. Later story checkpoints/full schedules and optional
  protocols remain unrun wherever their own evidence is absent.
- Keep Ch 6's unique development-NLL/exposure/schedule evidence. Make Ch 7
  the home of the detailed five-axis story rubric and reviewer disagreement;
  replace repeated rating detail in Ch 6 with a concise contrast and pointer.
- Make Ch 15 the single home for response-level distillation; reduce
  §9.12–9.13 to a forward pointer and keep exercises 13–16 there only if
  they stay answerable from Ch 9 content.
- Acceptance: no unresolved narrative-review findings in the selected chapter
  passages after contextual review; necessary evidence language is explicitly
  justified. Every relocation has a working origin link and exercise mapping.
  Each retained claim remains traceable to canonical unchanged evidence; removing
  duplicate numbers does not require copying them again into the appendix.
  A before/after review confirms that mechanisms, negative results, coverage and
  genuinely unrun stages have not disappeared. Math and integrity checks pass.

### Phase 2 — Standard evidence block and depth (≈3–4 days)

Define one evidence-block pattern and apply it wherever a Spark or CPU
campaign result enters a chapter:

> reader prediction → settings and comparison controls → results →
> representative outputs → interpretation → limitations

Label newly added predictions as reader exercises before the reveal. Historical
pre-run hypotheses are a separate evidence claim and require the original
specification; do not invent a preregistered prediction of the observed winner.
Use a stated example-selection rule (such as the first aligned held-out item),
retain stopping/cap information and label any shortened output. For numerical
microscopes, tensors/gradients serve as outputs; for genealogy, use ancestry and
evidence links. The pattern is a teaching guide, not six mandatory boxes in every
section. Preserve fair comparison boundaries and all source limitations.

Apply to: Ch 6 §6.20–6.21 (story 400), Ch 9 §9.9 (full vs LoRA; add actual
prompt/answer triples for Base, Full400, LoRA400 from the retained raw
records), Ch 11 §11.8.4 (closest already; add the reader prediction), Ch 13
§13.8.4 (show an actual retained G8 training group, responses/rewards/advantages
where recorded; label unavailable fields instead of inventing a complete dump),
Ch 15 §15.11 (a worked one-page defense following its own l.645–646 chain).

Depth additions, all from existing reports and notebooks:

- Ch 12: measured REINFORCE/baseline/RLOO/PPO comparison table and the
  learning-curve figure from `experiments/reports/2026-10-04-preference-policy-cpu.md`
  with a reader prediction; a concise PPO/RLOO update listing drawn from
  `src/dongxi_llms/policy_gradient_lab.py` (ratio, clip, detach). Preserve all
  three seeds, matched rollout exposure and unequal gradient work (480 PPO
  passes versus 160 for the other arms); this is a finite policy mechanism,
  not a generalization or equal-compute ranking. Restate Ch 11's $\beta$
  convention and align response/token advantages with Ch 12's existing GAE
  derivation. Preserve distinct advantage definitions and gradient boundaries.
- Ch 9: a readable SFT loop and token-weighted accumulation listing in §9.4–9.5
  drawn from `src/dongxi_llms/sft_lab.py`. Use the aligned raw responses linked
  by `experiments/reports/native-assistant-comparison-20261005-run-02/README.md`;
  show answer content together with EOS versus cap outcomes.
- Ch 8: embed existing Day 11 target-ownership, packing-visibility and token-share
  figures next to their mechanisms, with shape/axis labels and reading guides.
  The ownership image is not a combined ID/owner/label plot; provide its IDs
  and labels as adjacent text rather than claiming the existing figure shows all.
- Ch 4: convert plain-text math (l.71–73, 165, 176) to LaTeX; add a
  three-position worked numerical forward pass at §4.7.
- Ch 3: insert a reader prediction box before the §3.8 result table.
- Ch 15 §15.10: replace repeated summaries with a genealogy diagram, one brief
  explanation and evidence link per edge. Show separate branches:
  - September random initialization → historical DongxiGPT 14,000 updates;
    separately, fresh October initialization → control400 and half-LR400,
    with later declared checkpoints explicitly unrun.
  - Acquired Qwen3-0.6B-Base → full-SFT400 → chosen-only-SFT100 and DPO100;
    sibling Base → LoRA400 adapter → independently verified FP32 merged export.
  - Acquired pinned Instruct weights → G4/G8 RLVR policies, each 16 iterations.
    Thinking modes and generation caps are interface/evaluation annotations,
    not weight ancestors. Retain G4's failed supervision and separate export
    closure alongside G8's accepted pilot. Recovery artifacts are gates, not
    pilot parents; draw no local Base→Instruct or assistant→RLVR training edge.
- Acceptance: listed measured cases supply prediction, comparison settings,
  results, appropriate outputs, interpretation and limitations; diagrams and
  forward passes use the suitable subset. Every measured number/output links
  to a retained report or raw record; calculations are independently reproducible
  and explicitly labelled. Chapter snippets agree with the canonical modules
  and relevant existing tests/references. No new experiment or outcome is needed.

### Phase 3 — Labs as learner routes (≈2 days)

Lab template:

1. Header: machine, expected time, prerequisites, one-sentence deliverable.
2. One row per notebook: *question / prediction prompt / checkpoint with
   expected value or invariant / one intervention / what it cannot prove*.
3. Deliverable (run card, defense, diagram).
4. Collapsed footer: verification command and link to the runbook.

Models already in the repository: Lab 13 l.1–40, Lab 15 l.7–17, Lab 5's
route table, Lab 6b §1–4.

- Rebuild Labs 1, 2, 3, 4, 8, 10, 12 (thin) and 7, 9, 11 (manuals). Rename
  Lab 12's title to "Lab 12 — …".
- Move detailed CLI, journal and resume instructions to existing guides or
  focused runbooks under `docs/runbooks/` (for example `sft_spark_runner.md`,
  `dpo_spark_runner.md`, `evaluation_tools.md`, `rlvr_runner.md`), linked from
  the lab footer and Appendix D. Reuse adequate existing documentation; this
  directory does not exist yet. Keep scientific controls, comparison contracts
  and relevant token limits in the lesson with their meaning explained.
- Match checkpoint values to the actual notebook fixture, seed, dtype and
  tolerance; use qualitative invariants where a numeric target would mislead.
  Reference examples:
  - Lab 3 cross-entropy 0.6108643 and logit gap 0.8473:
    `experiments/reports/2026-09-03-next-token-distribution.md`.
  - Lab 4 stale-cache difference 1.80691:
    `experiments/reports/2026-09-05-attention-gradients-cache.md`.
  - Lab 7 pass@3 = 0.5333 for the fixed n=10, c=2, k=3 example:
    `experiments/reports/2026-10-04-evaluation-and-sft-course.json`.
  - Lab 2 notebook checkpoints use educational-BPE byte coverage, round trips
    and embedding-gradient invariants. The pinned Qwen counts 9 < 11 < 20
    belong to `experiments/reports/2026-08-30-qwen3-multilingual-tokenization.md`
    and may be a separately labelled report-reading checkpoint, not a claimed
    expected result of the educational tokenizer.
- Acceptance: every registered notebook in each lab's route has a prediction,
  checkpoint, intervention and evidence boundary; each lab names a deliverable.
  Aim for ≤120 body lines; document exceptions needed for clarity. Runner flags
  live in the footer/runbook. All lab/notebook/solution links resolve, and values
  are traceable to their matching fixture. No lesson is reduced to a link list
  to meet a line quota.

### Phase 4 — Coherence pass (≈2 days)

- Expand `book/front-matter/notation.md` and Appendix B into a binding symbol
  table (see §4) and reconcile every chapter against it.
- Add outcomes/prerequisites blocks to Ch 6, 7, 8 and 9 modelled on Ch 4
  l.25–51. Ch 1 and 2 already have outcomes: add missing prerequisites and
  improve placement without duplication. Add H3 structure to Ch 7 and 8;
  move Ch 8's transition to the end;
  relabel Ch 5 Day 5/6/7 structure as Parts A (baseline), B (modern variants),
  C (defense); define teacher forcing at first use in Ch 3; use "development
  split" per Ch 7's definition throughout Ch 6.
- Replace absolute paths (Ch 2 l.819–834 and elsewhere) with Appendix A/D
  references.
- Reconcile notation definitions, formulas and tensor shapes together. Preserve
  code API names and original report symbols with an explicit local mapping;
  a notation pass must not transpose weights or change an objective.
- Notebooks: normalize existing prediction/checkpoint/reference content to
  `### Prediction`, `### Checkpoint`, `### Reference solution` in Markdown
  across all 76. Retain code-cell source, IDs, metadata, outputs, execution
  counts and learner attempts unchanged. Add adjacent Markdown explanation
  when the existing reference explanation is only a code comment.
- Extend `scripts/verify_course_notebooks.py` or add a narrow static check for
  heading coverage and exercise/reference ordering. Review every exercise for
  prediction before reveal and an immediately adjacent runnable reference plus
  mechanism explanation; one heading per notebook alone does not establish this.
  If a runnable solution is actually absent, record the gap and affected route
  rather than presenting a heading change as a solution. A code-cell addition
  requires an explicitly amended scope before that route can be signed off.
- Acceptance: shared symbols follow §4, local symbols are introduced with
  unambiguous meanings, and shapes/layouts agree with the implementation.
  All 76 pass heading/ordering checks and exercise review; notebook preservation
  hashes pass. Fresh references pass for touched routes on the final source;
  the Phase 5 all-notebook run may satisfy this without duplicate execution.

### Phase 5 — Verification and record (≈0.5 day)

- Use `scripts/run_cpu_verification.py` and `scripts/verify_course_notebooks.py`
  for one complete final local run: prose checks with documented exemptions,
  `check_book_math.py`, `check_course_integrity.py`, notebook heading/ordering
  and preservation checks, all 76 fresh references and the unit suite. Use the
  isolated CPU environment and a new output directory; retain source hashes,
  host/environment identity and actual command outcomes. No hosted CI is added.
- Write `experiments/reports/<date>-book-editorial-pass.md` with command
  outcomes and before/after counts (F1 vocabulary, F2 glued tokens, lab line
  counts, notebook heading/adjacency coverage), scope exceptions, relocation and
  exercise mappings, and the exact final revision/source hashes. Failures and
  unavailable evidence remain recorded; old receipts are not updated to this run.
- Add a short entry to `PROGRESS.md` and a pointer in `BOOK.md`. No learner
  position change; no publication. Acceptance requires the final commands to
  pass, zero unresolved mechanical findings across book prose, and zero unresolved
  narrative-review findings across all 15 chapters after justified contextual
  exemptions. Complete a manual reader-flow review of the affected chapters/labs,
  including representative equation, figure and notebook previews.

Recommended order for fastest reader-facing gain: Phase 0 → Phase 1 (Ch 13,
14, 9, 6 first) → Phase 3 (Labs 9, 11, 12 first) → Phase 2 (Ch 12, Ch 9 first)
→ Phase 4 → Phase 5.

## 3. Per-chapter worklist

The line counts are the original audit snapshot, not reduction quotas. Apply
Phase 1's exercise-screening rule and Phase 2's evidence pattern throughout.

| Ch | Audit lines | Primary fixes |
|---:|---:|---|
| 1 | 567 | Extract operational recovery/changelog detail while retaining success criteria and tokenizer identity; §1.10 principle + eight-ID example; fix three/four notebook count; add prerequisites; screen ex.8–9 |
| 2 | 982 | Retain output-vocabulary/decoder-coverage boundary while shortening status detail at l.693–709; Ch 13 pointer; replace absolute paths; improve outcomes/notebook placement; align vocabulary/width notation |
| 3 | 1017 | Reader prediction before §3.8; define teacher forcing at first use; lab checkpoints matched to the actual fixture |
| 4 | 681 | LaTeX for inline math; three-position worked forward; drop unused $N_{vocab}$; lab checkpoints on invariants |
| 5 | 1535 | Parts A/B/C; preserve ragged-cache mechanisms and ex.31–32; extract §5.15 recovery tail, split ex.33 and route operational 34; vocabulary/value distinction; update obsolete status while retaining time-stamped primary-source frontier context; shorten lab inventory and move operational recovery instructions |
| 6 | 955 | Outcomes/prerequisites; extract operational detail, retain two work clocks and recovery principle; streamline §6.20–6.21 without conflating the historical 14k run with fresh 400-update arms; retain unique NLL/exposure evidence and link Ch 7 rubric; development terminology; focused evidence-lab checkpoints |
| 7 | 578 | Outcomes/prerequisites; H3s; retain evaluation mechanisms and distinct real-model cases in 7.16/7.17; reader predictions; keep conceptual exercises and split operational extensions; route checkpoints and CLI footer |
| 8 | 279 | Outcomes; transition to end; H3s; §8.11–8.14 rewritten around exposure example; one figure per mechanism; solutions ordering (sol. l.76–84) |
| 9 | 377 | Extract operational §9.6 tail/§9.11; reuse loop + accumulation code; §9.9 reader prediction, controls, table and aligned raw outputs; contextualize l.322, retain tooling-versus-execution principle; distillation pointer with working exercise route; rebuild lab |
| 10 | 594 | Streamline §10.13–10.16 around the UNK-collision lesson without losing calibration/adversarial mechanisms; reuse reward-head snippet; $q=0.7$ formatting; lab checkpoints |
| 11 | 603 | Retain original-reference identity/matched exposure; extract operational §11.7 tail and §11.8.3; fix true pending claim; reader prediction before §11.8.4; concise guided lab |
| 12 | 454 | Existing three-seed table + figure + reader prediction; reuse PPO/RLOO code; preserve unequal gradient-work boundary; restate $\beta$ and align existing advantage/GAE notation; lab checkpoints |
| 13 | 811 | Shorten §13.8.3 operational chronology; preserve rollout/objective identity, coverage/failures and zero-signal/KL lesson; reader predictions and retained training-group example; fix l.501–502, retain l.401–405 principle; screen lab Q7–Q13 and mixed exercises |
| 14 | 910 | Shorten §14.8 while retaining incident/cause, pending-update replay, budget and supervision principles; use retained entropy/length outputs with reader predictions; route task IDs/commands to runbook; screen ex.18–35 individually |
| 15 | 725 | Accurate separate-root genealogy with failure annotations; worked defense; single home for distillation; distinguish temperature, sequence length and update duration |

## 4. Symbol table to adopt

These conventions bind shared quantities in reader-facing prose. Define local
indices/auxiliary symbols at first use; preserve API and archived-report notation
with a mapping. Check equations and storage layouts together, not by global
text substitution. A response-level advantage may be broadcast across tokens;
a token-level GAE advantage is not generally equal to it.

| Quantity | Symbol | Notes |
|---|---|---|
| Vocabulary size | $V$ | Tokenizer entries vs model rows stay $V_t$, $V_m$ only in Ch 2 where the distinction is the point |
| Model width | $D$ | Replace lowercase $d$ in Ch 2 |
| Head dimension | $d_h$ | Map Ch 4's $d_k$ and Ch 5's local $d$ explicitly; define differing query/key/value widths if needed |
| Value activations | $\mathbf V$ or $V_{\mathrm{val}}$ | Avoid collision with vocabulary $V$ (Ch 4 l.101, Ch 5 l.143) |
| Output head | $z_t = h_t W_{\mathrm{out}}^\top$, $W_{\mathrm{out}}\in\mathbb R^{V\times D}$ | Row-vector states and stored `nn.Linear` weight layout; an equivalent $[D,V]$ mathematical projection is its transpose; align Appendix B and tied embeddings |
| Context / response | $(x, y)$ | Ch 9 switches from $(c_j, a_j)$ |
| Sequence length | $n$ or $\lvert y\rvert$ | Distinguish total/context length from response length; map code/report $T$ where used |
| Temperature | $\tau$ | Calibration, sampling and distillation; replace temperature $T$ in Ch 15 |
| Update duration | $\Delta t_{\mathrm{update}}$ | Reserve $\tau$ for temperature; map existing timing labels without changing measurements |
| Learned reward | $r_\phi(x,y)$ | Reward model |
| Environment / verifier reward | $R(x,y)$ | Ch 12–14 |
| Reference policy | $\pi_{\mathrm{ref}}$ | `\mathrm` everywhere |
| Policy log-probability | $\log\pi_\theta(y\mid x)=\sum_t \log\pi_\theta(y_t\mid x,y_{<t})$ | Do not mix $\pi_\theta$ and $p_\theta$ in one identity (Ch 11 l.156–158) |
| KL coefficient | $\beta$ | Already defined in Ch 11; restate its objective/direction in Ch 12 and carry the convention into Ch 13–14 |
| Advantage | $\hat A_i$ response-level, $\hat A_{i,t}$ token-level | Ch 12 defines broadcasting versus token-level return/GAE; preserve detached estimator boundaries |
| LoRA scaling | $\alpha_{\mathrm{LoRA}}$ | Frees $\alpha$ |
| Chosen-NLL weight (Ch 11) | $\lambda_{\mathrm{SFT}}$ | |
| Entropy bonus (Ch 14) | $\lambda_H$ | |
| Distillation mix (Ch 15) | $\lambda_{\mathrm{KD}}$ | |
| TD residual | $\delta_t$ | Ch 12 |
| Advantage normalisation epsilon | $\epsilon_{\mathrm{adv}}$ | Ch 13 l.30 |
| Development split | "development set" | Apply Ch 7's definition throughout prose; preserve literal historical `validation` fields with a mapping |

## 5. Out of scope

- New numbered core chapters, days, notebook routes or substantive experiments.
  The appendix fallback above is a navigation change; bounded CPU reference
  replay and worked calculations are verification, not a new model campaign.
- GPU/model-scale jobs, model acquisition or GPU profiling; CPU checks may run
  on the current host without a machine switch or shared-environment changes.
- Rewriting measured reports, specifications, archives or model cards.
- Mac verification, hosted CI, animation rendering or publication.
- Learner position, mastery assessment or review-backlog decisions.

## 6. Tracking

Open the new editorial report in Phase 0 and update it at each phase milestone;
Phase 5 closes that same record. Record actual scope, command outcomes,
before/after counts and relocation/exercise mappings. Add progress pointers at
milestones without changing learner position. Use statuses `not started`,
`in progress`, `verified`, or `incomplete`, with evidence for every `verified`
phase. Existing course-readiness receipts do not verify this editorial revision.

| Phase | Current status | Required completion evidence |
|---|---|---|
| 0 | Not started | Reproducible baseline, mechanical repairs, checker tests and local source checks |
| 1 | Not started | Relocation/exercise map, preserved conceptual/evidence boundaries and resolved narrative findings |
| 2 | Not started | Source-linked cases, canonical code agreement and accurate genealogy |
| 3 | Not started | Complete notebook-route checkpoints/deliverables and working runbook links |
| 4 | Not started | Notation/shape review, all-notebook contract review and preservation hashes |
| 5 | Not started | Complete final local verification and manual reader-flow/visual review |

When mechanism exposition materially changes, follow the project's animation
opportunity check and reuse existing candidates in `visuals/animations/PROPOSALS.md`.
Update `LEARNING_MEMORY.md` only when its index/queue needs a new pointer.
Candidate capture is separate from rendering; this plan includes no production
approval. The plan revision itself completes none of the implementation phases.
