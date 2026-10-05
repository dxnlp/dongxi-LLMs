# Critique, revision and acceptance: bounded CPU control-and-replay evidence

The original mechanism package is measured and replayed: 486 fixed case/mode
rows, twelve deliberately authored adversarial paths, 864 unchanged historical
tiny-model candidates, 23 focused tests and a fresh eight-cell/six-figure
notebook. Its critiques and revisions are programmatic. No neural self-critique,
new model generation, API call, acquisition or GPU run is demonstrated.

The exact serialized-token control retains 54 helpful and 54 harmful flip
proposals and returns to its initial answer after two rounds. Historical-pool
format repair increases delivered valid/correct answers on this fixed panel,
but that is an explicit postprocessing rule, not evidence of improved reasoning.

## 1. What the loop actually does

A critique is advice, a revision is a proposal, and acceptance is a state change.
Treating all three as a single “the model thought again” operation hides which
part produced a benefit or a failure. With prompt $x$ and current draft $d_r$,
write the operations separately:

$$
c_r=C(x,d_r),\qquad u_r=R(x,d_r,c_r).
$$

If $a_r$ is the acceptance decision, the next delivered state is

$$
d_{r+1}=\begin{cases}
u_r,&a_r=1,\\
d_r,&a_r=0.
\end{cases}
$$

Our gate asks only whether the proposal is exactly one binary token followed by
natural EOS, with no error or truncation. It accepts a different valid binary
answer as a format-score tie. That cannot certify correctness. The evaluation
reference is absent from the whitelisted callback views, decisions and stopping;
the separate evaluator grades proposed and delivered paths afterward.

The loop stops on an identical accepted token path or after two rounds. A wrong
but valid unchanged answer stops; an invalid unchanged rejected path uses the
remaining round. Correctness never determines either choice. A usable critique
must itself contain a known action followed by natural EOS. A capped, malformed
or errored critique is retained but cannot invoke the reviser.

The motivating
[upstream self-refinement function](https://github.com/rasbt/reasoning-from-scratch/blob/a788466dc85cfe8617b6c0b809ed4ab9084485de/reasoning_from_scratch/ch05.py)
separates critique, revision and score-based acceptance, including score ties.
Our implementation, prose and authored data are independently written. It is
not a reproduction of that neural loop and imports no upstream code or assets.

## 2. Frozen inputs and three separate panels

The [original premeasurement spec](../specs/2026-10-04-critique-revision.md) and
[JSON protocol](../../fixtures/critique-revision/protocol.json) fix two rounds,
three modes, seeds 26061/26062/26063, eight independent candidates, all source
items, the acceptance gate and stable majority tie breaking. No rule, seed or
favorable subset is selected from the outcomes.

The authored microscope has twelve chosen fixtures. It includes accepted
wrong→right and right→wrong changes, a double-flip return to the original,
correct and incorrect format repairs, invalid/empty/capped proposals, a critique
exception, a revision exception and an unknown action. These are instrument
tests, not sampled model accuracy. Across their 24 rounds, six proposals are
rejected, two critique callbacks fail validation/raise, and one revision callback
raises. Every raw returned emission, error, skip decision and delivered state
is retained. The extra capped-critique and failing-deepcopy regression tests
exercise the hardened instrument outside the unchanged twelve-case campaign.

The exact programmatic panel uses all eighteen DXI05 items with three fixed
coordinate seeds: 54 draft/pool cases, each evaluated under three modes. Its
binary generator draws by seed/item/sample hash and never reads the reference
or arithmetic operands. Every independent candidate emits a binary symbol and
EOS. A critique emits an action symbol and EOS. This is serialized program output,
not an LLM token, judgment or likelihood.

The actual-candidate replay preserves the
[DXI05 source responses](2026-10-04-inference-selection/responses.jsonl),
SHA-256 `3db261750b6c98675b658968de3258fdd745d6be07a7c674bf21e25cfde36f04`.
All 864 records remain: six frozen initial/SFT-final tiny checkpoints, eighteen
items and eight actual autoregressive candidates per model/item. They include
empty EOS, repeated numerals and capped paths. The source item digest is
`ff5572a060d7a88f06f6ae7d48f4c0e4e5e5f3c9629776b7c717eae95beffb29`.
Original source/encoded-problem split validation runs before replay. References
are checked for authored input consistency there, then used only by evaluation;
they never enter a policy fit, critique, revision or selection callback.

Each replay pool starts at candidate zero. Identity requests KEEP. Repair keeps
an already valid path, otherwise takes the first observed binary token (zero if
none) and supplies EOS. Contrarian flips a valid answer, otherwise first repairs
it. None solves the arithmetic. The historical models never consume critiques,
and revisions have no neural logits, likelihood or inherited model identity.

## 3. Delivered answers versus the rounds that produced them

The two controls are no-critique delivery of candidate zero and independent
majority under the revision arm's serialized-output ceiling. Ties choose the
earliest eligible candidate. Gold is used only to evaluate the resulting choice.
Below are counts, not population estimates; every mode reuses the same drafts.

| Panel and mode | Cases | No critique succeeds | Revision succeeds | Independent majority succeeds |
| --- | ---: | ---: | ---: | ---: |
| Exact programmatic: identity | 54 | 24 | 24 | 26 |
| Exact programmatic: repair | 54 | 24 | 24 | 26 |
| Exact programmatic: contrarian | 54 | 24 | 24 | 27 |
| Actual replay: identity | 108 | 30 | 30 | 47 |
| Actual replay: repair | 108 | 30 | 65 | 47 |
| Actual replay: contrarian | 108 | 30 | 60 | 45 |

The exact contrarian panel accepts 108 proposals: 54 wrong→right and 54
right→wrong. The final answer equals the initial one in every case. Independent
sampling has negative held-out-source differences in the fixed programmatic
panel; those are retained rather than hidden behind its small pooled increase.

Actual contrarian replay accepts 73 wrong→right and 78 right→wrong proposals,
plus 65 repairs from invalid states. A final pooled count conceals those harmful
rounds. Repair's apparent benefit is largely the recovery of invalid outputs:
65 initial drafts are invalid, and the rule makes every delivered answer obey
the binary/EOS interface. Only 65/108 are correct. It cannot turn a format
criterion into a correctness guarantee.

These pooled 108-case rows cover six different checkpoints, not one model.
The per-checkpoint counts below preserve that distinction (eighteen items each).
Full state hashes and per-mode independent controls remain in the raw records.

| Source checkpoint | No critique | Repair | Contrarian | Independent under repair ceiling |
| --- | ---: | ---: | ---: | ---: |
| 10051 initial | 3 | 12 | 9 | 10 |
| 10051 SFT-final | 9 | 12 | 11 | 10 |
| 10052 initial | 2 | 8 | 9 | 6 |
| 10052 SFT-final | 6 | 11 | 11 | 7 |
| 10053 initial | 1 | 10 | 8 | 4 |
| 10053 SFT-final | 9 | 12 | 12 | 10 |

All training, held-out-source, held-out-template and held-out-family slices
remain visible in the report JSON and notebook. Eighteen items cover fourteen
source groups, including related source/template siblings. Modes share drafts
and seeds. No IID confidence interval, causal model-quality effect or transferable
reasoning capability is inferred from these descriptive counts.

## 4. The cost control is explicit about its unit

The serialized-output budget includes the draft, every critique and every
emitted revision, with EOS and invalid tokens counted:

$$
B_{\mathrm{revision}}=n_{\mathrm{draft}}+
\sum_{r=1}^{R}\left(n_{\mathrm{critique},r}+n_{\mathrm{revision},r}\right).
$$

In the exact programmatic panel a draft costs two symbols. One unchanged round
costs six total; two flip rounds cost ten. Independent attempts cost two symbols
each, so three or five attempts exactly consume those respective ceilings.
All 162 case/mode comparisons match this serialization unit. Identity/repair
each spend 324 symbols over their 54 cases; contrarian spends 540. No-critique
spends only 108 per mode, so it is not a spent-budget-matched baseline.

The real-pool comparator replays whole historical attempts. An attempt that
overruns the remaining ceiling is retained and charged but excluded from
selection. Across its 324 case/mode rows, 202 have exact serialized equality,
135 attempted-output tokens exceed the ceilings, and eligible prefixes underfill
by 165 tokens. The revision arms spend 2,831 symbols; attempted independent
outputs total 2,941 historical model tokens. These mixed-source units do not
equate neural computation. Do not call this a compute-matched experiment or
pretend the replay saved partial model forwards.

The unique DXI05 source panel records 1,736 generated tokens including EOS,
1,736 rescored actions, 1,736 generation forwards over 8,074 full-prefix positions
and 864 rescoring forwards over 4,328 positions. Those are historical source
measurements. Reusing them across mode rows does not mean those models ran
again. Programmatic critique/revision executes zero new model forwards and
must not be assigned model-token billing or fabricated likelihood scores.

The final durable campaign newly measures 9.324 seconds of CPU replay/collection
wall time, including fsynced journals. Non-adversarial callbacks themselves sum
to 0.023257 seconds over 1,556 attempted/completed calls. The campaign retains
802 round records: 778 declared-panel rounds and 24 authored rounds. These
timings are instrument measurements, not an optimized inference benchmark.

## 5. Independent review, hardening and final verification

The original campaign and
[first passed acceptance](2026-10-04-critique-revision-acceptance.json) are
preserved. Independent source review found a real stopping-contract defect:
a known-action/EOS-looking critique marked `max_tokens` could still trigger a
revision. Thus that original acceptance did not establish capped-critique
handling, even though its declared callbacks all stopped naturally.

The [hardening premeasurement follow-up](../specs/2026-10-04-critique-revision-hardening.md)
requires natural EOS and no truncation before invoking revision, with capped
output still charged and retained. It also makes failing-deepcopy/JSON error
retention robust and preserves known returned token IDs/counts. The intermediate
[hardening campaign](2026-10-04-critique-revision-hardening/results.json) is
retained with its historical source identity; it broadened one legacy
unknown-action diagnostic. The
[final compatibility spec](../specs/2026-10-04-critique-revision-hardening-final.md)
separates the old known-action check from the new natural-stop check, restoring
that original diagnostic without changing the recipe or any answer decision.

The current-source
[final campaign](2026-10-04-critique-revision-hardening-final/results.json),
SHA-256 `41bf5995ff114358f03baa7a84f17496086993bd7098d4e206a148e0790f6b4f`,
replays all original continuations, decisions, errors, grades and budget counts
exactly after excluding measured wall times and source/run identities. No
normalization conceals a changed diagnostic or decision. Earlier snapshots are
historical, not claims about current file hashes.

The [release acceptance](2026-10-04-critique-revision-acceptance-release.json)
records current source hashes, full historical/current numerical replay, the
unchanged 864 raw source responses and costs, exact 162/162 programmatic budget
matches, the retained actual-pool overrun/underfill, 23 passing focused tests and
fresh eight-code-cell/six-image execution. Its captured math checker output
records the precise contemporaneous global count and zero issues. All six
figures were visually inspected. Two schematic-only layout repairs separated
the feedback line from evaluation and wrapped the natural-stop label; no
numeric recipe/source changed in those notebook repairs. Earlier fresh manifests
remain in their temporary verification directories rather than being rewritten.

The [notebook](../../notebooks/day-26/04_critique_revision_and_acceptance.ipynb)
has adjacent worked explanations, live harmful/beneficial transitions, a
capped-critique assertion, source-bound plots and full bounded replay. Source
notebooks are not overwritten with executed learner state. The appendix-like
adversarial panel is never mixed into a model accuracy average.

## 6. Reproduce and interpret

Use the clean CPU course environment, no acquisition or notebook server:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /tmp/dongxi-course-reproduction.ZfVaEu/venv/bin/python -m unittest discover \
  -s tests -p test_critique_revision_lab.py -v
```

For a new campaign, invoke `python -m dongxi_llms.critique_revision_lab` with
`--protocol fixtures/critique-revision/protocol.json`, the final hardening spec
and an unused `--output` directory. Existing evidence cannot be overwritten.
The [collector](../../scripts/verify_critique_revision.py) requires a matching
fresh-kernel manifest, current `--reference`, original `--historical-reference`
and unused `--report` path. Ordinary errors and KeyboardInterrupt retain fsynced
partial rounds and failure events. Resume/exactly-once execution is not claimed.

Three questions close the mechanism lesson. Why can valid-score ties be harmful?
Because formatting says nothing about which binary answer is true. Why preserve
a rejected revision? Because it incurred output/work and identifies what the
gate prevented. Why isn't equal serialized output equal compute? Because a
programmatic symbol, a model-generated token, prefill, decode, scoring and
durable journaling are different operations with different costs.

Actual same-model neural critique/revision remains optional and unexecuted.
It needs a separate frozen model/tokenizer/chat interface, critique prompt,
generation budgets, acceptance scorer, actual raw continuations, source/task
splits, failure retention and independent quality evaluation. This package
establishes its bounded CPU state-machine/replay mechanism, not that broader
behavioral or model-scale claim.
