# Margin versus the generated answer

Book home: [Chapter11 §11.8.4](../../book/chapters/11-direct-preference-optimization.md#1184-the-preferred-answer-can-win-a-pair-without-winning-generation),
[worked answer17](../../book/solutions/11-direct-preference-optimization.md#17-a-pairwise-winner-is-not-necessarily-the-generated-answer)
and [the Chapter11 lab](../../book/labs/11-direct-preference-optimization.md).

Recorded on 2026-10-05 from completed experiment evidence. The learner has not
yet supplied a live prediction or explanation for this lesson. Active learning
position remains Day 9; preparing Day 18 material does not assess mastery or
move the learner there.

## The tempting mental model

“The preferred answer has a much stronger margin, so the model should now
generate that answer more reliably.”

The missing distinction is between **improving recorded chosen/rejected odds**
and **emitting a required string in an independent generation task**. One
diagnostic does not substitute for the other. Nor is correct stopping itself
correct answering.

## What was actually held fixed

Both native 100-update pilots begin afresh from the predeclared full400 SFT
export, not from the weights produced by their recovery rehearsals. Their
original parent, eight training pairs, actual replacement-draw order, encoded
IDs and masks, 100 updates and 2,047 chosen-target presentations agree. Equal
seed numbers alone would not establish that match; the actual 100 draw lists
and encoded-input bindings do.

The chosen branch retains the native template's real turn-ending token 151645
and following newline 198 in its supervised span. Prompt tokens are masked.
Decoding stops at the declared end token; it need not emit every trailing token
that teacher-forced scoring supervises.

| Training-only quantity | Chosen-only100 | DPO100 |
| --- | ---: | ---: |
| Optimizer updates / sampled pairs |100 /400|100 /400|
| Chosen targets |2047|2047|
| Rejected targets |0|1600|
| Policy forward calls / positions |400 /13343|800 /25439|
| Fixed-reference forward calls / positions |0 /0|800 /25439|

Chosen-only minimizes global chosen-token NLL within each accumulation window.
DPO minimizes mean pair loss using complete-response sequence log-probability
sums against the original frozen reference. Chosen-only forwards the full
chosen input; DPO uses single-shift chosen/rejected scoring. Equal chosen
exposure therefore does not mean equal information, gradients, geometry,
objective units or compute. The chosen run's separate eight preference-validation
reference calls are diagnostics, not hidden reference training.

The [DPO100](../../experiments/reports/2026-10-05-native-dpo100-independent-review.md)
and [chosen100](../../experiments/reports/2026-10-05-native-chosen100-independent-review.md)
independent reviews retain the technical gates, exact exposure join and
work/I/O boundaries. Failed DPO01/02 remain historical failures; the accepted
replay03 is not a retroactive pass for them.

## What a DPO margin measures

For prompt $x$, recorded chosen completion $y^+$ and rejected completion $y^-$,
write complete supervised-response sequence log-probabilities as
$\ell_\theta^+=\log\pi_\theta(y^+\mid x)$ and
$\ell_\theta^-=\log\pi_\theta(y^-\mid x)$. Then the unscaled margin is

$$
M_\theta=(\ell_\theta^+-\ell_\theta^-)-(\ell_{\mathrm{ref}}^+-\ell_{\mathrm{ref}}^-),
\qquad
L_{\mathrm{DPO}}=-\log\sigma(\beta M_\theta).
$$

$M_\theta$ describes a change in the pair's relative log odds against the
fixed reference. The scaled loss margin here is $0.1M_\theta$. It is not a
token probability, exact-answer rate or total mass assigned to all valid
answers. In particular, the objective does not compare the chosen string with
every other vocabulary continuation.

Greedy generation chooses one next-token winner at each prefix. A recorded
completion can become more likely without each of its tokens becoming that
winner. If generation takes another first token, later decisions condition on
a different prefix. This is a general mechanism; the aggregate experiment
below does not isolate the causal source of a particular omitted article or
noun. Its four likelihood-validation pairs and four independent location
prompts are **different populations**, not a per-prefix probability trace.

## Native likelihood: both chosen scores improve

The completed common likelihood panel measures four validation pairs with FP32
policy/reference weights and BF16 CUDA autocast, using the unchanged native
single-shift masks. These are mean **complete-answer sequence logp sums**, in
nats, not average token logp or mean probabilities.

| Policy | Mean chosen logp | Mean rejected logp | Mean unscaled $M_\theta$ |
| --- | ---: | ---: | ---: |
| Unchanged full400 |−11.486888|−25.312824|0|
| Chosen-only100 |−0.003524|−27.318045|13.488587|
| DPO100 |−6.642914|−91.838970|71.370116|

Less negative logp means higher likelihood. **Both interventions improve
absolute chosen likelihood in this native case**, including on each of the
four validation pairs. DPO's much larger margin also includes a large reduction
of rejected likelihood. It must not be mislabeled as the earlier CPU example
where chosen likelihood falls while the pair margin improves.

The [actual source-bound likelihood figure](../../experiments/reports/native-preference-figures-20261005-run-01/likelihood-and-margin.png)
puts these absolute and relative quantities next to one another. The separate
CPU notebook remains useful for the possible falling-likelihood failure, not
as a substitute for this native observation.

## Generated answers, stopping and retention are separate tests

The common generation consumer loads each exported policy in BF16 and uses
eager CUDA, the saved template and unforced greedy decoding with cap64. This
is separate from pair scoring and from each pilot's own FP32-loaded/BF16-autocast
four-answer diagnostics. Agreement of the own-run and common location counts
was observed, not assumed across loading paths. These backend labels describe
configured code paths, not instrumented kernel traces.

| Common generation population | Unchanged full400 | Chosen-only100 | DPO100 |
| --- | ---: | ---: | ---: |
| Four location prompts: strict whole answers |0/4|4/4|1/4|
|120 instruction items: strict whole answers |120/120|120/120|120/120|
|20 annotated reasoning items: bounded-parser correct |5/20|6/20|6/20|

Every response in these panels ends at a declared natural stop; none reaches
the cap. There are432 completed generated records across the three arms, with
no planned common record missing. Technical acceptance passes52/52 checks;
this is evidence integrity, not52 separate capability successes.

DPO emits `purple pouch` rather than `the purple pouch`, and `green` rather
than `the green folder`; its other outputs are `the tall vase` and `orange tin`.
The strict contract is not changed after seeing the outputs. An omitted
article and an omitted object name are different qualitative errors even
though both fail exact matching. A strict nonmatch alone does not establish
that every answer names the wrong semantic location. Inspect actual text.

All120 instruction answers are retained in40 lexical-value source groups,
but they use three task templates: this is a specific retention check, not
broad assistant competence. In reasoning, both descendants preserve the same
five correct IDs and add only `math-10`. The nine seen/development diagnostic
items stay5/9; eleven controlled held-out items change0/11→1/11. The whole20
must not be called unseen. A bounded whole-output parser neither checks every
mathematical step nor proves faithful reasoning. This is also not the separate
Instruct thinking-on/off32/128 experiment.

The [separate retention figure](../../experiments/reports/native-preference-figures-20261005-run-01/separate-retention-panels.png)
keeps instruction and annotated reasoning populations distinct. Existing
source-group resampling intervals describe the fixed evaluation population,
not training-seed uncertainty or contamination freedom. No new score,
resampling, pooled rank or quality-selected checkpoint was created here.

## Interpretation and open questions

Under this one fixed parent/data/recipe, chosen-only learns all four strict
location answers while DPO has the larger validation relative margin but only
one strict location answer. Both retain the specified instruction panel and
gain one bounded-parser reasoning item. This supports a measurement lesson,
not a universal ranking of DPO versus supervised fine-tuning or a broad
reasoning/generalization claim.

Before a new experiment, ask which explanation can actually be distinguished:
objective reduction, rejected supervision, response length/template tokens,
decoding path, or another feature of the fixed recipe. None was independently
isolated by this comparison. More data, alternative recipes and multi-seed
studies are possible future questions, not experiments executed by this note
or new mandatory acceptance gates.

Suggested live prompt: “How can both chosen likelihoods improve, yet the arm
with the larger DPO margin produce fewer exact answers—and what evidence
would locate the cause of one omission?” Record the learner's prediction
before revealing the table. No learner answer is recorded yet.

## Reuse candidates and canonical evidence

Reuse **CAND-ANIM-025**: equal chosen-exposure cards split into unequal
chosen-only/DPO work; separate absolute chosen/rejected bars feed an unscaled
relative-margin meter; then reveal independently generated answers, stopping
and retention panels. The native bars must show both chosen scores improving.
The different earlier CPU falling-mass trajectory needs its own explicit
label, not a blended animation. Source: automatic agent mathematics capture.

Possible X article: **“The preference margin improved. Why didn't the answer?”**
Build it around matched exposure, unequal compute, actual raw answer omissions
and the distinction between diagnostics and the task. Preserve validation
versus generation populations and precision paths, unchanged strict scoring,
known reasoning overlap and the one-recipe limit. This is a suggestion only,
not a drafted/commissioned article, publication or rendering approval.
All later media production remains on Mac Studio after explicit approval.

Canonical sources are the [native synthesis report](../../experiments/reports/2026-10-05-native-preference-comparison.md),
[common acceptance](../../experiments/reports/native-preference-evaluation-20261005-run-01/acceptance.json),
[common comparison](../../experiments/reports/native-preference-evaluation-20261005-run-01/comparison.json)
and [figure acceptance](../../experiments/reports/native-preference-figures-20261005-run-01/acceptance.json).
The actual figures consume retained metadata/raw records without another model
run. They are evidence-backed static companions, not produced animations.
