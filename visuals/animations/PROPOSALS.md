# Animation Proposal Inbox

This file is the shared inbox for animation ideas suggested during learning and
course development. Suggestions may come from Dongxi or from a contributing
agent. A suggestion is not approval to produce media.

Approved animation tasks move to `LEARNING_MEMORY.md`, where they receive a
stable `ANIM-*` task ID, a complete task packet, dependencies, acceptance checks,
and machine ownership. This inbox retains deferred and rejected ideas so the
same proposal is not repeatedly rediscovered.

## Candidate queue

### Foundations teaching extensions — existing candidates, 2026-10-10

- Source: agent, automatic mathematical opportunity check during
  `BOOK-FOUNDATIONS-2026-10`. Extend existing candidates 001–005 (prediction
  and learning), 008/013–016 (retrieval, residuals, norms and positions),
  018 (optimizer updates), 022/023 (evaluation and supervision), and
  025–029 (preferences, policy gradients, groups and teacher transfer).
  Reuse their packets; this is candidate capture rather than a new commission.
- Canonical mechanisms: an illustrative prefix follows categorical lookup,
  contextualization and $\nabla_zL=p-q$; the same response-level reward fixture
  follows $\nabla_zJ=p\odot(R-J)$, detached baselines, current/old likelihood
  ratios and sign-dependent clipping. Group centering and random scaling remain
  distinct from an action-independent baseline. Teacher targets retain their
  prefix distribution, temperature and gradient contract.
- Evidence state: deeper book explanations reuse existing executable sources,
  saved reference plots and retained empirical failures. New small numeric
  fixtures receive independent bounded calculations in the
  [foundations record](../../experiments/reports/2026-10-10-book-foundations-pass.md).
  No new model-scale outcomes or learner assessments are implied.
- Production dependency: stable chapter examples, the existing candidate
  packets and explicit media approval. Animation production remains on
  Mac Studio. These extensions are proposed and unrendered.

### Editorial mechanism extensions — existing candidates, 2026-10-10

- Source: agent, automatic opportunity check during the book editorial goal.
  Reuse CAND-ANIM-008 (causal retrieval), 019 (checkpoint state), 022 (paired
  evaluation), 023 (supervised masks), 026 (policy gradients), and 003/018
  (updates and learning-rate steps); no duplicate candidate or new production
  packet is created.
- Canonical mechanisms: $A=\mathrm{softmax}_{\mathrm{row}}(QK^\top/\sqrt{d_h}+M)$
  followed by $AV_{\mathrm{val}}$; valid-token summed NLL divided by total
  targets; detached response/token advantages with current/old/reference roles;
  pending observations replayed before new collection; matched exposure
  beside unequal gradient work and separate weight ancestry.
- Evidence state: Chapter 4 adds a calculated three-position fixture. Chapters
  6–15 reuse retained CPU curves, gradients, generated outputs and native
  comparisons, including caps, zero-reward groups and failed supervision.
  These are editorial explanations of existing evidence, not new training
  measurements, learner mastery or rendered animation.
- Production dependency: the existing candidate packets, stable chapter
  derivations and explicit production approval; animation ownership remains
  Mac Studio. This extension is proposed and unrendered. The
  [editorial record](../../experiments/reports/2026-10-10-book-editorial-pass.md)
  binds the source review separately from historical reports.

### README previews — existing candidates 001, 008 and 027, 2026-10-05

- Source: user requested beautiful, relevant README GIFs and explicitly allowed
  new animations. Promote bounded previews of CAND-ANIM-001 (next-token decoding),
  CAND-ANIM-008 (causal attention) and CAND-ANIM-027 (group-relative advantages)
  together as `ANIM-README-001`; these are extensions, not duplicate candidates
  or approval of their remaining full-length/evidence-backed storyboards.
- Canonical mechanisms: prefix → vocabulary softmax → greedy selection → append;
  causally masked scaled Q/K softmax → weighted V; population-normalized
  reward-minus-group-mean, including a constant-reward group with zero advantages.
- Evidence state: calculated illustrative fixtures only, with a schematic
  decoder. No trained-model activations, capability improvement or optimizer
  update is presented. The earlier measured native campaign stays separate.
- Production dependency: Mac Studio, the existing Pillow environment and
  FFmpeg; GIFs, PNG stills, editable source and hashes under `visuals/readme/`.
  No video is produced or added to Git. The packet and acceptance evidence live
  in `LEARNING_MEMORY.md` and `visuals/readme/manifest.json`.

### CAND-ANIM-001 extension — output rows versus decoder coverage,2026-10-05

- Source: automatic agent capture of a central softmax/interface mechanism,
  anchored in two actual failed Basechat cap profiles. Extend the existing
  logits→probabilities→chosen-ID→decode candidate, not a duplicate commission.
  State: evidence-backed suggestion only; no media or article production
  approved. All later animation production remains on Mac Studio.
- Keep input byte coverage and output mapping as separate lanes. Complete byte
  coverage allows an input encoder to emit known pieces for unfamiliar text;
  it does not ensure every model output row has a decoder entry. For this pinned
  cache, show151936 configured output rows,151643 BPE base entries and151669
  mapped IDs including the sidecar. Bare tokenizer JSON has22 added-token
  records, while the sidecar has26; do not present151669 as the bare-JSON count.
- Canonical mechanism: $p_i=e^{z_i}/\sum_{j=0}^{V-1}e^{z_j}$ over numeric rows.
  Full-support temperature1 sampling chooses ID151768 at the24th action;
  it is inside the configured range but lacks a tokenizer mapping, not an
  out-of-bounds embedding index or input unknown word. Reveal the actual
  selected logp−14.50478178687033 and conditional probability approximately
  $5.019417364343248\times10^{-7}$, not a total tail mass or failure frequency.
- In both cap32/cap128 profiles, retain all24 actions/likelihoods and828
  forwarded positions for the failing attempt. Keep the panel's40 observed
  rows, including its ERROR, separate from60 missing responses. Both actual
  native exits are1. The trajectories and derived attempt seed agree; they are
  not two independent rare-event samples and both fail before their cap.
  Do not animate filtering, resampling, retry or retroactive acceptance.
- A hypothetical valid-ID constraint belongs in a separate future-contract
  lane: restrict support to decoder set $D$ and renormalize over $D$. It changes
  the behavior distribution/likelihood contract; it is not this experiment's
  repair. No assertion about why the extra267 rows exist or their pretraining.
- Anchor: [focused Day2 evidence bridge](../../learning_artifacts/day-02-text-tokens-and-embeddings/output-vocabulary-and-decoder-coverage.md),
  [cap32 acceptance](../../experiments/reports/native-reasoning-base-chat-cap32-20261005-run-01/acceptance.json)
  and [cap128 acceptance](../../experiments/reports/native-reasoning-base-chat-cap128-20261005-run-01/acceptance.json).
  Possible `X-EVAL-001` extension, suggestion only: “How a valid model output
  index can still fail to become text.” Keep decode errors, missing responses,
  output caps and answer grading distinct. No learner advancement or public
  publication is authorized by capturing the idea.

### CAND-ANIM-025 extension — actual margin versus generated answer,2026-10-05

- Source: automatic agent capture of Chapter11 §11.8.4's explicit DPO
  mathematics and completed native comparison. Extend the existing candidate,
  not a duplicate commission. State: evidence-backed concept only; no film or
  article production is approved. Mac Studio owns any later production.
- Canonical equation:
  $M_\theta=(\ell_\theta^+-\ell_\theta^-)-(\ell_{\mathrm{ref}}^+-\ell_{\mathrm{ref}}^-)$,
  $L=-\log\sigma(0.1M_\theta)$. Show absolute sequence logp and unscaled
  reference-relative margin as separate meters, with clear nats/sequence-sum
  labels. Native chosen logp improves in **both** arms: unchanged−11.486888,
  chosen-only−0.003524, DPO−6.642914. Margin grows to13.488587/71.370116;
  rejected logp is−27.318045/−91.838970. Do not reuse the earlier CPU
  falling-chosen-likelihood trajectory as this native observation.
- Start from one byte-bound full400 parent. Identical400 replacement draws
  deliver2,047 chosen targets over100 updates to both trainable branches.
  DPO adds1,600 rejected targets,800 policy/800 reference forwards; chosen-only
  uses400 policy/zero training reference forwards. Keep unequal position
  geometry and separate diagnostic-reference work visible: matched exposure
  is not equal compute or loss units.
- Reveal the **different** four-location generation population only after
  the four validation-pair likelihood panel: unchanged0/4, chosen-only4/4,
  DPO1/4, all naturally stopped. Retain `purple pouch` versus `the purple pouch`
  and `green` versus `the green folder`; missing articles and missing nouns
  are different errors, without relaxing exact matching. This storyboard
  does not establish the causal reason for either omission.
- Then open separate retention lanes:120/120 instruction answers in all arms,
  annotated reasoning5/20→6/20 in both descendants, only `math-10` gained.
  Keep nine seen/development items5/9 and eleven controlled held-outs0/11→1/11
  distinct; natural stopping and bounded-parser correctness are not broad
  capability or reasoning-faithfulness certificates. Pair scoring is
  FP32/BF16-autocast; common generation is BF16-loaded/eager/greedy64. Own-run
  diagnostics and common evaluation remain separately labeled.
- Production dependency: freeze actual report/figure input bindings, retain
  strict contracts and raw answers, then obtain explicit Mac production
  approval. Anchors: [focused Day18 lesson](../../learning_artifacts/day-18-dpo-experiment/margin-versus-generated-answer.md),
  [native report](../../experiments/reports/2026-10-05-native-preference-comparison.md)
  and [static figure acceptance](../../experiments/reports/native-preference-figures-20261005-run-01/acceptance.json).
  Possible X article, suggestion only: “The preference margin improved. Why
  didn't the answer?” Keep one recipe, unmatched compute, populations and
  precision paths explicit; no universal DPO/SFT ranking or publication approval.

### Four evidence lanes from actual story and native recovery,2026-10-05

- Extend existing CAND-ANIM-003/004/019; do not create a duplicate commission.
  Keep process acceptance, numerical recovery, teacher-forced likelihood and
  free-generation quality in separate lanes. At the predeclared400 boundary,
  show falling fixed NLL beside real story continuations and the frozen rubric:
 45/48 natural control stops coexist with ending-quality mean0.020833/2.
  Later4000/8000/14000 cells remain visibly empty rather than interpolated.
- For the native chosen path, replay a20-label tail:40 numerical training
  labels become60 physical source/resume presentations, while the0/4 answering
  panel still stops naturally4/4 times. Separate validation/snapshot cost lanes.
  For failed DPO01/02, let durable update2/export tiles appear but keep final
  process result and known native exit absent. Parent exit1 is not the child's
  unknown exit. No “restored state means refunded work” transition is allowed.
- Source: agent, automatic mechanism/evidence capture. Anchors: Chapter6.21,
  Chapter11.8.3/solution16 and the
  [focused record](../../learning_artifacts/day-09-pretraining-run-and-diagnosis/technical-success-is-not-quality.md).
  Evidence is actual bounded Spark execution and separate AI ratings, not
  broad capability, human review or a completed14k recipe. Suggested only;
  storyboard/rendering still needs explicit approval on Mac Studio. No media
  or X article is produced by this entry.

### Native likelihood and completed replay,2026-10-05

- Extend CAND-ANIM-003/004 with the measured Chapter9 profile: teacher-forced
  target probability improves while the self-selected path remains incorrect;
  keep NLL4.061965→1.100189 beside0/8 exact answers and0/8 natural endings.
  Show the gold-prefix lane separately from the generated-prefix lane, including
  the supervised message-end target that still fails to win during decoding.
- Extend CAND-ANIM-019 with actual full-SFT checkpoint10→20 native replay:
  numerical-tail and same-producer serialized-tensor bytes align; keep the
  earlier conflict-stopped attempt on a separate cost/ownership timeline.
  Do not animate a numerical restart as a cost refund. The chosen-only CPU
  example retains seven reserved windows versus six completed updates.
- Source: agent, automatic mechanism/state capture. Evidence: actual pinned
  pretrained BF16 full replay and measured negative development generation;
  chosen-only exact recovery is tiny CPU. Arbitrary Python pickle metadata,
  cross-machine bitwise replay, merged LoRA and broad capability are not proved.
  Existing candidates only; no animation commissioned/rendered. Production
  remains separately approved on Mac Studio.

### A budget and a shutdown are contracts 2026 10 05

- Extend **CAND-ANIM-019**: cumulative valid targets $c$, complete next group $n$,
  cap $C$, and accept only $c+n\le C$. Animate9 then8 targets under cap16:
  retain7 unused, return the refused group and stream permutation/cursor/RNG,
  and keep gradients/LR/Adam/weights unchanged. Parallel tiles show16 physical
  positions per accepted group, not16 valid targets. Matching-cap resume restores
  the cumulative allowance rather than resetting it. Anchor: Chapter6.5, worked
  answer19, evidence-lab5 and the actual tiny CPU target-budget report.
- Extend the process-freeze storyboard with a leader exiting0 while its worker
  runs, then owned discovery/TERM/KILL/reap and separate safety status. Contrast
  a hard per-file limit with a sampled sum of growing files; display actual
  overshoot rather than drawing a false total quota. Anchor: Chapter14.8 and
  the fixed CPU worker/disk controls. Typed tensor bytes preserve shape/dtype,
  including BF16 bits, without pretending tiny recovery adds production resume.
- Source: agent, automatic mathematics/mechanism capture. Evidence: actual
  bounded CPU controls plus authored ownership faults, not a Spark pilot,
  cgroup or new model quality. Status suggested; production remains a separately
  approved Mac Studio task. No animation or article produced.

### Real cache state and realized teacher responses 2026 10 04

- Extend **CAND-ANIM-019** with ragged masks accumulating logical positions
  $p_{b,t}=\sum_{u\le t}m_{b,u}-1$ on valid tokens, RoPE on new Q/K and
  compact cached old keys. EOS and padding share an ID but not validity; finished
  rows leave the active batch while a cap does not fabricate EOS. Count keys
  and values into $M_{\mathrm{KV}}=2LBH_{\mathrm{kv}}Sdb$, explicitly tensor
  payload only, not total device memory. Keep actual slower tiny-CPU batching
  beside reduced forwarded positions rather than animating an assumed speedup.
- Follow an interruption after collection: preserve actions/old likelihoods,
  policy version, Adam, reference, data cursor and RNG; restore the pending
  update before collecting again. Show ambiguous dictionary flattening versus
  typed length-framed identities. Historical states remain visible and are
  compared by actual tensor bytes, not rewritten hashes. Anchor: Chapter14.7–8,
  Day25/02 and the independently verified v2 recovery/migration report.
- Extend **CAND-ANIM-029** from actual teacher sampling to first format-eligible
  parents and two student paths: complete `STEP total ANS binary EOS` versus
  derived `ANS binary EOS`. Prompt labels are masked; response/EOS targets enter
  $L_{\mathrm{response}}=-N_{\mathrm{targets}}^{-1}\sum_t\log p_\theta(y_t\mid x,y_{<t})$.
  Keep this one-hot sequence objective separate from a full teacher-vector KL.
  Split each evaluated response into final correctness and printed-step validity;
  preserve wrong-step/right-answer and valid-step/wrong-answer actual outputs.
  A checker does not reveal causal internal reasoning. Anchor: Chapter15.5,
  Chapter9, Day26/03 and the three-seed response-distillation report.
- Source: agent, automatic mathematics capture. State: suggested. Evidence:
  actual bounded CPU mechanisms and root independent replays. No pretrained,
  cross-host recovery, speedup or faithfulness claim. Production remains a
  separately approved Mac task; no media is created by this entry.

### An answer is not yet a demonstration 2026 10 04

- Extend **CAND-ANIM-029** with the verified teacher-data pipeline. One source
  prompt branches into fixed attempt slots; an error receives a distinct retry
  identity, and a repeated answer remains in the raw journal but is rejected
  from the unique candidate pool. Keep rejected output and incurred cost visible.
- A format/END gate preserves well-formed wrong responses. Branch the same
  frozen pool into top-per-prompt and seeded random selection, with gold outside
  the selection path. Global top6 visibly loses all three-word sources; one-
  per-prompt keeps coverage. This is an executed programmatic teacher, not
  neural self-critique, human labels or faithful reasoning.
- Animate target-count tiles $n_{a,j}$, including real END, into
  $E_a=U\sum_j n_{a,j}$. Show42/46/42 targets times80 updates, separating
  examples, supervised exposure, padded computation and upstream teacher cost.
  Finish with held-out correctness and stopping panels side by side: all
  greedy polite-prefix controls remain zero although END is learned.
- Canonical anchors: Chapter8.11–8.14, Chapter15.4, the Day11/04 notebook and
  preserved teacher-data journal/report. Source: agent, automatic mathematics
  capture. Evidence: bounded CPU measurement and root exact numerical replay.
  State: suggested; production requires a separately approved Mac task.

### Actual candidate availability and a wrong majority 2026 10 04

- Extend **CAND-ANIM-029** using the verified DXI-05 pool: eight actual parity
  answer cards arrive, three0 and five1; a separately labeled evaluator shows
  correct0 is available while the gold-blind vote chooses wrong1. Keep gold
  labels out of the selector path. Increasing a nested prefix can increase
  oracle availability while a real vote or likelihood decision worsens.
- Count every EOS/cap/invalid attempt. Move the whole attempt crossing a token
  cap into a charged-but-unselectable lane; distinguish actual consumed tokens,
  nominal cap, mandatory rescoring and selector overhead. There is no free
  rejected answer, exact token-level interrupt or optimized-serving claim.
- Anchor: saved864-candidate ledger, Day10 notebook05, three declared seeds and
  actual selection decisions. Reproduce source hashes and preview arrays before
  a separately approved Mac production task. No animation was produced here.

### Reward identity and objective weighting controls 2026 10 04

- Source: automatic mathematics capture while implementing the reference-audit
  improvements. Proposed only; no animation or article commissioned. Production
  remains on approved Mac Studio task packets.
- Extend **CAND-ANIM-024** with word-token aliases splitting into distinct fixed
  character sequences. Keep the unchanged historical word-model result beside
  the new character experiment: distinct encodings remove an impossibility, not
  automatically poor held-out predictions. Anchor: Day 16 text/process notebook,
  both saved reward exports and independent encoding-gate evidence.
- Extend **CAND-ANIM-026** with a continuous short trajectory: reward only at
  emitted EOS → TD residual
  $\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t)$ → backward GAE accumulation
  $A_t=\delta_t+\gamma\lambda(1-d_t)A_{t+1}$ → detached policy advantage and
  separate critic MSE. Show a collector cap retaining bootstrap, then the broken
  cap-as-EOS switch. Distinguish terminal proxy reward from scoring an unfinished
  prefix; retain learned-value failure, not just an ideal oracle. Anchor: new
  verified Day20 critic notebook and `critic_policy_lab.py`. The hand-checked
  gamma0.9/lambda sweep supplies static arrays; actual negative capped paths
  remain separately labeled.
- Extend **CAND-ANIM-025** with a growing DPO margin while chosen and rejected
  absolute likelihoods move separately. Add chosen-NLL/rehearsal branches and
  count their supervision; cross to fixed generation/retention evaluation rather
  than implying a universal repair. Anchor: Day 18 retention notebook and all
  verified predeclared noisy/length arms. Show imperfect warm parity before
  labeling rehearsal as continued learning versus retention. A wrong chosen
  label reinforces the wrong answer; answer→STYLE→STYLE→EOS can be correct at
  its first atom but fail the fixed canonical-output contract.
- Extend the same **CAND-ANIM-025** for the original-fixture matched chosen-SFT
  path: one byte-identified parent splits into untouched, chosen-only CE and
  DPO; identical replacement-draw cards enter both trainable lanes. Keep prompt
  masks and terminal targets visible, then show global chosen-target averaging
  versus mean-pair response-sum DPO. The chosen lane has no rejected or frozen
  reference arrows. Count those unmatched forwards beside equal chosen exposure,
  then reveal every fixed held-out generation, including failures. Source:
  automatic agent capture during Chapter11 section11.8.2 integration and
  `chosen_sft_control.py`. This is a proposed mechanism storyboard, not evidence
  of pretrained quality, accounted recovery, a commissioned film or a rendered
  animation; production stays on Mac after approval.
- Extend **CAND-ANIM-027** with identical sampled action cards assigned
  $m_{it}/(NT_i)$, $m_{it}/\sum_jT_j$ or $m_{it}/(NC)$. Animate coefficients
  flowing into $\partial L/\partial\ell_{it}=-w_{it}A_i$ at ratio one; show
  parameter-gradient rotation versus scalar rescaling, positive/negative clip
  branches, and component versus total reward normalization. Then move rejected
  all-wrong groups into a visibly counted cost ledger before selected groups.
  Anchor: verified Day 22 objective notebook, analytical tests and three-seed
  raw ledger. No training gain or complete named-algorithm reproduction.
- Required production evidence: exact source revisions and static preview
  arrays, inclusive EOS/masks, detached old/reward paths, enumerated GAE endpoints,
  all failed outcomes and attempted-token counts. Film selection/rendering remains
  separate from tested scientific notebook plots.

### Refinement and conditional-distillation extensions — 2026-10-04

- Source: agent automatic mathematics capture; state suggested, not approved.
  Extend **CAND-ANIM-029** rather than commission duplicate media. Mac Studio
  owns any later production; no animation or article is created by this entry.
- Refinement: hold the delivered draft identity fixed while critique $C$ emits
  advice, reviser $R$ proposes $u_r$, and acceptance chooses
  $d_{r+1}=u_r$ or $d_r$. Animate an accepted right→wrong→right path whose final
  score is unchanged. A format-score tie is not a correctness check. Show capped
  critique tokens being charged and retained while the revision arrow is blocked.
- Cost: accumulate $B=n_{\mathrm{draft}}+\sum_r(n_{\mathrm{critique},r}+
  n_{\mathrm{revision},r})$ alongside independent attempts. Keep programmatic
  serialized symbols and historical neural tokens/forwards on distinct ledgers;
  overshooting whole attempts are charged but cannot vote. No compute equality.
- Conditional distribution: hold one visited state fixed, detach teacher $q$,
  and contrast forward $p_i-q_i$ with reverse
  $p_i[\log(p_i/q_i)-\sum_jp_j\log(p_j/q_j)]$ logit gradients. Then move the
  state source from offline teacher histories to actual student histories,
  without silently reversing KL. Training-only teacher hint cards never enter
  the student's input or the evaluation branch.
- Occupancy: draw separate conditional-loss and state-distribution derivative
  arrows for $J=\sum_s d_\theta(s)L_\theta(s)$; explicitly cross out the latter
  in this implemented fixed-history backward. Current-policy collection is not
  a complete occupancy-gradient estimate or a reproduction of SDPO.
- Tail bucket: preserve selected coordinates and sum the others into one tile.
  Two different internal tail allocations can yield zero bucket KL and positive
  full KL. Reveal the exact teacher/student-tail-weighted conditional divergence
  term; distinguish full-support main data from undefined zero-support controls.
  No invented probability floor, identical-gradient promise or measured savings.
- Anchors: Chapter15 sections15.8–15.9, Day26 notebooks04–05, their original
  specifications, hardened tests and preserved measured campaign reports.
  Dependencies: accepted final identities and mathematical replay, learner
  selection of a complete task packet, then explicit production approval.

### Text reward and sampled reasoning controls — 2026-10-04

- Automatic capture for DXI-04,07 and the real-response adapter in DXI-02;
  suggested only, Mac Studio production awaits approval.
- Extend **CAND-ANIM-024**: prompt/completion token states → last actual
  nonpadding endpoint (explicit inclusive/excluded EOS) → scalar reward →
  pair margin. Slide left/right padding while the endpoint score stays fixed;
  then supervise terminal outcome versus individual step boundaries. A process
  label is not a critic's expected future reward. Anchor: Day16 notebook03,
  `text_reward_lab.py`, and its separately labeled original/negative evidence.
- Use the unknown-token failure as a continuous fork: two different texts
  become the same ID sequence, so a deterministic scorer must agree:
  $\mathrm{encode}(x)=\mathrm{encode}(x')\Rightarrow r_\theta(x)=r_\theta(x')$.
  Opposite labels cannot both be fitted for those identical inputs. Keep raw
  source separation distinct from encoded calibration/test independence;
  a later tokenizer intervention is a new experiment, not a repaired old score.
- Extend **CAND-ANIM-027**: sampled two-action paths → binary exact reward →
  within-prompt detached advantage → old/current conditional-grammar likelihoods
  → shared decoder update. Animate the complete-path factorization
  $p_\theta(a,\mathrm{EOS}\mid x)=p_\theta(a\mid x)
  p_\theta(\mathrm{EOS}\mid x,a)$; both answer and EOS are learned here.
  Show all three predeclared seeds, the minority-answer failures and a
  constant-answer comparator. Longer computation or a perfect imbalanced slice
  must not become a reasoning-transfer claim. Anchor: Day23 notebook03 and
  `reasoning_controls.py`; preserve the earlier negative GRPO experiment.
- Extend **CAND-ANIM-022**: actual local HF forward → sampled ID with recorded
  raw/behavior log probabilities → unmodified raw decoding → separately declared
  terminal-stop removal for grading. Count attempted forward positions,
  generated actions, EOSturndone, caps and context-limit/error branches. These
  are real tiny-random CPU adapter events, not pretrained performance or an
  optimized latency benchmark. Reuse the inference lifecycle packet for timing.
- October5 continuation of existing **CAND-ANIM-022**: keep the actual native
  Instruct/thinking-off/cap32 `5 + 6 = 11` response visible in three separate
  lanes: preserved raw/scoring text, natural turn-stop versus cap, and frozen
  extraction/canonicalization. The whole-equation fallback reaches `UNSUPPORTED`
  while `format_policy="any"` remains valid; do not animate a last-numeral rescue
  or replace the accepted negative grade. Show sampled repetitions versus the
  separate greedy diagnostic and eleven overlap-excluded held-out problems,
  not100 independent questions. Anchor: the [Day23 focused artifact](../../learning_artifacts/day-23-grpo-rlvr/generated-answers-and-grader-interfaces.md)
  and [partial actual report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md).
  State: suggested only, assigned to later Mac Studio production; no rendering
  or publication. Later baseline/RLVR outcomes stay pending, and any improved
  instruction or extraction is a new declared intervention, not current-result
  repair. BF16/eager generation is not a MATH control or optimized latency claim.
- Required checks: exact padding/endpoint invariance, recorded encoding aliases,
  EOS-inclusive gradient and likelihood masks, sampler/recomputation agreement,
  independently replayed grades and all failure/cost rows. No film/article
  commissioned by this capture; production remains a separate Mac task.

### Verified foundation extensions — 2026-10-04

- Source: agent automatic mathematics/opportunity capture for DXI-01,02,08,13.
  State: suggested; no rendering approval. Production stays on Mac Studio.
- Extend **CAND-ANIM-019**: hold embedding/output rows fixed while token-ID
  labels2/3 swap. Show shape checks still passing, then semantic checks rejecting;
  keep an unchanged-map normalization intervention beside legitimate save/reload.
  Anchor: Day1 notebook04 and the identity report. Tiny CPU serialization is
  executable evidence, not pretrained ancestry or Spark compatibility.
- Extend **CAND-ANIM-022**: raw answer → bounded extraction → supported equivalence
  → separate format/task/stop decisions → paired source-group resampling.
  Anchor: Day10 notebook04 and mathematical-grading report. Preserve unsupported
  versus wrong, ambiguous versus malformed, and fixture intervals versus model scores.
- Extend **CAND-ANIM-024**: blind display slots A/B move while canonical candidate
  IDs stay fixed; ties, abstentions and invalid outputs follow separate paths.
  Then distribute one source's weight over its sibling comparisons:
  $L=\frac{1}{S}\sum_s\frac{1}{n_s}\sum_j\ell_{sj}$.
  Anchor: Chapter10/Day15 notebook03 and preference-audit report. The averaging
  convention is a declared population objective, not human consensus. Authored
  references and simulated judges must stay visibly labeled; added injection
  text also changes length, so require a matched-length benign control for causal claims.
- October5 continuation extends **CAND-ANIM-022/024** with the actual story-rating
  instrument: opening/complete response → shuffled anonymous display → two
  explicitly declared reviews → retained disagreement/declared adjudication →
  private-codebook join → paired source-opening resampling. Keep empty review
  cells visibly empty and keep all four decoding recipes attached to one
  opening when it moves into a bootstrap draw. Authored fixture scores stay
  labeled authored; current corpus audit is separate from unrun model
  continuations and missing human ratings. Anchors: Chapter7section7.17,
  solution23, Lab7 and `src/dongxi_llms/story_rubric.py`. State: suggested,
  agent opportunity capture; not approved or rendered. Production stays on Mac.
- Extend **CAND-ANIM-027** with raw $p$ → transformed collector $b$ → declared
  target $t$ → ratio $t(a)/b(a)$. Animate the exact support condition in
  $\mathbb E_b[(t(a)/b(a))R(a)]=\mathbb E_t[R(a)]$; an excluded action cannot
  return through a zeroed ratio. Fixed-support conditioning changes the objective,
  and temperature remains part of its probability calculation.
- Extend **CAND-ANIM-026/027** with equal forward KL values but different frozen
  sample gradients. Display detached collection/reference arrows versus live target
  weights before checking the exact categorical gradient. Separate EOS terminal,
  continuing-task cap, prompt and padding masks. Anchor: Chapter14/Day20 notebook03
  and sampling-support report. Fixed-state finite identities do not establish
  general trajectory off-policy correctness or a universally valid PPO regularizer.
- Required before production: reproduce numerical traces at the exact recorded
  source revision, verify the support/normalization/detach masks, select and approve
  a complete packet. Static reference plots do not count as produced animation.

### Reference audit mechanism extensions 2026 10 04

- Source: agent opportunity check during the learner-requested two-repository
  audit and [improvement plan](../../docs/COURSE_IMPROVEMENT_PLAN.md).
- Reuse CAND-ANIM-024 for text hidden-state pooling→reward margin→outcome or
  step-boundary labels; preserve the difference between process correctness
  and a critic's expected future return. Linked packages: DXI-07,08.
- Extend CAND-ANIM-026 with reward→return/TD residual→GAE→detached advantage→
  actor/critic updates, including EOS versus truncation and learned-value error;
  extend CAND-ANIM-025 with DPO margin versus absolute likelihood/retention.
  Linked packages: DXI-10,11. Canonical equations await those tested modules.
- Extend CAND-ANIM-027 with identical-rollout denominator/scaling comparisons
  and raw versus transformed behavior probabilities/support boundaries.
  Linked packages: DXI-12,13; no named-algorithm or off-policy guarantee.
- Extend CAND-ANIM-029 with candidate pool→vote/rank→selected/random corpus→
  sequence student, draft→critique→revision decisions, and offline versus
  student-prefix distillation. Count tokens/cost and right→wrong transitions.
  Linked packages: DXI-05,06,09,15,16. Keep oracle selection visibly diagnostic.
- Evidence state: planned mechanisms, no new numerical results or produced
  media. Reuse existing packets rather than commission duplicate films.
  Production dependency: implement/verify the underlying lesson, learner
  selects a complete task packet, then render on Mac Studio with approval.

### Full-course mathematics capture — 2026-10-04

The learner requested the complete28-day/15-chapter teaching build. Automatic
math-opportunity capture now adds CAND-ANIM-022 through029, with source modules,
continuous storyboards, mathematical acceptance checks, evidence limitations
and Mac ownership in [COURSE_STORYBOARDS.md](COURSE_STORYBOARDS.md).

The new topics are paired evaluation, SFT masks/gradient paths, Bradley–Terry,
DPO ratios, policy gradients/baselines/PPO, GRPO, reward hacking/rollout versions,
and distillation/selection. Earlier BPE, embedding, attention/KV, modern-decoder,
looped-transformer and pretraining packets remain authoritative and are reused.
Status: proposed, not approved or rendered. No media/publication task is inferred.


### CAND-ANIM-009 extension — document example, two waits, 2026-09-15

- Source: user approved a standalone adaptation of the lifecycle animation,
  replacing symbolic tokens with readable document facts and output phrases.
- Task: `ANIM-PD-002`, produced on Mac; revised MP4 and 960px GIF
  in `projects/two-kinds-of-waiting/`. No modification to the article or original
  loops. All source/return evidence is in the task packet and project metadata.
- Mechanism: request send → first arrival (TTFT), then each adjacent pair of
  output arrivals (ITL). Initial document-processing version rejected; timing
  revision explicitly approved. One continuous timeline; no benchmark claim.
- Precision: each word card stands in for one token arrival; no literal segmentation.
  State: review. No new proposal or follow-on production inferred.

### CAND-ANIM-009 extension — first-token boundary and KV handoff, 2026-09-14

- Current state: review. Learner approved all seven article loops as
  `ANIM-PD-001`, superseding the earlier deferrals below. Seven GIFs and 1080p
  MP4s rendered locally and integrated in `X-PD-001`; original stills retained.
  Numerical, event, media and source-hash checks pass. Previews/final frames
  inspected; browser playback is not claimed. Await learner visual feedback.
- Editorial refinement: use report → three conclusions and short prompt → long
  speech to motivate phase workloads; A's active answer and B's incoming document
  motivate chunk scheduling. Static images updated. Any future clip should carry
  these examples without converting schematic sizes into timing/throughput claims.
- Add the learner's parallel-position question: animate h1/h2 → Q/K/V, share
  K1/V1 into both causal reads, then produce O1/O2 independently before the next
  layer. Static dependency figure completed; reverse-row and batched attention
  checked numerically. Motion remains deferred with all other article clips.
- Earlier learner direction: add the Decode feedback-loop still now, defer all
  article animations for later production together. Still is complete; no new
  ANIM task or render was started. Await renewed production approval.
- Source: user's standalone `X-PD-001` article approval and automatic mechanism
  check. Static figures approved for production; motion is still a proposal.
- Mechanism: prompt positions execute together within a causal layer; prefill
  predicts y1 while cache length stays N; feeding y1 appends per-layer KV to N+1
  and predicts y2. Then contrast chunk scheduling and cross-worker KV handoff,
  including layerwise compute/transfer overlap with a visible communication tail.
- Evidence: Chapter 4 invariance and Chapter 5 resource accounting; deterministic
  two-layer fixture checks cached/full and chunked/full equality, with a broken
  noncausal control. Handoff is a primary-source schematic, not a local benchmark.
- Production record: `publications/x-articles/x-prefill-decode-001/visual-plan.md`
  and `ANIM-PD-001`. Preserve object identity; no new candidate duplicate.

### CAND-ANIM-009 extension — inference clocks and phase scheduling, 2026-09-13

- Source: user-approved `X-INFER-001` overview and agent motion-opportunity check.
- State: discuss motion; seven static article figures exist, no new video render.
- Mechanism: distinguish TTFT/ITL; preserve per-request KV as batch members
  complete/join; retain prefix context through chunks; move KV across separate
  prefill/decode resource pools. Extend existing cache-flow grammar rather than
  duplicate the article's earlier prefill/decode loop.
- Evidence: Chapter 4 §4.11 and current primary documentation in the article
  `source-map.md`; timelines are schematic, not measured speed comparisons.
- Production dependency: learner selects batching and/or phase-flow storyboard;
  define admission/preemption and KV ownership invariants in a Mac task packet.
  Detailed plan: `publications/x-articles/x-inference-001/visual-plan.md`.

### CAND-ANIM-020 — speculative acceptance and the new branch

- Source: user requests speculative decoding in the inference article; agent
  identifies draft/verify/reject as a mechanism that motion can clarify.
- State: discuss motion; static rejection-branch figure prepared in `X-INFER-001`.
- Mechanism: propose candidate sequence, target-model verification, accept a
  prefix, reject at the first failed position, discard dependent draft suffix,
  produce a corrected token and continue. Verification concerns probabilities.
- Evidence: Leviathan et al. ICML 2023 and vLLM documentation in source map;
  illustrative acceptance counts, no model output or throughput observation.
- Production dependency: choose exact sampling or greedy scope explicitly;
  implement acceptance/correction fixture before render, including all-accepted
  and early-rejection boundaries. No unconditional speed or text-identity promise.

### CAND-ANIM-021 — representation size versus total memory

- Source: agent automatic arithmetic capture from user-approved quantization topic.
- State: discuss motion; static calculated weight-payload comparison exists.
- Canonical equation: weight payload = parameter count × bits per weight / 8.
  Exactly 8 billion weights at 16 bits and 4 bits give 16 GB and 4 GB before
  metadata; decimal GB, not measured GPU allocation.
- Evidence: Chapter 5 §5.18 plus source-mapped quantization documentation.
- Production dependency: fixed parameter identity, separate weight/KV/buffer
  categories, no implication that payload shrinkage guarantees speed or quality.
  Extend existing cache-memory visual language; static article chart may suffice.

### Day9 completed-run extensions — 2026-09-14

- Source: agent, automatic mechanism/budget capture during Chapter6 synthesis.
- State: discuss; extends existing budget, next-token and temperature candidates,
  not approval for additional films. Production remains on Mac Studio.
- Budget scene: separate processed positions (229,376,000), valid targets
  (48,839,975), and prepared targets (390,708,926). Animate masked positions
  remaining in computation; valid/processed =21.29%. Rate denominators switch
  between summed update timers and full trainer duration. No linear packing
  speedup or unique-information claim is justified by those counters.
- CAND-ANIM-001/003 extension: contrast fixed recorded prefixes during evaluation
  with model-generated prefixes feeding subsequent predictions. This mechanism
  explains why the two evaluations differ, not the uniquely proven cause of
  our story failures. Keep causal masking correct in both lanes.
- CAND-ANIM-007 extension: hold weights fixed while decoding settings change;
  seed has no effect in greedy mode, and higher temperature does not teach new
  knowledge. Backend probes are limited evidence, not browser-flow verification.
- Sources: Chapter6 sections6.15–6.19 and the portable completed-run report.
  Animation should retain poor samples and literal checkpoint/update labels.
  Next decision: choose whether these strengthen an existing approved narrative;
  no render or new production packet authorized by chapter writing.

### Day9 tokenizer/model accounting —2026-09-13

Measured SFT extension,2026-10-05: agent automatic budget/mask capture for the
existing training-update/budget scene, using Chapter9's
[pinned tokenizer result](../../experiments/reports/2026-10-05-base-tokenizer-sizing.md).
Render the selected80 transcripts as3,154 input-position tiles; keep455 shifted
assistant targets lit, including each answer's end marker and newline. Add two
separate development lanes of360 targets each: the work counter becomes1,175,
but training exposure stays455. Keep16 proposed generation attempts/1,024
new-token slots in a third lane, not mixed with teacher-forced labels. All these
encodings are measured; updates/generations are unexecuted. This is a source-bound
candidate extension, not a new film, throughput claim or render approval. Mac
production/publication remains separately authorized.

Budget extension: connect measured valid-target throughput to training exposure
(rate × training time), separating padding, evaluation/checkpoint overhead and
unique data from repeated presentations. Capture under the existing training
update/budget candidate, not a new film. The10k targets/s illustration in the
Day9 design note is hypothetical; no measured Spark training rate yet.

Source: agent automatic math capture while settling the story-model baseline.
Extend existing ANIM-EMB-001 and head-shape candidate CAND-ANIM-012: IDs[B,T]
become states[B,T,512], split into8 heads of64, and project through the shared
50,257×512 embedding/output matrix to logits[B,T,50257]. Keep the table's
25,731,584 parameters counted once despite two computation roles; contrast this
with51,463,168 per-sequence logit elements atT=1024. Parameter storage and
activation storage are different. Source: Day9 tokenizer/model design note;
meta-device count verified, no GPU peak measurement or trained result.
This extends existing candidates only; production remains unapproved on Mac.

### CAND-ANIM-009 extension — A/B/C prefix reuse, 2026-09-13

- Source: user; approved for social-post animation as `ANIM-KV-007`.
- State: review; rendered MP4/GIF and checked stills in
  `projects/kv-prefix-abc/` on Mac. Approximately 19 seconds; no article edits.
- Mechanism: unchanged causal token prefix retains its per-layer K/V; an edit
  instruction is a new suffix, whereas edited input changes the cache identity.
- Evidence: canonical causal dependency plus executable illustrative-prefix
  fixture; no measured tokenizer/model output or speed claim.
- Motion: stable input/cache columns; B changes instruction only; C changes
  budget and recomputes dependent suffix while keeping earlier state fixed.
- Acceptance/return tracked in the `ANIM-KV-007` packet in LEARNING_MEMORY.md.

### CAND-ANIM-009 extension — replace article code with a decode close-up

- User approved `ANIM-KV-006` on 2026-09-12. One head/new position: Q/K/V
  projections → append → scaled Q/K matching → softmax → weighted V sum → retain.
- Source: existing executable article example and Chapter 4 §4.10. Toy fixture
  arithmetic is explicit; no model observation is implied. Mac production only.
- New article slot 4 replaces code; prior loops remain unchanged. Packet in
  `LEARNING_MEMORY.md`; source in `projects/kv-decode-step/`.

### CAND-ANIM-009 extension — cache memory growth, 2026-09-12

- User approved the remaining memory figure as `ANIM-KV-005`; packet in
  `LEARNING_MEMORY.md`, Mac production in `projects/kv-memory-growth/`.
- Canonical equation: separate K/V bytes = 2 × B × L × T × H_KV × d × b.
  Calculated payload only. Grow retained positions, then explicitly compare
  KV-head configurations; no inference-time head deletion or allocation benchmark.
- Replace article slot 5, preserve the other loops and static equation cards.

### CAND-ANIM-009 extension — remaining article GIFs, 2026-09-12

- User approved append/edit and global-sharing loops after viewing `ANIM-KV-001`.
  Promoted as `ANIM-KV-003` and `ANIM-KV-004` with packets in `LEARNING_MEMORY.md`.
- Reuse the existing causal-invariance and CED evidence; no competing candidate.
- Mac production only. Preserve the existing prefill GIF and longer CED films.
  Equations and the memory bar chart remain static.

### CAND-ANIM-009 extension — short prefill/decode article loop, 2026-09-12

- Source: user approval following agent suggestion; `ANIM-KV-001` promoted in
  `LEARNING_MEMORY.md`. Only the first article loop is approved for production.
- Mechanism: prompt prefill → retained K/V → first prediction → feed selected
  token back → append new K/V → fresh Q reads past/self → next prediction.
- Evidence: canonical causal-cache derivation; token labels/tensor glyphs are
  illustrative. No new numerical result or timing benchmark is implied.
- Scope: Mac Studio Manim production, then GIF/MP4/PNG export; article slot 1.
  Append/edit and global-sharing GIF proposals remain unapproved for production.

### CAND-ANIM-009 extension — CED global/local KV flow, 2026-09-10

Design requested later on 2026-09-10 for two specific points. Storyboards:
[`projects/deepseek-ced/DESIGN.md`](projects/deepseek-ced/DESIGN.md).
Film 1 separates moving the global-KV source (CED) from sharing storage (CSA2).
Film 2 extends CAND-ANIM-017 with the same selected KV buffer's score and payload
roles, confirmed in the released sparse-attention kernel. English, motion-first;
initial design was followed by explicit implementation/render approval.
Return: ANIM-CED-001 and ANIM-KV-002 are locally rendered (50.3s/56.3s) with
independent NumPy traces and nine passing checks. Review:
[`projects/deepseek-ced/review.html`](projects/deepseek-ced/review.html).
Source, media, geometry and browser evidence is in that project's manifests;
no model-scale quality/speed result, publication, or Git integration claimed.

- Source: user DeepSeek V4.1 deep-dive request; agent automatic equation and
  dependency-flow check. Reuses the existing KV-cache candidate, not a new film.
- Mechanism: move decoder global-KV creation from layer-local inputs to final
  encoder states; keep current-layer main Q and SWA KV distinct. Overlay Full,
  Reindex, and Reuse; shared selections must not imply equal attention weights.
  Finally contrast full local dependency reconstruction with bounded tail replay.
- Canonical source: `learning_artifacts/day-06-modern-architecture/deepseek-v41-causal-encoder-decoder.md`,
  report sections 2.2-2.3 and 3.2.2, pinned reference model/config.
- Evidence: report and code inspection; inline dependency schematic only.
  No model-quality reproduction or measured serving speedup. Bounded replay is
  approximate, and the public reference prefill traverses all backbone layers.
- The global-source and two-role portions are now approved and rendered under
  the two packets above. The bounded-replay extension still requires separate
  approval and exact/approximate numerical traces. Article creation and public
  publication remain outside this production authorization.

### Day 8 canonical material — 2026-09-09

Source: agent automatic mathematics/opportunity check during the learner-requested
Chapter 6 and notebook build. Reference: `book/chapters/06-pretraining-as-a-controlled-system.md`
and `notebooks/day-08/`. Static reference figures are not produced animation.

- Extend **CAND-ANIM-003 / ANIM-NTP-001**: follow document bytes into shifted
  windows; fade ignored targets; accumulate summed NLL divided by the same total
  N across microbatches. Canonical relation: gradient(mean over valid targets)
  = sum of microbatch summed-loss gradients / N. Contrast a one-target tail with
  sixteen targets and visibly expose why equal microbatch weights are wrong.
  Notebook 1 verifies the correct and broken gradients with the actual decoder.
- Extend **CAND-ANIM-018 / ANIM-EMB-001**: show actual decoder gradient coordinates
  entering AdamW's m and v history, bias correction, adaptive update and separate
  decay. Pair with the update-clock warmup/cosine schedule and a clip-after-sum
  intervention. Canonical formulas are Chapter 6 sections 6.7–6.9. Preserve the
  distinction between gradient norm, optimizer update and observed next loss.
  Do not replay the isolated quadratic as the default: learner found it boring.
  Notebook 2 verifies the recurrence and clipping, not optimizer superiority.
- Precision/memory extension to the same stability segment: contrast range and
  resolution with the actual FP16/BF16 casts, then distinguish 16P persistent
  bytes from total memory. CPU cast evidence is not measured GPU BF16 training.

### CAND-ANIM-019 — A checkpoint freezes a process, not only weights

- Source: agent automatic transition/state check, Day 8 material creation.
- State: candidate only; production not approved; Mac Studio production lane.
- Mechanism: at update12 freeze weights, Adam moments, shuffle cursor/RNG,
  update/token clocks and recipe identity. Branch into uninterrupted continuation,
  complete restoration, missing moments, and missing cursor through update24.
- Motion: align the first two trajectories; show the same next data but changed
  update in the missing-moments branch; show changed next data in the cursor
  branch. Keep “file loads,” “loss is finite,” and “trajectory replays” distinct.
- Evidence: Notebook 3 runs the exact small decoder branches; full local CPU
  replay matches, both omissions differ. Use report values for labels. Never
  imply bitwise Mac/Spark equivalence or distributed recovery coverage.
- 2026-10-05 extension: use the [durable-boundary packet](../../learning_artifacts/day-12-sft-mechanics/durable-training-boundaries.md)
  to contrast crash-before-commit with commit-before-metric failure, preserve the
  original DPO reference, and later show RLVR's retained post-collection pool.
  The shared format has CPU failure-path evidence; actual runner and Spark
  replay gates remain distinct. Reuse this candidate, not a new commissioned film.
- Current source extension: actual SFT/DPO/RLVR CPU gates now verify complete
  history/original reference and pending-pool application without resampling.
  Animate a pending group spending collection work before its optimizer step;
  after restart, consume it once instead of drawing a favorable replacement.
  Add a separately labeled cost inset: one DPO pair branches into chosen/rejected
  on both policy and reference, so update geometry is not total logical forwards;
  evaluation/replay/export counters remain separate. Source-derived bounds in
  [the supervisor plan](../../docs/PRODUCTION_SUPERVISOR_PLAN.md) are not measured
  GPU FLOPs or backward recomputation. Source: agent automatic mathematics/state
  capture,2026-10-05. Extend this candidate only; no media production approved.
- Whole-job extension,2026-10-05: freeze the numerical checkpoint cursor and a
  separate cumulative work-journal cursor. Spend a reservation after the snapshot,
  crash, restore old weights, then retain the later attempt charge before retry.
  Contrast this with the invalid animation of rewinding both cursors. In a storage
  inset, show old payload/new partial/temporary marker coexistence and reserve
  entries before creation. Source: Chapter14 section14.8 and worked answers21–24;
  numbers and backend observations are labelled source bounds/authored controls,
  not actual quota, GPU clearance or model-scale replay. Reuse this Mac candidate;
  no new rendering or publication authority is implied.
- Recovery-validation extension: use canonical Chapter6/solutions and the
  current Chapter14 answers25–28. Additional source inset: retain three histories
  (numerical state, spent work, partial storage), charge a pending-pool validation
  forward/draw replay, then restore weights without refunding either ledger.
  The snapshot report's six completed versus ten charged updates and eight-byte
  partial/256-byte reservation are tiny CPU controls, not physical/GPU claims.
  Preserve cached SFT versus uncached DPO/RLVR dispatch. No rendering approved.
- Semantic-validation extension,2026-10-05: Chapter14/answers29–30 separate the
  live sampler from a temporary verifier. For $u$ completed updates and
  accumulation $a$, animate $ua$ private replay draws while the live cursor
  stays fixed; label repeated callback/restore panels as separate work. Show
  generic byte/load/tree scans before the semantic callback, then the proposed
  independent pre-read receipt outside the checkpoint. Source: agent automatic
  mathematics/state capture and the inspection-budget plan. Do not label pending
  reader integration implemented or infer CPU units as physical/GPU guarantees.
  Reuse this candidate; Mac rendering/publication remains separately approved.
- Dependency: approve storyboard, use canonical Chapter 6/solutions and the
  verified notebook manifest, then create a complete portable ANIM task packet.
  No rendering or public publication authorized by this capture.
- Shared-reader extension,2026-10-05: reuse Chapter14/answers30–32 and Day25
  Exercise7. Move a small retained receipt outside the tensor payload, bind the
  same journals, reserve, then hash/load/check/apply. Show two equal returned
  tensors and a rejected third load; three load reservations remain after
  reopen, and the fourth read never starts. Animate save completion entering
  later journal history rather than appearing inside its own already-written
  payload. Pair $W_{\mathrm{fresh}}=\sum_{c\in C}S_c+D L_U$ with
  $W_{\mathrm{resume}}(k)=I_k+L_k+S_k+\sum_{c\in C,\ c>k}S_c+D L_U$;
  label componentwise capacity as a conservative schedule, not measured work
  or retry authority. Source: agent automatic mathematics/state capture; CPU
  source/actual tiny reader evidence only. No physical/pretrained/cross-host
  guarantee or new media approval; rendering stays a separately approved Mac task.

Native-RLVR extension to CAND-ANIM-019,2026-10-05: use Chapter13§13.7.1,
Chapter14/answer33 and Day25 Exercise8. Freeze two applied updates in both
lanes; move the pending lane's source/RNG cursor through the retained response
IDs before the crash, then apply that exact pool without a new draw. The
completed lane collects first. Show matching final native policy/reference/Adam
histories alongside the two physical journals retaining later rejected reads
and validation work. Label the manually measured one-boundary trace separately
from source-derived $1+2U$, $1+2(U-k)$ and $2+2(U-k-1)$ publication counts.
Source: automatic mathematical/state capture and the predeclared native reader
lesson. Candidate only: no media, article publication or Mac replay commissioned.

Measured native-recovery extension to the same **CAND-ANIM-019**, with the
existing **CAND-ANIM-027** group-advantage lane,2026-10-05: use
[the actual reasoning/RLVR report](../../experiments/reports/2026-10-05-native-reasoning-rlvr-evidence.md),
Chapter13/answers22–24 and the focused Day23 artifact. Keep failed G4/01 visible
beside accepted G4/02 and G8/01; all twelve recovery checks and final ten-component
equality are measured, not quality evidence. Move numerical cursors to2 while
physical histories retain four applications/three collections, then apply each
saved pending pool without a new draw. Split G8's384 dense sampled slots,
314 valid fresh actions and434 applied targets into separate counters. In a
reward/KL inset, show all-zero task advantages beside the recorded tiny first
gradients and nonzero later exact-KL/AdamW movement; do not claim task learning
or a causal explanation of the historical cleanup timeout. Pair with the
existing adjacent CPU zero-loss/gradient control, explicitly distinguishing its
nonconstant advantages from these constant native reward groups. Preserve
recovery cap16 versus pilot cap64 versus evaluation caps32/128, BF16/SDPA
training versus BF16/eager generation, and full-support/stopping/mask boundaries.
Actual G4 pilot01 subsequently reaches native16/export/exit0 but fails terminal
writer acknowledgment: add a separate retention lane, not a fake rollback of
sixteen spent applications or an accepted quality parent. Its1508 dense slots,
1004 valid targets,33 saves and64 natural zero-reward responses are measured;
G8 pilot/common20 were not started after that failed entry. Suggested X angle:
“Zero rewards do not mean zero updates.” Mac Studio assigned production lane,
candidate/suggestion only; no film, article draft, rendering, upload, publication
or new model experiment commissioned.

Story-work extension to CAND-ANIM-019,2026-10-05: use Chapter6's two-clock
recovery explanation and worked answer20. Freeze nine successful target tiles,
admit an eight-tile group, fail after computation starts, and restore the nine
tiles without rewinding the work-journal cursor. Retry the same group: successful
exposure becomes17 while admitted target places reach25. Animate the vector
condition $\mathbf{W}_{\mathrm{reserved}}+\mathbf{c}_{\mathrm{next}}\le\mathbf{C}$
componentwise; a remaining allowance in one dimension cannot pay for another.
Show cap refusal returning the stream untouched, versus partial optimizer
failure making the live session unusable. The [Day9 companion](../../experiments/reports/2026-10-05-story-work-lesson.md)
now supplies actual native CPU counters and a freshly executed two-panel plot;
cap24 refuses the retry with9 successful targets/17 reservations. Byte I/O, physical quota, story quality
and Mac/Spark equality remain outside the claim. Reuse this candidate only;
rendering/publication remains a separately approved Mac task.

Watchdog extension to CAND-ANIM-019,2026-10-05: use Chapter14/answer34.
Show a blocked model lane beside an advancing external clock. Then freeze the
memory-observer and incident-log lanes separately; the controller must still
advance to owned TERM/KILL/cleanup. Contrast a leader exit with its remaining
worker and keep shutdown, sampled memory and durable-log receipts as separate
status tiles. Use the native-profile source verification's actual inert outcomes
when available, not a fabricated GPU success. This is an automatic state-flow
capture for a later Mac storyboard; no rendering or publication is authorized.

### Interactive-learning design — 2026-09-09

Follow-up: first one-step SGD visual now demonstrated directly in conversation
on Spark (not a produced film). It reuses CAND-ANIM-018's mechanism. Full
momentum/AdamW interaction and animation production remain separate pending work;
the learner explicitly chose direct in-app interaction over notebook-only study.

Source: user request for real-time visual lessons and Mac/Spark workflow.
The proposed Day 8 Optimizer Playground reuses CAND-ANIM-018 and ANIM-EMB-001's
gradient/update teaching motivation. Preserve the distinction between a gradient
arrow and an optimizer update, matched initial conditions, and extra history
state for momentum/AdamW. This is a proposed interactive tool, not new measured
evidence or approval for a produced animation. The workflow and portable state
requirements are in `docs/LEARNING_WORKFLOW.md`; rendering remains on Mac Studio
only after explicit production approval. No duplicate animation candidate added.

### Day 7 chapter synthesis — 2026-09-07

Source: user-requested chapter update; agent automatic mathematics review.
Chapter 5 sections 5.23–5.27 reuse the embedding/CE backward packets for
`dL/dz = (p-q)/12` in the twelve-target mean-loss fixture; CAND-ANIM-014 for the
future-state → time-mean → earlier-state leak; CAND-ANIM-016 for incorrect cache
rotation offsets; and CAND-ANIM-017 for distinct query-count, KV-payload, and
whole-model-budget counters. The fixed-parameter/compute/time comparison in
5.25 strengthens CAND-ANIM-011's accounting caveats. Evidence is the executable
Day 7 teaching audit, not trained comparative quality. Two existing static
figures illustrate the prose. No duplicate animation or rendering task is
created; production still requires explicit approval and runs on Mac Studio.

### Day 7 defense notebook — 2026-09-07

Automatic mechanism check: the new core defense notebook supplies actual module
shape traces, parameter/cache accounting, backward connectivity, and two
controlled failure signatures. Reuse the existing embedding/attention/loss
packets and CAND-ANIM-014/016/017 rather than creating duplicate animations.
CAND-ANIM-011 and X-LOOP-001 can reuse its comparison-design worksheet; that
worksheet is a proposal, not a trained result or rendering approval. All media
production remains approval-gated on Mac Studio. Canonical evidence:
`experiments/reports/2026-09-07-day7-architecture-defense.md`.

### Day 6 narrative readiness — 2026-09-07

Source: agent automatic mathematics check during requested chapter creation.
Canonical material is Chapter 5 sections 5.11–5.22 and worked answers 13–24.
Refine existing proposals rather than creating duplicates:

- CAND-ANIM-014: contrast feature centering with RMS magnitude; show the constant
  vector and common-offset cases, epsilon, and the untouched residual bypass.
- CAND-ANIM-015: follow the two expanded branches through SiLU and multiplication;
  extend backward with dL/dc=delta*g and dL/da=delta*c*SiLU'(a). A zero forward
  gate is not necessarily a dead gradient path. Verify numeric motion before render.
- CAND-ANIM-016: preserve coordinate-pair norms and show R_m^T R_n=R_(n-m),
  then keep the causal mask correct while deliberately restarting decode offsets.
- CAND-ANIM-017: keep Hq attention distributions visible while compact Hkv
  storage shrinks. Distinguish tensor payload, dense FLOPs, and actual latency.
  Add Q/K RMS rescaling as an optional subscene; the lab's learned scales are
  shared across heads, and the original QKNorm paper uses a different formula.
- CAND-ANIM-011 / X-LOOP-001: both requested primary papers are linked in the
  canonical frontier introduction. Fixed sharing, variable depth, token-level
  routing, and explicit KV sharing remain distinct. The trained comparison and
  architecture defense still belong to Day 7; no quality result is inferred.

Static notebook previews and reference calculations are available. All animation
production remains on Mac Studio under the existing approval gates. This
chapter-writing request does not commission rendering, article drafting, or
public publication.

### Day 5 foundation synthesis — 2026-09-07

Source: agent automatic math-opportunity check while writing the requested
chapter foundation. Canonical source is now
`book/chapters/05-building-a-modern-decoder.md`, with worked answers in
`book/solutions/05-decoder-notebook-solutions.md`. Reuse existing candidates:

- CAND-ANIM-012: extend the multi-head scene backward through distinct W_O
  blocks, showing dL/do_r=g W_O,r^T. One shared loss need not send identical
  gradients to each head. Symmetric heads can remain duplicates; diverse
  initialization is not proof of learned semantic specialization. The scalar
  1-versus-2 gradient illustration is algebraic, not a trained-model result.
- CAND-ANIM-013: show the skip carrying X into addition, with only the sum
  passed onward. Preserve the distinction from an untouched recoverable backup.
- CAND-ANIM-014/015: canonical readable normalization and affine-collapse
  explanations are now in sections 5.5–5.6; keep position and feature axes clear.
- ANIM-CE-001 / ANIM-NTP-001: reuse section 5.8's twelve aligned target losses
  combining into one mean and one parameter update. Generation selects new IDs
  sequentially with fixed weights; training does not update after each token.
- CAND-ANIM-018 and ANIM-EMB-001 retain the optimizer and shared-parameter paths.

Existing notebooks verify the baseline mechanisms. This refinement captures
storyboard opportunities, not a new rendering commission. All production stays
on Mac Studio after the applicable explicit approval; no media produced here.

### CAND-ANIM-018 — Same gradient, different learning-rate steps

- Source: agent automatic math trigger following learner's optimizer question,
  2026-09-07; introduced in Chapter 5, deeper treatment in Chapter 6/Days 8–9.
- State: discuss; production not approved; rendering stays on Mac Studio.
- Mathematics: scalar L=(e-1)^2, gradient=2(e-1), SGD e_new=e-eta*gradient.
- Motion: place three identical markers at e=.2 on the same quadratic, expose
  the common gradient -1.6, then step with eta=.01, .1, and 1.5. Show small
  progress, larger progress, and overshoot to e=2.6 with increased loss. Keep
  parameter position, gradient sign, update arrow, and loss height distinct.
- Precision: illustrative quadratic, not an LLM loss landscape or recommended
  learning-rate range. Local descent direction does not guarantee a decrease
  for an arbitrary step. AdamW uses adaptive gradient-history statistics and
  must not be animated as plain raw-gradient SGD.
- Evidence: analytical one-step values .216, .36, 2.6; loss .64 to 2.56 in
  the large-step case. Verify rendered trajectories numerically before production.
- Dependencies: approve storyboard and connect to existing ANIM-EMB-001 optimizer
  segment without silently replacing its SGD sketch with an AdamW claim.

### Chapter 5 notebook evidence update — 2026-09-06

Source: agent automatic mathematics check while fulfilling the learner's request
to build the entire Chapter 5 notebook pathway. All eleven reference notebooks
execute successfully; evidence is in
`experiments/reports/2026-09-06-decoder-notebooks.md`.

- `CAND-ANIM-012`: head splitting, independent/vectorized agreement, head ablation,
  and causal invariance now have executable examples in Day 5 notebook 02.
- `CAND-ANIM-013`: zero-branch identity, direct/branch gradient sum, cancellation,
  and pre/post-norm contrast now have examples in Day 5 notebooks 03–04.
- `CAND-ANIM-011`: optional Day 7 notebook verifies fixed shared-block outputs,
  gradient accumulation, parameter/application accounting, and distinct K/V
  states across applications. This is not the planned trained architecture
  comparison or an adaptive-routing implementation. `X-LOOP-001` remains queued.
- Reuse `ANIM-EMB-001` and `ANIM-ATTN-001` for the embedding and attention paths
  rather than duplicating their existing storyboards.

### CAND-ANIM-014 — Centering, scaling, and the residual bypass

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: LN(x)=gamma*(x-mean(x))/sqrt(var(x)+eps)+beta;
  RMSNorm(x)=gamma*x/sqrt(mean(x*x)+eps).
- Motion: keep one position's feature vector together; show mean subtraction,
  scale normalization, then learned affine change. Contrast RMS scaling without
  centering. Zoom out to pre/post-norm placement and the residual bypass.
- Precision: feature axis, not time; epsilon prevents exact scale invariance;
  affine outputs need not have zero mean/unit variance. Wrong time-axis
  normalization can leak the future despite a valid attention mask.
- Evidence: Day 5 notebook 04 and Day 6 notebook 01 forward/backward references,
  constant-vector and wrong-axis controls. Canonical source: Chapter 5 lab.
- Dependency: learner approval and final storyboard; production on Mac Studio.

### CAND-ANIM-015 — Nonlinear and gated feature transformations

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: down(GELU(up(x))) versus down(SiLU(gate(x))*up(x)); removing
  nonlinearity collapses two affine layers, including their combined bias.
- Motion: expand one position's features, transform them, and contract. Reveal
  a second gate branch and elementwise multiplication; zero it and watch the
  bias-free output disappear. Keep token positions fixed throughout.
- Precision: SiLU is not a probability; three matrices alter parameter budgets;
  isolated MLPs do not directly mix tokens but can read contextual input states.
- Evidence: Day 5 notebook 05 and Day 6 notebook 01, affine-collapse and
  positionwise-gradient checks. No trained quality ranking.
- Dependency: learner approval and canonical storyboard; render on Mac Studio.

### CAND-ANIM-016 — Relative angles and a cache position mistake

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: R(theta)(a,b)=(a*cos(theta)-b*sin(theta),a*sin(theta)+b*cos(theta));
  (R_m q) dot (R_n k) = q dot (R_(n-m) k).
- Motion: rotate coordinate pairs while preserving length; shift both positions
  together and keep their dot product fixed. Retain old rotated keys in a cache;
  contrast correct continuing query positions with a mistaken restart at zero.
- Precision: adjacent-pair toy layout, base 10,000; not every checkpoint layout.
  Fixed-vector geometry does not establish long-context generalization.
- Evidence: Day 6 notebook 02 norm/relative-position/gradcheck tests and full
  decoder cached replay with an intentionally wrong offset.
- Dependency: learner approval; coordinate with the cache candidate; Mac only.

### CAND-ANIM-017 — Many queries, compact shared keys and values

- Source: agent, automatic math trigger, 2026-09-06; Chapter 5.
- State: discuss; production not approved.
- Mathematics: Hq/Hkv query heads share each KV head; logical cache payload
  equals 2*layers*batch*Hkv*length*head_width*bytes_per_element.
- Motion: retain four query distributions while KV stores contract from four
  to two to one head. Track source sharing, gradient accumulation, and compact
  cache bytes independently from query attention arithmetic.
- Precision: grouped sharing is not dropping query heads; logical payload is not
  allocator memory or measured speed. The teaching kernel expands KV temporarily.
- Evidence: Day 6 notebook 03 grouped/reference forward and backward checks,
  MHA/MQA endpoints, compact cache bytes, and analytical parameter agreement.
- Dependency: learner approval; consider extending the existing cache candidate
  rather than commissioning a separate film; production on Mac Studio.

### CAND-ANIM-013 — Attention as an update to the residual stream

- Source: agent, automatic mathematics trigger, 2026-09-06
- Book placement: Chapter 5, residual connections
- State: discuss; production not approved
- Equation: X_next = X + MHA(X), initially omitting norm/dropout; later reveal
  the pre-norm form X + MHA(LN(X)).
- Motion: keep the incoming [T,D] stream visible while a branch computes a
  same-shaped update; add coordinatewise. Set the update to zero and retain
  the unchanged input. Reverse gradients through the direct and learned paths.
- Precision: addition differs from feature concatenation; X is the current
  state, not always the original embedding. The identity path does not guarantee
  information preservation or prevent all gradient cancellation.
- Evidence: forward zero-update, backward split, and cancellation controls are
  verified in `day-05/03_residual_stream.ipynb` and the 2026-09-06 report.
- Source: `learning_artifacts/day-05-decoder-only-transformer/residual-stream-and-attention-updates.md`.
- Production: Mac Studio after explicit approval; consider scope alongside the
  existing multi-head candidate rather than assuming an additional film.

### CAND-ANIM-012 — Several retrieval mixtures for one position

- Source: agent, automatic mathematics trigger; introduced 2026-09-06
- Book placement: Chapter 5, bridge from single-head attention
- State: discuss; no production approval
- Mechanism: O_h=A_h V_h, then O_multi=Concat(O_1,...,O_H) W_O.
- Motion: preserve X and receiver identity while multiple head-specific Q/K/V
  branches create different weight rows and value mixtures; concatenate feature
  segments and project them back to model width, leaving token positions fixed.
- Precision: heads see the allowed prefix rather than disjoint token subsets;
  fixed human-readable roles are not guaranteed; W_O differs from the vocabulary
  head; concatenation is across features and is not averaging.
- Evidence: executable split/merge, ablation, and causal controls are in
  notebook 02 and the 2026-09-06 report; Chapter 5 section 5.3 now supplies
  the canonical narrative. No trained specialization experiment is claimed.
- Source artifact: `learning_artifacts/day-05-decoder-only-transformer/multiple-heads-and-output-projection.md`.
- Next decision: consider after multi-head implementation and verification.
  Production requires explicit approval and belongs on the Mac Studio.

| Candidate ID | Source | Book placement | Mechanism | State | Dependency or next decision |
|---|---|---|---|---|---|
| `CAND-ANIM-010` | User | Chapter 2 supporting example | Five supplied Chinese strings → counted BPE merges `流 + 星`, `流星 + 雨` → retained vocabulary → illustrative IDs 1–5 → encoding | review; promoted into `ANIM-BPE-002` | 40.23-second 1080p Mac Studio render ready on 2026-09-03; round-2 tie disclosed; source integration approved on 2026-09-04; final visual approval pending |
| `CAND-ANIM-001` | Roadmap + agent, automatic math trigger confirmed | Chapter 3 | Input tokens → contextual state `[D]` → dense projection against all `[V,D]` output rows → `[V]` logits → probabilities → selected vocabulary index/token ID → append and repeat, while next-token targets align with preceding positions; optionally reveal the `T`-versus-`V` compute trade-off | discuss | Canonical Chapter 3 treatment is complete; decide on the Mac Studio whether to expand `ANIM-CE-001` or defer this separate animation |
| `CAND-ANIM-002` | User | Chapter 3 | Place negative log-likelihood and cross-entropy in the LLM training context: target probability → per-token NLL → masked aggregation across next-token positions → cross-entropy | approved; promoted into `ANIM-CE-001` | Day 3 derivation and verification complete; produce only on the Mac Studio |
| `CAND-ANIM-003` | User | Chapter 3 | Standard next-token update: predicted distribution `p` versus one-hot target `q` → logit gradient `p-q` → target score rises and non-target scores fall → repeated diverse examples shape a distribution | approved; promoted into `ANIM-NTP-001` | Gradient and tiny-model evidence complete; produce only on the Mac Studio |
| `CAND-ANIM-004` | User | Chapter 3 | Why negative log loss: sequence probability products become additive token surprise; confident errors retain strong gradients; expected log loss rewards matching the full data distribution | approved; promoted into `ANIM-LOGLOSS-001` | Canonical derivations and controlled evidence complete; verify the `1-p_target` comparator during Mac production |
| `CAND-ANIM-005` | Agent, automatic math trigger | Chapter 3 | Mean token cross-entropy → exponentiation → perplexity as an effective equal-choice branching factor, while preserving tokenizer and evaluation-distribution dependence | discuss | After derivation, decide whether to extend `ANIM-CE-001` or create a separate short; production only on Mac Studio after approval |
| `CAND-ANIM-006` | Agent, automatic math trigger from learner question | Chapter 2–3 bridge | Consistently permute token IDs, dataset symbols, embedding rows, and output rows while decoded text behavior remains unchanged; reveal that IDs are categorical addresses rather than numerical linguistic features | discuss | Prefer extending `ANIM-EMB-001` if the permutation can remain concise; otherwise defer beyond v0.1. Production only on Mac Studio after explicit approval |
| `CAND-ANIM-007` | Agent, automatic math trigger from learner question | Chapter 3 decoding bridge | Hold logits fixed while temperature continuously rescales their gaps: low temperature sharpens, high temperature flattens, ranking stays fixed, and exact tied maxima reveal the difference between greedy tie-breaking and sampling | discuss | Verify ratios and limiting behavior in the Day 3 notebook; likely defer or use as a compact decoding short. Production only on Mac Studio after explicit approval |
| `CAND-ANIM-008` | Agent, automatic math trigger; approved by user | Chapter 4 | Input states → learned `Q`, `K`, and `V` projections → scaled query-key score matrix → causal mask → row-wise attention distributions → weighted value retrieval; reverse the loss gradient through a value/content path and a query-key/routing path; contrast broken scaling and masking | approved; promoted into `ANIM-ATTN-001` | Chapter, forward, gradient, finite-difference, mask, and detach evidence verified on 2026-09-05; ready for Mac production |
| `CAND-ANIM-009` | Agent, automatic math trigger from learner inference | Chapter 4–modern decoder bridge | Contrast the same token in two contexts to establish distinct request-local K/V states; show optional runtime retention across prefill and decoding, each transient new query reading the unchanged-prefix cache, and logical cache release at sequence completion | discuss | Toy cache shapes and equivalence verified on 2026-09-05; lifecycle derived, no allocator benchmark; distinguish first-layer and contextual deeper-layer keys; production only on Mac after explicit approval |
| `CAND-ANIM-011` | User reminder + agent automatic math trigger | Chapter 5 frontier section | Reuse one visually identical Transformer stack for recurrent state updates; let stored-parameter, effective-depth, and compute counters diverge; later consider adaptive exit and token-space reasoning | fixed-recurrence act approved as `ANIM-LOOP-001`, 2026-09-09 | Mac production from the verified Day 7 block; adaptive routing and trained comparison remain pending |

Q/K/V workflow refinement (user request, 2026-09-06): extend `CAND-ANIM-008`
and its approved `ANIM-ATTN-001` packet. Keep the V content branch separate
from Q/K scores and softmax weights until O=AV. Include fixed-Q/K, changed-V
and fixed-V, changed-routing contrasts. See the packet for precision and
Mac production requirements; no duplicate candidate is created.

### CAND-ANIM-002 — NLL to cross-entropy in an LLM

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-CE-001`
- Learning objective: Show that, for a one-hot next-token target, token-level
  cross-entropy equals the negative log-likelihood of the observed token, and
  that the training loss aggregates those terms only across valid target
  positions.
- Why motion is better than a static figure: The same target token must retain
  its identity while its probability is selected, transformed by `-ln`, repeated
  across causal positions, filtered by the loss mask, and reduced into one scalar.
- Moving objects and stable anchors: Keep the token sequence and target alignment
  stable; move from probability bars to highlighted `p_y`, per-position NLL
  tiles, masked/ignored positions, and the final mean cross-entropy.
- Canonical source material:
  `book/chapters/03-learning-the-next-token.md` and
  `learning_artifacts/day-03-probabilities-and-next-token-loss/`.
- Evidence status: derivation, target alignment, masked mean, and PyTorch
  reference computations are complete.
- Precision risks and required caveats: Do not present NLL and one-hot
  cross-entropy as different numerical objectives; distinguish sequence NLL sum
  from the common mean over valid tokens; make visible that target label index
  `k` pairs with logit index `k-1`; exclude padding or ignored labels; retain
  natural logarithms; do not imply that low loss proves truthfulness.
- Dependencies: Day 3 derivation, causal label alignment, loss-mask denominator,
  and PyTorch agreement are complete.
- Suggested destination: Chapter 3, course site, and the embedding-training
  animation handoff.
- Next decision: Mac Studio production may begin from the committed Chapter 3
  material when the learner starts the animation task.

### CAND-ANIM-003 — Standard next-token training update

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-NTP-001`
- Learning objective: Make one-hot next-token supervision and its softmax
  cross-entropy gradient visible: `dL/dz = p-q`. Show why one example rewards
  the observed token and locally suppresses every non-target, including
  alternatives that could be valid in another sample.
- Why motion is better than a static figure: The learner needs to preserve token
  identity while probability bars become gradient bars, logits move in opposite
  directions, and repeated examples with different observed targets accumulate
  into a learned conditional distribution.
- Moving objects and stable anchors: Keep candidate tokens and their colors fixed;
  place `p` beside one-hot `q`; transform them into `p-q`; move the target logit
  upward and non-target logits downward; continue the signal through
  `z = Wh+b`, splitting it into `dW = (p-q)h^T` and `dh = W^T(p-q)`; then replay
  a small controlled stream whose target frequencies are visibly known.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  `book/chapters/03-learning-the-next-token.md`; and the committed tiny-model
  report.
- Evidence status: analytical gradient, PyTorch agreement, optimizer trajectory,
  and controlled 70/30 learned-frequency convergence are verified and reported.
- Precision risks and required caveats: Show gradient descent direction rather
  than confusing gradient sign with parameter motion; state that one-hot
  supervision does not mark unobserved alternatives as valid; do not imply that
  one update sets the target probability to one; distinguish a single local
  update from the expectation over a representative data distribution; use fixed
  verified numbers in the final render.
- Dependencies: core analytical gradient, PyTorch agreement, optimizer-step
  trajectory, and controlled target-frequency experiment are complete;
  output-head chain-rule verification remains for the expanded version.
- Suggested destination: Chapter 3, course site, and the later embedding-gradient
  animation sequence.
- Next decision: Produce and render only on the Mac Studio; decide there whether
  it is a standalone short animation or a
  companion segment to `ANIM-CE-001`.

### CAND-ANIM-004 — Why negative log loss has three jobs

- Source: user
- Proposed during: Day 3
- State: approved; promoted into `ANIM-LOGLOSS-001`
- Learning objective: Explain why `-log p_target` is structurally suited to
  language modeling rather than merely displaying its curve.
- Why motion is better than a static figure: Three transformations occur over
  time: conditional probability factors combine into a sequence product and
  unfold into additive surprise; competing loss choices produce different
  gradient strength near confident errors; repeated outcomes reveal whether a
  scoring rule recovers a distribution or collapses onto its mode.
- Moving objects and stable anchors: Preserve the same target probabilities and
  candidate colors across three acts. Act 1 moves probability factors from a
  product into additive NLL tiles, then contrasts two equal-sum sequences whose
  surprise is either diffuse or concentrated in one catastrophic token. Act 2
  synchronizes loss and gradient curves for log loss versus `1-p_target`. Act 3
  reveals a hidden population distribution `q` through a finite stream of
  one-hot samples and their evolving empirical frequencies `q_hat`, then moves a
  model distribution `p` toward the expected-loss minimum while distinguishing
  generalization from memorization.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/probability-as-competition-and-surprise.md`;
  `book/chapters/03-learning-the-next-token.md`; and the controlled Day 3 report.
- Evidence status: sequence, logit-gradient, and proper-scoring mechanisms are
  derived; $p-q$ autograd and controlled expected-loss behavior are verified.
  The alternative `1-p_target` softmax-gradient comparison remains a Mac
  production check.
- Precision risks and required caveats: Distinguish loss magnitude from gradient
  through softmax; do not claim `1-p_target` itself has a small value for a
  confident error—its softmax gradient becomes small; distinguish one-hot sample
  targets from the population conditional distribution; show expected loss when
  discussing proper scoring; do not use KL nonnegativity before defining it.
- Dependencies: sequence chain rule, log-product identity, analytical and
  autograd gradients for both losses, and a controlled expected-loss comparison
  under a fixed categorical `q`.
- Suggested destination: Chapter 3 and course site.
- Next decision: Mac Studio chooses one three-act animation or three coordinated
  shorts after the verified Day 3 evidence is committed; every act must remain in
  the approved scope.

### CAND-ANIM-005 — Cross-entropy to perplexity

- Source: agent, automatic mathematics trigger
- Proposed during: Day 3
- State: discuss
- Learning objective: Show why exponentiating mean natural-log token loss returns
  to probability scale and yields an effective branching factor, not a literal
  count of equally likely next tokens.
- Why motion is better than a static figure: A mean surprise value in nats is
  abstract. Motion can transform equal-choice distributions with $k$ candidates
  through `ln(k)` loss and back through `exp` to $k$, then morph to unequal
  distributions with the same perplexity while keeping their different shapes
  visible.
- Moving objects and stable anchors: Keep the tokenizer, valid-position mask, and
  evaluation corpus fixed; move per-token NLL tiles into their mean, exponentiate
  the mean, and compare equal and unequal distributions sharing an effective
  branching factor. Then hold one text probability fixed while splitting one
  token into two, showing total NLL remains fixed as mean token NLL and perplexity
  change solely because the counting unit changed.
- Canonical source material:
  `learning_artifacts/day-03-probabilities-and-next-token-loss/nll-cross-entropy-and-perplexity.md`;
  `book/chapters/03-learning-the-next-token.md`.
- Evidence status: formula, geometric-mean interpretation, and tokenizer-unit
  counterexample are analytically demonstrated; executable verification remains
  pending.
- Precision risks and required caveats: Use natural logs so `PPL=exp(mean NLL)`;
  do not call perplexity the literal number of available tokens except in the
  equal-probability teaching case; compare only under the same tokenizer, target
  mask, and evaluation distribution; distinguish population entropy from finite
  test loss; do not imply that lower perplexity alone proves better capability.
- Dependencies: verified mean-loss calculation, equal-choice example, unequal
  distribution counterexample, and tokenizer-comparison limitation.
- Suggested destination: Chapter 3 and course site.
- Next decision: Discuss after the perplexity lesson; if approved, Mac Studio
  decides whether it is the final act of `ANIM-CE-001` or a separate short.

### CAND-ANIM-011 — Reusing depth without pretending compute is free

- Source: user reminder plus agent automatic mathematics trigger
- Proposed during: Day 4 as a future Chapter 5 module
- State: fixed-recurrence first act approved on 2026-09-09; promoted to
  `ANIM-LOOP-001`. Adaptive routing remains outside production scope.
- Learning objective: Show how one parameterized block or stack can be applied
  repeatedly, increasing effective depth and compute without duplicating its
  stored weights; distinguish fixed recurrence from adaptive token-level depth
  and from visible chain-of-thought tokens.
- Why motion is better than a static figure: The same weights must retain their
  identity while successive hidden states revisit them. Independent counters
  must make clear that parameter storage can stay fixed while layer applications,
  latency, and transformation depth grow.
- Moving objects and stable anchors: Keep the recurrent block and its parameter
  label fixed. Move $s_0,s_1,\ldots,s_r$ through it under
  $s_j=R(e,s_{j-1};\theta_R)$; update
  $L_{\mathrm{effective}}=L_P+rL_R+L_C$ as $r$ changes while the recurrent
  parameter counter remains fixed. In a second act, let token positions exit at
  different depths under a router, then contrast hidden-state loops with
  separately emitted reasoning tokens.
- Canonical source material:
  `learning_artifacts/day-04-attention-and-causal-information-boundary/future-recurrent-depth-and-looped-transformers.md`;
  [*Mixture-of-Recursions*](https://arxiv.org/abs/2507.10524) for adaptive
  token-level routing; the future Chapter 5 treatment; and the planned Day 7
  comparison.
- Evidence status: Chapter 5 section 5.20 and the Day 7 notebook now provide
  verified fixed sharing, gradient accumulation, and parameter/application
  accounting (September 6 update above). A trained architecture comparison and
  adaptive-routing implementation remain pending.
- Precision risks and required caveats: Do not say that effective depth is model
  parameter size; do not promise linear quality gains; do not call shared and
  untied layers equivalent; do not equate latent recurrence with hidden or
  suppressed chain-of-thought; do not present the reported Astra architecture as
  verified.
- Dependencies: complete the ordinary decoder, implement the optional recurrent
  variant, and verify parameter-, compute-, and wall-clock-accounting examples.
- Suggested destination: Chapter 5 and course site; possible later architecture
  article only after the canonical treatment is stable.
- Next action: produce the approved fixed-recurrence first act as `ANIM-LOOP-001`
  on Mac Studio, then review it with the learner.

## Two-way proposal mechanism

### Agent-suggested

During a lesson or course-development session, the agent should actively suggest
an animation when motion would materially clarify at least one of these:

- a state transition or ordered computation;
- object identity across several representations;
- information, gradient, or mask flow;
- competition among alternatives;
- a failure that emerges over time;
- a relationship that becomes misleading when reduced to a static figure.

The suggestion should be short and concrete: name the learning objective, the
objects that move, the decisive transition, and the current evidence boundary.
Do not propose animation merely to decorate a section.

### Automatic mathematics trigger

Do not wait for Dongxi to request an animation when explicit mathematics is
central to an LLM mechanism. Automatically record a candidate for:

- objectives and losses;
- probability transformations and sampling distributions;
- analytical gradients and credit assignment;
- tensor operations whose shapes or axes change;
- causal, attention, padding, or loss-mask mathematics;
- optimization updates and training dynamics;
- multi-step derivations whose intermediate identities must remain visible.

The trigger applies to important mechanisms, not every decorative equation.
Consolidate equations that form one argument into one animation concept, and
check the queue before adding a new ID. Each automatically captured candidate
must state the canonical equation, why motion helps, current evidence state,
precision risks, and the derivation or experiment required before production.
Use source `agent` unless the learner or roadmap originated the idea.

Automatic candidate capture is not production approval. The candidate remains
`suggested` or `discuss` until Dongxi explicitly approves it, and all production
and rendering remains on the Mac Studio.

### User-suggested

Dongxi can propose an idea at any time with ordinary language, for example:

> Animation idea: show how each next token becomes the training target for the
> preceding position.

The agent records it here with `Source: user`, links it to the relevant chapter
or learning artifact, and identifies any conceptual or empirical dependency.

## States and approval gate

1. `suggested` — captured without evaluation.
2. `discuss` — the mechanism, scope, and learning value need joint review.
3. `approved` — Dongxi has approved production; create or update the `ANIM-*`
   task packet in `LEARNING_MEMORY.md`.
4. `producing` — editable source and review media are being created.
5. `review` — content accuracy and visual quality are ready for inspection.
6. `done` — approved source, render command, media, metadata, and integration
   location are preserved.
7. `deferred` or `rejected` — retain the reason and any reconsideration trigger.

No candidate moves from `discuss` to `approved` without Dongxi's explicit
decision. Approval of the concept does not approve publication.

## Proposal template

```markdown
### CAND-ANIM-NNN — short mechanism name

- Source: user | agent | roadmap
- Proposed during: Day NN / chapter / article
- State: suggested | discuss | approved | producing | review | done | deferred | rejected
- Learning objective:
- Why motion is better than a static figure:
- Moving objects and stable anchors:
- Canonical source material:
- Evidence status: conceptual | executable teaching example | measured result
- Precision risks and required caveats:
- Dependencies:
- Suggested destination: book | lab | X article | course site
- Next decision:
```

## Scope discipline

The `v0.1` target remains four excellent signature animations. Additional ideas
may become short supporting animations or move to the `v0.2` backlog. Recording
an idea is cheap; production competes for course-development time and requires an
explicit priority decision.
