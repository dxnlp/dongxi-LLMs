# Repetition is not by itself evidence of overfitting

2026-09-14. Learner reported similar continuations despite parameter changes,
including repeated "big and scary" descriptions, and proposed overfitting.
Exact user prompt/settings are not retained by the playground and were requested.
This is diagnosis only; no model, decoding policy or training recipe was changed.

## Checked evidence

- Fixed512-window validation NLL decreased at every saved observation through
  update14000:3.3941 at400,1.8282 at8000,1.6743 at14000. There is no validation
  deterioration in this series while training loss improves.
- Last500 online training batch losses average1.661964; final fixed development
  NLL1.674315. These are different samples and averaging/time conventions, not
  an exact matched train/evaluation generalization-gap measurement.
- Four direct generation API requests used the assumed opening "Once upon a
  time, there was a little girl." and max128 new tokens: (temperature,seed)
  (0,909),(.8,909),(.8,910),(1.2,909). All four continuations differed.
  Greedy output repeated climbing a ladder. Temperature.8 produced differing
  adventure/building narratives with semantic inconsistencies. Temperature1.2
  produced much more malformed language. Backend parameter effects work in
  these probes; this does not reproduce the learner's unknown exact inputs or
  verify the browser event flow.
- Literal search of the full raw training source found no match for the narrow
  fragment "She was only three years old. She was scared of the dark." This is
  not a near-duplicate/memorization audit and does not rule out copying elsewhere.

## Refined interpretation

The observed symptom is repetitive/degenerate free-running generation. A plausible
interpretation is that local familiar sentence patterns are stronger than global
narrative consistency. Narrow story templates, limited training and decoding can
all contribute; the current evidence does not uniquely assign cause or establish
overfitting. The run consumed48.84M presentations from390.71M prepared targets;
limited coverage is context, not proof that overfitting is impossible.

Distinguish controls: temperature0 is greedy and ignores sampling seed; fixed
prompt/positive temperature/seed repeats the same sampling trajectory in this
tested runtime; changing only maximum length normally extends/truncates that
trajectory. Higher temperature changes the distribution, not model knowledge,
and can amplify poorly learned alternatives rather than repair coherence.
An unfinished sentence can be a token-cap stop, not a natural model ending.

Next: obtain the exact prompt and compared settings; reproduce before changing
anything. Do not prescribe dropout, penalties or additional training as an
established fix from this one symptom. The existing checkpoint-comparison view
supports the discussion; no animation production authorized or needed here.
