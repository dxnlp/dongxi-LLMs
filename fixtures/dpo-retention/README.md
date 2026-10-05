# Original DPO retention fixtures

`contract.json` freezes the recipes, seeds, atomic symbol table, templates,
source groups and generation budget before fitting. First-slot copying uses
four color tokens; parity uses distinct numeral/result tokens. These are tiny
symbolic tasks, not natural-language data or human judgments. An independent
copy rule/integer arithmetic supplies evaluation targets.

The eight warm-up demonstrations also supply the four preference sources and
four rehearsal sources in their own training stages; that reuse is deliberate.
No held-out source/problem/prompt is used for training or tuning. All held-out
copy operands have seen token IDs but unseen ordered pairs. Held-out parity
operands use numeral tokens absent from warm-up, making that slice a deliberately
severe input novelty control, not a pure arithmetic extrapolation claim.

The noise control swaps fixed indices, not randomly selected good outcomes.
The length controls insert the declared STYLE token, never unseen lexical text.
They change the recorded demonstrations, while independent generation continues
to require the canonical short answer+EOS. Absolute likelihood, first-answer
accuracy and full completion success must remain separate measurements.
