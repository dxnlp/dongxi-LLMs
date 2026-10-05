# Preference verdict parser hardening

An independent acceptance review found that a deeply nested JSON value could
raise `RecursionError` and abort preference collection. The original
[results](2026-10-04-preference-audit.md), [484-observation ledger](2026-10-04-preference-audit.json)
and [earlier verification](2026-10-04-preference-audit-verification.json) remain
unchanged historical evidence. This report records the later source revision
and adversarial checks, not a replacement for those results.

## Correction and adversarial evidence

The collector catches decoder recursion exhaustion and records an invalid
parse-stage observation with the complete original raw string. It also limits
JSON decoding to 65,536 input characters, rejecting oversized verdicts while
retaining their full raw text. This bounds decoder work; it is not a truncation
policy or a claim that storing arbitrary raw output has bounded memory cost.
The accepted verdict grammar and canonical mapping are unchanged.

The regression inputs are a roughly 20KB verdict containing 10,000 nested
brackets in `reason`, and an oversized string-valued reason. Both yield invalid
outcomes, no selected candidate and a retained parse-stage error. Replaying the
nested raw record through the auditor also retains the failure rather than
raising. All twenty preference-audit tests pass, including the original eighteen
checks and these two adversarial regressions.

A current-source reference replay reproduces the complete original ledger
exactly, including all four earlier failure controls. Its SHA256 remains
`c5ae08b06077554cdcb5485d2c9720a06221b05462480e5b092f7a701bdeadf9`.
The fresh Day15 verification also passes: the audit notebook executes eight
code cells and produces the same three figure hashes. Across all three Day15
sessions, sixteen code cells execute and seven figures are produced without
skips or failures. Source notebooks remain unchanged.

The [hardening evidence JSON](2026-10-04-preference-audit-hardening.json)
preserves both complete adversarial records, current source/test hashes, actual
test count, original evidence hashes, replay equality, environment and fresh
notebook verification. Earlier source hashes remain historical rather than
being rewritten to imply they described the hardened collector.

No model, GPU campaign, live API, installation or notebook server was launched.
This verifies parser robustness on declared cases, not all possible denial-of-
service attacks or general judge quality.
