# Frozen story openings against the actual TinyStories corpus

The twelve unchanged publication openings have no normalized exact whole-story,
prefix or substring matches in the actual pinned corpus. Their defined lexical
near-overlap scores also stay below the predeclared0.4 threshold. This resolves
a bounded corpus-audit subgate, not global contamination absence, semantic
novelty, a trained comparison or actual story ratings.

The [protocol](2026-10-05-story-panel-audit/protocol.json) was saved before the
scan. It reads the original frozen panel and the actual three repeatedly
inspected development openings; it does not replace either based on the results.
Both scans used existing local bytes, with no acquisition, weights, inference,
training, service, remote execution or Git publication.

## Corpus and split reconstruction

The dataset revision is `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
Actual raw training bytes total1,924,281,556; validation bytes total19,447,282.
Their SHA-256 values match the historical pinned manifest, as do all four
prepared token/window files. Streaming reconstruction follows the original
validation-first normalized exact deduplication policy, including the final
undelimited document in each raw file.

| Split | Raw stories | Retained stories | Within-split duplicates | Excluded validation matches |
|---|---:|---:|---:|---:|
| Training |2,119,489 |1,792,647 |320,241 |6,601 |
| Validation |21,990 |21,990 |0 |0 |

The retained document counts and length-framed raw-text digests reproduce the
historical manifest exactly. Normalization joins casefolded whitespace-separated
text; punctuation remains. Zero retained normalized exact cross-split collisions
therefore does not mean zero paraphrases, shared themes or upstream leakage.

## Publication and development findings

All twelve openings have zero exact whole-story, prefix and substring matches
across both splits, including the discarded duplicate/excluded categories.
For near overlap, compare each opening's consecutive word-trigram set with the
first equally many words of every story. The score is their intersection divided
by union. This is an explicitly lexical, prefix-limited definition, not a
semantic similarity model or a search through every internal passage.

| Population | Maximum training score | Maximum validation score | Hits at or above0.4 |
|---|---:|---:|---|
| Twelve publication openings |0.125 |0.090909 |0 in either split |
| Historical rabbit development opening |0.5 |0.384615 |1 retained training hit |

The development near-match is retained, not removed to improve the audit.
The other development openings have training maxima0.111111 and0.2.
All36 publication/development pairs are distinct under normalized equality,
either-direction prefix/substring checks and the defined whole-opening
trigram comparison. This separates the literal opening populations; it cannot
certify that no human or agent ever encountered related material.

## Actual execution and memory measurements

The original [run01](2026-10-05-story-panel-audit/run-01/audit.json) exits0 in
60.255386s. Its `getrusage` high-water field is832,471,040bytes; that number
alone is not an observed audit-specific RSS peak. The separately retained
[run02](2026-10-05-story-panel-audit/run-02/audit.json) repeats the same science
with an explicit diagnostic, exits0 in58.378660s and reproduces the findings.

Run02 observes the same832,471,040-byte `getrusage` value at process start and
finish. Linux `/proc/self` instead reports final lifetime-since-exec
VmHWM39,374,848bytes and VmPeak95,809,536bytes. Actual soft/hard address-space
limits are both536,870,912bytes. Keep these distinct counters: a512MiB address-
space limit is not a declaration that inherited resource attribution equals
current-process RSS. No reported number is silently replaced.

The second scan retains215 point samples of host availability, minimum
126,300,545,024bytes (117.62655GiB). This exceeds the predeclared25GiB sampled
reserve but is not continuous monitoring. Each hash-only SQLite scratch index
is74,129,408bytes, retained locally under a narrow ignore rule; small receipts
retain its actual digest without publishing a large corpus-derived index.

Thirteen bound source/input identities remain stable during each scan. Prepared
file bytes are hashed before collection; file metadata stays unchanged afterward.
The [independent verification](2026-10-05-story-panel-audit/independent-verification.json)
passes nineteen checks, inspecting ninety metric groups and recomputing all
sixty-four retained nearest-example scores. It is a separate receipt review,
not another full corpus scan or pretrained-model test.

## Integration and remaining work

Chapter6 keeps lower loss separate from story coherence; Chapter7's new rating
instrument keeps the audit, real continuations and independent review separate.
The original panel contract and its pending-status bytes are preserved: this
report supplies external audit evidence rather than re-signing the frozen
contract as though it always contained a successful result.

Actual model continuations and independent scores still do not exist for this
publication comparison. The original DXI-03 dependency on actual supported Mac
verification also remains. Material/code progress continues under existing
authorization; neither these negative overlap findings nor authored rating
controls can manufacture18/18 completion or advance the learner beyondDay9.
