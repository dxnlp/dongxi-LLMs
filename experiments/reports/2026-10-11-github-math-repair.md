# Repairing GitHub book mathematics

Chapter 3's masked mean loss contained its equality sign, but GitHub interpreted
the standalone `=` inside a dollar display block as a Markdown heading underline.
It split the equation and removed the equality from the displayed formula.
Markdown also stripped TeX escapes and left some hyphen-adjacent inline formulas
literal. The repair covers all fifteen chapters, affected worked solutions,
front-matter notation and the notation appendix.

All 282 display expressions now use fenced `math` blocks; all 1,589 inline
expressions use dollar/backtick delimiters. These protected forms follow
[GitHub's documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
Twenty-four displays also use equivalent `\lt` spellings or row-end comments,
and one inline table heading uses `\Vert` instead of an escaped literal pipe.
These address additional HTML reparsing, row-continuation and GFM table hazards.
Of the 1,871 original TeX bodies, 1,846 remain byte-identical; all 1,871 retain
their mathematical output and order in the exact MathML comparisons below.
Only 32 of the book's 55 Markdown files require formula changes. Prose, literal
code, notebooks and measured model results are preserved.

The [authoring guide](../../docs/MATH_FORMATTING.md) and existing source checker
now require these conventions. Protected inline spans are consumed atomically,
preventing one formula's closing backtick from pairing with another's opener.
Regression tests cover the reported equality line, matrix row separators,
HTML reparsing, table pipes, set/norm escapes, hyphen adjacency and literal code.

## Verification

The [baseline](2026-10-11-github-math-repair/baseline.json) records Git commit
`29bfee102ab8c1c877fca5b5cbcfbd50c9d22188`, original book hashes and formula bodies.
The initial [conversion receipt](2026-10-11-github-math-repair/source-conversion.json)
and independent reviews preserve the delimiter-only stage:
[Chapters 1–5](2026-10-11-github-math-repair/peer-review-01-05.json),
[Chapters 6–10](2026-10-11-github-math-repair/peer-review-depth.json),
[Chapters 11–15](2026-10-11-github-math-repair/routes-peer-review.json).
The [display safety delta](2026-10-11-github-math-repair/client-source-safety-delta.json)
records the 17 less-than substitutions and 16 row comments in 24 expressions.
The [display review](2026-10-11-github-math-repair/final-safety-review.json) and
[table review](2026-10-11-github-math-repair/final-table-safety-review.json) verify
these source changes and the additional table-safe double bar by whole-file
reversal. Intermediate hashes describe their own stage, rather than being
rewritten as final-source evidence.

The [complete source audit](2026-10-11-github-math-repair/final-source-audit.json)
checks all 55 files against the Git baseline: only the declared TeX changes are
present, and every byte outside mathematical spans is identical. The
[final checker review](2026-10-11-github-math-repair/final-checker-review.json)
also checks the observed client's macro, dimension and brace-budget gates.

The [final verification](2026-10-11-github-math-repair/final-verification.json)
binds the current book/checker/guide hashes and passes:

- 15 focused regression tests;
- 55 Markdown files and 1,871 formulas, with zero source issues;
- prose and navigation checks, retaining 15 chapters, 28 days and 83 notebooks;
- all 1,871 MathJax 3.2.2 TeX-to-SVG conversions;
- all 1,871 exact original-versus-repaired MathML comparisons, including actual
  GitHub API HTML processed through the observed client decoding.

The temporary renderer explicitly loads base, AMS and `boldsymbol`. Its first
base/AMS-only attempt rejected two existing Chapter 14 indicators. Loading the
[documented extension](https://docs.mathjax.org/en/v3.2/input/tex/extensions/boldsymbol.html)
fixed that tooling omission without changing the indicators. The
[renderer/DOM lock](2026-10-11-github-math-repair/renderer-and-dom-package-lock.json)
is isolated tooling, not a course runtime dependency or GitHub version claim.

## GitHub preprocessing and client decoding

The [original API receipt](2026-10-11-github-math-repair/github-before/receipt.json)
has only 1,822 math elements for 1,871 source expressions. Protected delimiters
restore all 1,871 elements, but are insufficient alone: the
[first client replay](2026-10-11-github-math-repair/intermediate-client-mathml.json)
matches only 1,847 original MathML outputs. Literal display less-than signs can
be parsed as HTML tags; physical row separators acquire an extra escape; the
table heading loses its double-bar escape. Fixing the displays raises the
[second replay](2026-10-11-github-math-repair/final-client-mathml.json) to 1,870.
Both failed stages remain available.

The final [GitHub API receipt](2026-10-11-github-math-repair/github-after-03/receipt.json)
contains all 1,871 math elements, with the expected count in every file. Its raw
payload comparator still flags intentional spellings and HTML entity encodings;
raw payload equality is not the semantic acceptance test. The
[final client comparison](2026-10-11-github-math-repair/final-client-mathml-03.json)
matches **all 1,871** original MathML outputs, for both repaired source and
API-derived client input. Response hashes bind the archived HTML.

The retained [API check](2026-10-11-github-math-repair/check_github_markdown.py)
and [MathML replay](2026-10-11-github-math-repair/check_mathml.cjs) reproduce these
distinct checks. The replay follows the first-party client's temporary HTML
document write/read behavior established in the
[client investigation](2026-10-11-github-math-repair/client-serialization-review.json).
It uses jsdom 27 and pinned local MathJax 3.2.2. This is an emulation of observed
client decoding with actual GitHub preprocessing, not a live browser-layout
measurement or a claim about GitHub's deployed MathJax version.

Historical editorial, foundations and CS336 receipts retain their original
source hashes. The presentation repair neither reruns model experiments nor
changes learner Day 9.
