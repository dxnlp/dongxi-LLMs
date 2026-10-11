# GitHub-compatible book mathematics

The canonical book is read directly on GitHub as well as in local previews.
A formula rendering in a notebook is not sufficient evidence that its Markdown
source renders on GitHub.

## Authoring rules

- Write inline math with dollar/backtick delimiters: `$` + backtick + TeX +
  backtick + `$`, for example ``$`p_i-q_i`$``. Use this protected form
  consistently, including formulas in headings, lists and table cells.
  Ordinary `$...$` can lose escaped punctuation or remain literal when next
  to a hyphen, as in `top-$k$`.
- Inside a Markdown table, spell a single bar as `\vert` and a double bar as
  `\Vert` instead of using a literal pipe, even in protected inline math.
  GFM consumes table pipe escapes before the formula reaches the math client.
  For example, write `D_{KL}(q\Vert p)` with a space after the control word.
- Write display math in a fenced `math` block, with blank lines outside the
  block. Do not add `$$` inside the fence. This keeps Markdown from treating a
  standalone `=` as a heading underline, `*` as emphasis or `\\` as a Markdown
  escape before MathJax receives the formula.
- In display math, spell a less-than sign as `\lt`, with a separating space
  before a following letter, for example `y_{\lt t}`. GitHub's math client
  reparses its text through HTML, so a literal `<t` can be read as a tag.
  Protected inline formulas can retain their original less-than signs.
- End a physical display line's TeX row separator with a percent comment:
  `\\%`. This prevents GitHub's Markdown serialization from adding a third
  backslash and changing the next row's spacing. The comment ends at the
  newline. Preserve the mathematical content; these two protective spellings
  produce the same MathML as the original TeX.
- Do not use `\(...\)` or `\[...\]` as Markdown math delimiters in the book.
- Avoid `\operatorname` and its starred form: the learner encountered GitHub's
  forbidden-macro error. Use upright names such as `\mathrm{softmax}` and
  `\mathrm{RMSNorm}`. For an argmax with its index underneath, use
  `\underset{i}{\mathrm{argmax}}`. Preserve subscripts and mathematical meaning.
- Pair math delimiters, braces, and `begin`/`end` environments. A backtick cannot
  close an expression that began with a dollar sign.
- Keep code examples literal. Do not rewrite shell variables, Python strings,
  or notebook code as mathematical notation.

These are conservative project conventions, not a claim that all versions of
GitHub reject the same TeX commands. Sources:
[GitHub's math formatting documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)
and the [reported operator-name rendering issue](https://github.com/github/markup/issues/1688).

## Checks

The dependency-free source check covers every Markdown file under `book/`,
including all15chapters, worked solutions, lab guides, front matter and appendices:

```bash
python3 scripts/check_book_math.py
python3 -m unittest discover -s tests -p test_book_math.py -v
```

It excludes literal code examples, requires protected inline/display forms,
extracts their TeX, and checks known macro/delimiter hazards, brace balance and
environment pairing. Regression cases cover standalone equality lines,
matrix row separators, display HTML reparsing, table pipe escapes, set/norm escapes,
hyphen-adjacent math and multiple
protected formulas separated by prose. It is
not a complete Markdown or TeX parser, and does not claim to model every table,
HTML, indentation, currency, or escape-sequence edge case.

An optional independent rendering check converts every extracted expression to
SVG using MathJax's base, AMS and boldsymbol packages. The boldsymbol extension
checks the book's existing indicator notation without rewriting its TeX;
[MathJax documents its explicit loading](https://docs.mathjax.org/en/v3.2/input/tex/extensions/boldsymbol.html).
With Node/npm available, install its
historical pinned renderer only into a temporary tooling directory:

```bash
math_check_dir=$(mktemp -d /tmp/book-math-render-XXXXXX)
npm install --prefix "$math_check_dir" --no-audit --no-fund --ignore-scripts mathjax-full@3.2.2
NODE_PATH="$math_check_dir/node_modules" node scripts/check_book_math_render.cjs
```

MathJax 3.2.2 is a reproducible local syntax check, not the current MathJax
release or a claim about GitHub's deployed version. Its dependency tree is
legacy tooling; do not deploy it as a service or add it to the course runtime.
The SVG conversion does not emulate GitHub's Markdown parsing, sanitizer, or
browser layout. Final live-page inspection after publication remains useful.

## Repair and verification — 2026-09-08

The book-wide repair replaced 54 operator-name macros, converted 78 display
and seven inline delimiter pairs, and fixed a dollar/backtick mismatch in a
Chapter 3 exercise. Ten existing files changed across Chapters 1–5, their
affected solutions, and the Chapter 5 lab. Equations and notebook content were
preserved; this is a presentation fix, not a new experiment or learning result.

All 13 book Markdown files passed the source check: 560 expressions, no issues.
All 560 expressions rendered to SVG in MathJax 3.2.2 with no errors. All 59
repository tests passed. A preliminary latex2mathml conversion accepted the
expressions, but inspection exposed that converter's incomplete handling of
`aligned`; the MathJax check is the rendering evidence used here instead.
No GitHub live-page rendering result is claimed before these edits are published.

## Markdown preprocessing repair — 2026-10-11

The previous source check and TeX-only renderer could both pass while GitHub
split an equation into a heading or stripped its escapes. The
[repair record](../experiments/reports/2026-10-11-github-math-repair.md) separates
actual GitHub Markdown preprocessing from local MathJax syntax conversion.
All book display formulas now use `math` fences and all inline formulas use
dollar/backtick delimiters. Twenty-four display expressions also use equivalent
less-than commands or row-end comments for the observed client decoding.
One inline table expression uses the equivalent double-bar command.
All formula MathML and order are preserved; historical rendering receipts
remain evidence for their own source revisions.
