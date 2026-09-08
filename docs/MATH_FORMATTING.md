# GitHub-compatible book mathematics

The canonical book is read directly on GitHub as well as in local previews.
A formula rendering in a notebook is not sufficient evidence that its Markdown
source renders on GitHub.

## Authoring rules

- Write inline math with `$...$`. GitHub also supports `$`-backtick-delimited
  inline math when Markdown punctuation would otherwise interfere.
- Write display math between standalone `$$` lines, with blank lines outside
  the block and no blank paragraph inside it. Fenced `math` blocks are another
  GitHub-supported form, but use the existing dollar style consistently.
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
including all five chapters, worked solutions, the companion lab, and appendix:

```bash
python3 scripts/check_book_math.py
python3 -m unittest discover -s tests -p test_book_math.py -v
```

It excludes literal code examples, extracts inline/display math, and checks
known macro/delimiter hazards, brace balance, and environment pairing. It is
not a complete Markdown or TeX parser, and does not claim to model every table,
HTML, indentation, currency, or escape-sequence edge case.

An optional independent rendering check converts every extracted expression to
SVG using MathJax's base and AMS packages. With Node/npm available, install its
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
