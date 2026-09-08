# Tests

Tests verify reusable course implementations, numerical examples, tensor shapes,
and failure conditions independently of notebook execution.

Book math regression checks run with the ordinary test suite. They scan every
Markdown file under `book/` for the project's known GitHub rendering hazards.
For the standalone source check and optional MathJax rendering check, see
[math formatting](../docs/MATH_FORMATTING.md).
