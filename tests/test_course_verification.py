"""Verification must not erase learner exercises or skip complete answers."""
import unittest
import nbformat
from scripts.verify_course_notebooks import mark_unfinished_scaffolds


class ScaffoldVerificationTests(unittest.TestCase):
    def test_only_unfinished_tagged_exercises_are_skipped(self):
        cells = [
            nbformat.v4.new_code_cell("answer = ...", metadata={"tags": ["learner-exercise"]}),
            nbformat.v4.new_code_cell("answer = 4", metadata={"tags": ["learner-exercise"]}),
            nbformat.v4.new_code_cell("raise RuntimeError('reference failure')", metadata={"tags": ["solution"]}),
            nbformat.v4.new_code_cell("index = tensor[..., 0]"),
        ]
        original_sources = [cell.source for cell in cells]
        self.assertEqual(mark_unfinished_scaffolds(cells), [0])
        self.assertEqual([cell.source for cell in cells], original_sources)
        self.assertNotIn("skip-verification", cells[1].metadata["tags"])
        self.assertNotIn("skip-verification", cells[2].metadata["tags"])
        self.assertNotIn("skip-verification", cells[3].metadata.get("tags", []))

    def test_markdown_and_string_ellipsis_are_not_scaffolds(self):
        cells = [
            nbformat.v4.new_markdown_cell("Fill in ...", metadata={"tags": ["learner-exercise"]}),
            nbformat.v4.new_code_cell("message = '...'", metadata={"tags": ["learner-exercise"]}),
        ]
        self.assertEqual(mark_unfinished_scaffolds(cells), [])
