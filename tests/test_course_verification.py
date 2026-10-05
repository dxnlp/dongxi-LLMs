"""Verification must not erase learner exercises or skip complete answers."""
import unittest
from pathlib import Path
import tempfile
import nbformat
from scripts.verify_course_notebooks import (KERNEL_IDENTITY_PREFIX, kernel_identity,
                                             mark_unfinished_scaffolds, prepare_output,
                                             check_kernel_prefix)


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

    def test_identity_is_from_execution_preamble_only(self):
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('pass')])
        notebook.cells[0].outputs = [nbformat.v4.new_output('stream', name='stdout',
            text=KERNEL_IDENTITY_PREFIX + '{"executable":"/tmp/isolated/python"}\n')]
        self.assertEqual(kernel_identity(notebook)['executable'], '/tmp/isolated/python')
        notebook.cells[0].outputs = []
        with self.assertRaisesRegex(RuntimeError, 'identity'):
            kernel_identity(notebook)

    def test_complete_cells_cannot_carry_a_skip_tag(self):
        cells = [nbformat.v4.new_code_cell('answer = 4', metadata={'tags':['skip-verification','solution']}),
                 nbformat.v4.new_code_cell('value = tensor[...,0]',
                                          metadata={'tags':['learner-exercise','skip-verification']})]
        self.assertEqual(mark_unfinished_scaffolds(cells), [])
        for cell in cells:
            self.assertNotIn('skip-verification', cell.metadata['tags'])

    def test_old_outputs_without_a_manifest_are_preserved(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-output-test-') as directory:
            output = Path(directory)
            old = output/'learner.ipynb'
            old.write_text('original learner evidence')
            with self.assertRaises(FileExistsError):
                prepare_output(output)
            self.assertEqual(old.read_text(), 'original learner evidence')
            self.assertEqual(prepare_output(output/'new'), output/'new')

    def test_shared_python_binary_is_not_environment_equivalence(self):
        with tempfile.TemporaryDirectory(prefix='dongxi-prefix-test-') as directory:
            first, second = Path(directory)/'venv1', Path(directory)/'venv2'
            first.mkdir(); second.mkdir()
            check_kernel_prefix({'prefix':str(first)}, first)
            with self.assertRaisesRegex(RuntimeError, 'isolated environment'):
                check_kernel_prefix({'prefix':str(second)}, first)
