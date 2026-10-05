"""File-only checkpoint chaining tests; no Transformers imports/model loading."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("course_dpo_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_chapter11_spark_dpo.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class SavedTemplateTests(unittest.TestCase):
    def test_modern_hf_jinja_storage_and_genealogy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            template = "{% for message in messages %}{{ message.content }}{% endfor %}\n"
            (path / "chat_template.jinja").write_text(template)
            (path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "Qwen2Tokenizer"}))
            (path / "course-genealogy.json").write_text(json.dumps({
                "kind": "full-HF-model", "template_sha256": hashlib.sha256(template.encode()).hexdigest()}))
            self.assertEqual(runner.restore_saved_template(path), template)

    def test_legacy_json_template(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "legacy template"}))
            self.assertEqual(runner.restore_saved_template(path), "legacy template")

    def test_missing_or_changed_template_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                runner.restore_saved_template(path)
            (path / "chat_template.jinja").write_text("changed template")
            (path / "course-genealogy.json").write_text(json.dumps({"template_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "genealogy"):
                runner.restore_saved_template(path)

    def test_actual_saved_tokenizer_and_downstream_mapping_revision(self):
        from transformers import AutoTokenizer
        from test_run_identity import interface, local_tokenizer, REVISION, TEMPLATE
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = local_tokenizer()
            original.save_pretrained(path)
            (path / "course-genealogy.json").write_text(json.dumps({
                "template_sha256": interface(original)["template_sha256"],
                "checkpoint_interface": interface(original)}))
            saved = AutoTokenizer.from_pretrained(path, local_files_only=True)
            observed, adoption = runner.verify_parent_tokenizer(path, saved, local_tokenizer(),
                tokenizer_id="original-local-fixture", tokenizer_revision=REVISION)
            self.assertFalse(adoption["legacy_adoption"])
            self.assertEqual(observed, interface(original))
            for changed, revision in ((local_tokenizer(permuted=True), REVISION),
                                      (local_tokenizer(lowercase=True), REVISION),
                                      (local_tokenizer(), "2" * 40)):
                with self.assertRaises(ValueError):
                    runner.verify_parent_tokenizer(path, saved, changed,
                        tokenizer_id="original-local-fixture", tokenizer_revision=revision)


if __name__ == "__main__":
    unittest.main()
