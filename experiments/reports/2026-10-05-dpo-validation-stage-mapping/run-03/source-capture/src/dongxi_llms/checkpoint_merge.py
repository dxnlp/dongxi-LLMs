"""Explicit local PEFT-to-full-HF export, not a download or training launcher."""
import json
from pathlib import Path
import time

from .run_identity import (IdentityJournal, artifact_hashes, assert_compatible, canonical_hash,
                           collect_run_identity, parent_interface, tokenizer_interface,
                           write_json)


def merge_local_adapter(base, adapter, output, *, base_revision, environment_lock,
                        allow_legacy=False, before_chunk=None):
    """CPU export with actual parent/interface checks and per-invocation evidence.

    The caller owns sizing and resource authorization. A saved declaration does
    not authenticate upstream ancestry; recorded base hashes, when available,
    must agree with the actual local base. Output must be new, even after failure.
    """
    base, adapter, output = map(Path, (base, adapter, output))
    if output.exists():
        raise FileExistsError("Merge output must be a new directory")
    if not base.is_dir() or not adapter.is_dir():
        raise ValueError("Merge requires existing local base and adapter directories")
    journal = IdentityJournal(output, {"base": base, "adapter": adapter,
                                      "base_revision": base_revision,
                                      "environment_lock": environment_lock,
                                      "allow_legacy_interface": allow_legacy})
    start = time.monotonic()
    try:
        root = Path(__file__).resolve().parents[2]
        journal.attach(collect_run_identity(root, source_files=[__file__,
            root / "src/dongxi_llms/run_identity.py"], environment_lock=environment_lock,
            config=journal.data["config"], device={"mode": "cpu", "name": "preflight"}))
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        import torch
        journal.stage("checking_parent")
        genealogy_path = adapter / "course-genealogy.json"
        if not genealogy_path.is_file():
            raise ValueError("Adapter requires course genealogy with a pinned base revision")
        genealogy = json.loads(genealogy_path.read_text())
        if not isinstance(base_revision, str) or len(base_revision) != 40 or \
                any(c not in "0123456789abcdef" for c in base_revision) or \
                genealogy.get("base_revision") != base_revision:
            raise ValueError("Requested base revision differs from adapter genealogy")
        tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        expected = genealogy.get("checkpoint_interface")
        stops = expected.get("generation_stop_ids") if isinstance(expected, dict) else None
        if stops is None:
            stops = [tokenizer.eos_token_id]
        interface, adoption = parent_interface(adapter, tokenizer,
            template=tokenizer.chat_template, stop_ids=stops, allow_legacy=allow_legacy)
        base_hashes = artifact_hashes(base, before_chunk)
        recorded = genealogy.get("base_checkpoint_files")
        if recorded is not None and recorded != base_hashes:
            raise ValueError("Actual base artifact bytes differ from adapter's recorded parent")
        base_tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        source = interface["source"]
        observed = tokenizer_interface(base_tokenizer, template=tokenizer.chat_template,
            stop_ids=stops, tokenizer_id=source.get("tokenizer_id"),
            tokenizer_revision=source.get("tokenizer_revision"))
        assert_compatible(interface, observed)
        identity = collect_run_identity(root, source_files=[__file__,
            root / "src/dongxi_llms/run_identity.py"], input_files=[genealogy_path],
            checkpoint=base, environment_lock=environment_lock, interface=interface,
            config=journal.data["config"], device={"mode": "cpu", "name": "CPU"},
            before_chunk=before_chunk)
        identity["adapter_files"] = artifact_hashes(adapter, before_chunk)
        identity["identity_sha256"] = canonical_hash({k: v for k, v in identity.items()
                                                    if k != "identity_sha256"})
        journal.attach(identity)
        journal.stage("loading_local_base")
        if before_chunk:
            before_chunk()
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True,
                                                    dtype=torch.float32)
        if model.get_input_embeddings().weight.shape[0] <= interface["tokenizer"]["max_token_id"]:
            raise ValueError("Model embedding rows cannot represent the tokenizer IDs")
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        if adapter_config.get("revision") not in (None, base_revision):
            raise ValueError("PEFT adapter revision disagrees with course genealogy")
        journal.stage("merging_adapter")
        network = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        merged = network.merge_and_unload(safe_merge=True)
        if before_chunk:
            before_chunk()
        journal.stage("exporting_full_checkpoint")
        merged.save_pretrained(output / "policy", safe_serialization=True)
        tokenizer.save_pretrained(output / "policy")
        exported_tokenizer = AutoTokenizer.from_pretrained(output / "policy", local_files_only=True)
        exported = tokenizer_interface(exported_tokenizer, template=exported_tokenizer.chat_template,
            stop_ids=stops, tokenizer_id=source.get("tokenizer_id"),
            tokenizer_revision=source.get("tokenizer_revision"))
        assert_compatible(interface, exported)
        write_json(output / "policy/course-genealogy.json", {
            "kind": "full-HF-model", "method": "explicit-local-PEFT-merge",
            "base_revision": base_revision, "base_model": genealogy.get("base_model"),
            "parent_adapter": str(adapter.resolve()), "base_checkpoint_files": base_hashes,
            "adapter_files": identity["adapter_files"], "checkpoint_interface": exported,
            "template_sha256": exported["template_sha256"], "legacy_adoption": adoption,
            "upstream_revision_evidence": "declaration, not upstream authentication",
            "merge_precision": "CPU FP32", "exact_resume": False})
        journal.complete(elapsed_seconds=time.monotonic() - start,
                         exported_files=artifact_hashes(output / "policy", before_chunk))
        return output / "policy"
    except BaseException as error:
        journal.fail(error)
        raise
