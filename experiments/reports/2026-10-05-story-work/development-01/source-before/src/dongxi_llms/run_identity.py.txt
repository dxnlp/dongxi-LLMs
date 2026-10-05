"""Shared evidence identities for optional model runs; no model work on import.

An upstream revision is a declaration. Local file digests identify input bytes.
Tokenizer semantics and the inference interface are separate from volatile run
metadata. This module never reads credential environment variables.
"""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from uuid import uuid4

SCHEMA_VERSION = 1
SPECIAL_IDS = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
               "sep_token_id", "cls_token_id", "mask_token_id",
               "additional_special_tokens_ids")
ARTIFACT_PATTERNS = ("config.json", "generation_config.json", "adapter_config.json",
                     "course-genealogy.json", "tokenizer*",
                     "special_tokens_map.json", "chat_template*",
                     "vocab*", "merges*", "*.model", "*.tiktoken",
                     "*.safetensors", "*.safetensors.index.json",
                     "pytorch_model*.bin", "pytorch_model*.bin.index.json")
SECRET_KEYS = frozenset(("token", "access_token", "hf_token", "api_key",
                         "authorization", "password", "secret"))
TOKENIZER_PATTERNS = ("tokenizer*", "special_tokens_map.json", "chat_template*",
                      "vocab*", "merges*", "*.model", "*.tiktoken")


def plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def canonical_hash(value):
    data = json.dumps(plain(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_digest(path, before_chunk=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            if before_chunk is not None:
                before_chunk()
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(directory, before_chunk=None, *, patterns=ARTIFACT_PATTERNS):
    """HF artifact bytes, not hidden caches, credentials or arbitrary files."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Artifact directory must exist")
    files = {p for pattern in patterns
             for p in directory.glob(pattern) if p.is_file()}
    if not files:
        raise ValueError("No recognized local checkpoint/tokenizer artifacts")
    return {str(p.relative_to(directory)): file_digest(p, before_chunk)
            for p in sorted(files)}


def _check_revision(revision):
    if revision is not None and (not isinstance(revision, str) or
                                re.fullmatch(r"[0-9a-f]{40}", revision) is None):
        raise ValueError("Tokenizer revision must be an immutable 40-character SHA")


def tokenizer_interface(tokenizer, *, template, stop_ids, tokenizer_id=None,
                        tokenizer_revision=None):
    """Hash token meanings AND encoding rules; exclude volatile backend buffers.

    Fast-tokenizer JSON includes normalizer, pre-tokenizer, model/merges,
    decoder, post-processor and added-token properties. Runtime padding and
    truncation are deliberately excluded. Slow tokenizers require a future
    explicitly verified serialization contract; vocabulary-only is rejected.
    """
    _check_revision(tokenizer_revision)
    vocabulary = tokenizer.get_vocab()
    if not vocabulary or any(not isinstance(k, str) or type(v) is not int or v < 0
                             for k, v in vocabulary.items()):
        raise ValueError("Need nonempty string→nonnegative integer vocabulary")
    if len(set(vocabulary.values())) != len(vocabulary):
        raise ValueError("Vocabulary token IDs must be unique")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not callable(getattr(backend, "to_str", None)):
        raise ValueError("Full tokenizer encoding serialization is required; vocab-only is unsafe")
    encoding = json.loads(backend.to_str())
    if not isinstance(encoding, dict) or "model" not in encoding:
        raise ValueError("Tokenizer backend has no serialized model")
    encoding = {k: v for k, v in encoding.items() if k not in ("padding", "truncation")}
    special = {}
    for name in SPECIAL_IDS:
        value = getattr(tokenizer, name, None)
        values = value if isinstance(value, (list, tuple)) else [value]
        if any(v is not None and (type(v) is not int or v not in vocabulary.values()) for v in values):
            raise ValueError(f"Invalid vocabulary binding for {name}")
        special[name] = plain(value)
    stops = list(stop_ids)
    if not stops or any(type(v) is not int or v not in vocabulary.values() for v in stops):
        raise ValueError("Declared stop IDs must exist in the actual vocabulary")
    if template is not None and (not isinstance(template, str) or not template):
        raise ValueError("Template must be nonempty text or explicitly None for raw mode")
    semantics = {"vocab_sha256": canonical_hash(sorted(vocabulary.items())),
                 "encoding_sha256": canonical_hash(encoding),
                 "special_ids": special, "vocab_size": len(vocabulary),
                 "max_token_id": max(vocabulary.values()),
                 "wrapper_settings": {key: plain(getattr(tokenizer, key, None)) for key in
                    ("add_bos_token", "add_eos_token", "split_special_tokens",
                     "clean_up_tokenization_spaces", "spaces_between_special_tokens")}}
    result = {"schema_version": SCHEMA_VERSION,
              "tokenizer": semantics,
              "template_sha256": hashlib.sha256(template.encode()).hexdigest()
                  if template is not None else None,
              "generation_stop_ids": sorted(set(stops)),
              "source": {"tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision},
              "implementation_class": type(tokenizer).__name__}
    result["interface_sha256"] = _interface_digest(result)
    return result


def _interface_digest(value):
    return canonical_hash({k: value[k] for k in
                           ("schema_version", "tokenizer", "template_sha256",
                            "generation_stop_ids")})


def validate_interface(value):
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unknown or missing checkpoint interface schema")
    try:
        digest = _interface_digest(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Incomplete checkpoint interface") from error
    if digest != value.get("interface_sha256"):
        raise ValueError("Checkpoint interface fingerprint fields were changed")
    semantics = value.get("tokenizer")
    source = value.get("source")
    if not isinstance(source, dict) or not isinstance(semantics, dict):
        raise ValueError("Malformed tokenizer semantics or source declaration")
    for key in ("vocab_sha256", "encoding_sha256"):
        if not isinstance(semantics.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", semantics[key]):
            raise ValueError("Malformed tokenizer semantic digest")
    if not isinstance(semantics.get("special_ids"), dict) or set(semantics["special_ids"]) != set(SPECIAL_IDS):
        raise ValueError("Malformed special-token bindings")
    if type(semantics.get("vocab_size")) is not int or semantics["vocab_size"] < 1 or \
            type(semantics.get("max_token_id")) is not int or semantics["max_token_id"] < 0:
        raise ValueError("Malformed vocabulary bounds")
    stops = value.get("generation_stop_ids")
    if not isinstance(stops, list) or not stops or any(type(x) is not int or x < 0 for x in stops) or \
            stops != sorted(set(stops)):
        raise ValueError("Malformed generation stop IDs")
    template_hash = value.get("template_sha256")
    if template_hash is not None and (not isinstance(template_hash, str) or
                                      not re.fullmatch(r"[0-9a-f]{64}", template_hash)):
        raise ValueError("Malformed template digest")
    if source.get("tokenizer_id") is not None and not isinstance(source["tokenizer_id"], str):
        raise ValueError("Malformed tokenizer source identifier")
    _check_revision(source.get("tokenizer_revision"))
    return value


def assert_compatible(expected, observed, *, compare_source=True):
    validate_interface(expected)
    validate_interface(observed)
    for key in ("tokenizer", "template_sha256", "generation_stop_ids"):
        if expected[key] != observed[key]:
            raise ValueError(f"Parent checkpoint interface mismatch: {key}")
    if compare_source:
        source = expected.get("source", {})
        actual = observed.get("source", {})
        for key in ("tokenizer_id", "tokenizer_revision"):
            if source.get(key) is not None and source[key] != actual.get(key):
                raise ValueError(f"Parent tokenizer declaration mismatch: {key}")
    return True


def parent_interface(checkpoint, saved_tokenizer, *, template, stop_ids,
                     tokenizer_id=None, tokenizer_revision=None, allow_legacy=False):
    """Verify genealogy against actual saved tokenizer; explicit legacy adoption.

    Legacy adoption does not prove old upstream revision ancestry. It establishes
    actual saved semantics and records the new declaration for downstream use.
    """
    checkpoint = Path(checkpoint)
    path = checkpoint / "course-genealogy.json"
    genealogy = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(genealogy, dict):
        raise ValueError("Checkpoint genealogy must be an object")
    expected = genealogy.get("checkpoint_interface")
    if expected is None and not allow_legacy:
        raise ValueError("Legacy checkpoint has no interface fingerprint; explicitly migrate/adopt after audit")
    if expected is not None:
        validate_interface(expected)
    source = expected.get("source", {}) if expected is not None else {
        "tokenizer_id": tokenizer_id, "tokenizer_revision": tokenizer_revision}
    observed = tokenizer_interface(saved_tokenizer, template=template, stop_ids=stop_ids,
        tokenizer_id=source.get("tokenizer_id"), tokenizer_revision=source.get("tokenizer_revision"))
    if expected is not None:
        assert_compatible(expected, observed)
    return observed, {"legacy_adoption": expected is None,
                      "revision_evidence": "declared metadata; actual saved semantics separately verified",
                      "parent_genealogy": genealogy}


def safe_config(value):
    """Redact named credential fields, but not token IDs/tokenizer settings."""
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower().replace("-", "_") in SECRET_KEYS
                else safe_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_config(v) for v in value]
    if isinstance(value, str):
        return re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", value)
    return plain(value)


def cached_snapshot(repository, revision, filename="config.json"):
    """Resolve only an already-cached pinned HF snapshot; never download."""
    _check_revision(revision)
    if revision is None:
        raise ValueError("Cached input identity requires an immutable revision")
    from huggingface_hub import try_to_load_from_cache
    value = try_to_load_from_cache(repository, filename, revision=revision)
    if not isinstance(value, str) or not Path(value).is_file():
        raise ValueError("Loaded model's pinned local snapshot cannot be identified")
    snapshot = Path(value).parent
    if snapshot.name != revision:
        raise ValueError("Local snapshot path does not match the declared immutable revision")
    return snapshot


def prepare_model_snapshot(repository, revision, *, allow_download=False):
    """Explicit acquisition boundary; returned inputs are loaded locally only."""
    if not allow_download:
        return cached_snapshot(repository, revision)
    _check_revision(revision)
    if revision is None:
        raise ValueError("Model acquisition requires an immutable revision")
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(repository, revision=revision,
                                      allow_patterns=list(ARTIFACT_PATTERNS)))
    if snapshot.name != revision:
        raise ValueError("Acquired snapshot does not match the declared immutable revision")
    return snapshot


def safe_command(command):
    result, redact_next = [], False
    for argument in map(str, command):
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
            continue
        name = argument.split("=", 1)[0].lstrip("-").replace("-", "_").lower()
        if argument.startswith("--") and name in SECRET_KEYS:
            if "=" in argument:
                result.append(argument.split("=", 1)[0] + "=[REDACTED]")
            else:
                result.append(argument)
                redact_next = True
        else:
            result.append(re.sub(r"hf_[A-Za-z0-9]{20,}", "[REDACTED]", argument))
    return result


def _command_output(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True,
                                timeout=5, check=True)
        return {"status": "measured", "value": result.stdout.strip()}
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "value": None}


def environment_identity(environment_lock=None):
    names = ("torch", "transformers", "tokenizers", "peft", "numpy",
             "matplotlib", "nbformat", "nbclient", "huggingface-hub")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    lock = {"status": "not_supplied", "path": None, "sha256": None}
    if environment_lock is not None:
        path = Path(environment_lock)
        if not path.is_file():
            raise ValueError("Selected environment lock must be an existing file")
        lock = {"status": "hashed", "path": str(path.resolve()), "sha256": file_digest(path),
                "boundary": "byte identity, not proof all installed versions resolved from this lock"}
    return {"python_version": platform.python_version(), "interpreter": sys.executable,
            "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions, "environment_lock": lock}


def collect_run_identity(root, *, source_files, input_files=(), checkpoint=None,
                         environment_lock=None, interface=None, config=None,
                         device=None, before_chunk=None, command=None):
    """Capture reproducibility inputs without importing/loading model code."""
    root = Path(root).resolve()
    def paths(values):
        result = {}
        for value in values:
            path = Path(value).resolve()
            try:
                key = str(path.relative_to(root))
            except ValueError:
                key = str(path)
            result[key] = file_digest(path, before_chunk)
        return result
    status = _command_output(["git", "status", "--porcelain", "-z"], root)
    device = safe_config(device or {"name": "CPU"})
    driver = _command_output(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"]) if device.get("mode") == "cuda" else {
                                  "status": "not_requested", "value": None}
    # Under python -m, sys.argv[0] is rewritten to a module file; replaying
    # that file may break relative imports. Original interpreter argv retains
    # -m/-c/options. Keep the application argv separate and label the fallback.
    original_argv = getattr(sys, "orig_argv", None)
    invocation = command if command is not None else (original_argv or [sys.executable, *sys.argv])
    origin = ("explicit-caller-invocation" if command is not None else
              "original-interpreter-argv" if original_argv else "application-argv-fallback-not-original")
    result = {"schema_version": SCHEMA_VERSION,
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "git": {"commit": _command_output(["git", "rev-parse", "HEAD"], root),
                      "dirty": bool(status["value"]) if status["status"] == "measured" else None,
                      "status_sha256": canonical_hash(status["value"]),
                      "changed_paths": [item for item in (status["value"] or "").split("\0") if item]},
              "source_sha256": paths(source_files),
              "input_sha256": paths(input_files),
              "checkpoint_files": artifact_hashes(checkpoint, before_chunk) if checkpoint else {},
              "checkpoint_path": str(Path(checkpoint).resolve()) if checkpoint else None,
              "environment": environment_identity(environment_lock),
              "device": device, "gpu_driver": driver,
              "command": safe_command(invocation), "command_origin": origin,
              "python_argv": safe_command(list(sys.argv)),
              "config": safe_config(config or {}), "checkpoint_interface": interface,
              "revision_evidence": "declared metadata and actual local-file hashes are distinct"}
    if interface is not None:
        validate_interface(interface)
    result["identity_sha256"] = canonical_hash(result)
    return result


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class IdentityJournal:
    """Per-invocation identity, stages and failures; not a recovery implementation."""
    def __init__(self, output, config, *, invocation_id=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.invocation_id = invocation_id or uuid4().hex
        self.path = self.output / f"identity-{self.invocation_id}.json"
        if self.path.exists():
            raise FileExistsError("Cannot overwrite an existing invocation identity")
        self.started = time.monotonic()
        self.data = {"schema_version": SCHEMA_VERSION, "invocation_id": self.invocation_id,
                     "status": "running", "stage": "created", "config": safe_config(config),
                     "stages": [], "identity": None}
        self.flush()

    def flush(self):
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        write_json(self.path, self.data)

    def stage(self, name, **fields):
        self.data["stage"] = name
        self.data.update(safe_config(fields))
        self.data["stages"].append({"stage": name, "utc": datetime.now(timezone.utc).isoformat()})
        self.flush()

    def attach(self, identity):
        self.data["identity"] = plain(identity)
        self.flush()

    def fail(self, error):
        self.data["status"] = "failed"
        self.data["failure"] = {"stage": self.data["stage"], "type": type(error).__name__,
                                "message": safe_config(str(error))}
        self.flush()

    def complete(self, **measurements):
        self.data.update(status="completed", measurements=plain(measurements))
        self.stage("completed")
