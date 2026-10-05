"""Local HF checkpoint to frozen response records, with no acquisition path.

The deliberately transparent loop recomputes the full prefix (no KV cache).
Token positions passed to a forward are not FLOPs or provider billing units.
Imports do not load a model. Random tiny-model tests establish the adapter, not
pretrained reasoning quality, human preference, or hardware-independent speed.
"""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from dongxi_llms.reasoning_evaluation import (SCHEMA_VERSION, MAX_RESPONSE,
    freeze_contract, replay_records, validate_record)
from dongxi_llms.run_identity import (artifact_hashes, assert_compatible,
    canonical_hash, collect_run_identity, file_digest, safe_config,
    tokenizer_interface, validate_interface, TOKENIZER_PATTERNS)

ADAPTER_VERSION = "local-hf-response-v1"
GENERATION_KEYS = {"input_mode", "template", "context_window", "samples",
                   "device", "dtype", "add_special_tokens", "max_run_seconds",
                   "scoring_text", "interface"}
SOURCE_FILES = ("src/dongxi_llms/reasoning_generation.py",
                "src/dongxi_llms/reasoning_evaluation.py",
                "src/dongxi_llms/run_identity.py",
                "src/dongxi_llms/sampling_likelihood_lab.py",
                "scripts/generate_reasoning_records.py")


def validate_settings(settings, *, require_interface=True):
    """Reject unknown or silently unused settings; return a detached JSON copy."""
    value = json.loads(json.dumps(settings, allow_nan=False))
    if set(value) != {"template_id", "thinking_mode", "decoding", "stopping",
                      "max_new_tokens", "generation"}:
        raise ValueError("Generation settings must declare exactly the supported fields")
    if not isinstance(value["template_id"], str) or not value["template_id"]:
        raise ValueError("Nonempty template ID required")
    g, d, stop = value["generation"], value["decoding"], value["stopping"]
    expected = GENERATION_KEYS if require_interface else GENERATION_KEYS - {"interface"}
    if not isinstance(g, dict) or set(g) not in (expected, GENERATION_KEYS):
        raise ValueError("Unknown or missing generation setting")
    if not isinstance(d, dict) or set(d) != {"mode", "seed", "temperature", "top_k", "top_p"}:
        raise ValueError("Explicit decoding mode, seed, temperature, top-k and top-p required")
    if d["mode"] not in ("greedy", "sample") or type(d["seed"]) is not int or not 0 <= d["seed"] < 2**63:
        raise ValueError("Greedy/sample and a nonnegative 63-bit seed required")
    for key, upper in (("temperature", None), ("top_p", 1)):
        v = d[key]
        if type(v) not in (float, int) or not math.isfinite(v) or v <= 0 or (upper and v > upper):
            raise ValueError(f"Invalid {key}")
    if d["top_k"] is not None and (type(d["top_k"]) is not int or d["top_k"] < 1):
        raise ValueError("top_k is a positive integer or null")
    if d["mode"] == "greedy" and (d["temperature"] != 1 or d["top_k"] is not None or d["top_p"] != 1):
        raise ValueError("Greedy mode cannot silently ignore sampling transformations")
    if not isinstance(stop, dict) or set(stop) != {"eos_token_ids", "turn_stop_token_ids", "pad_token_id"}:
        raise ValueError("Explicit EOS, turn-stop and padding IDs required")
    for key in ("eos_token_ids", "turn_stop_token_ids"):
        ids = stop[key]
        if not isinstance(ids, list) or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Stop IDs must be unique nonnegative integers")
    if not stop["eos_token_ids"] + stop["turn_stop_token_ids"] or set(stop["eos_token_ids"]) & set(stop["turn_stop_token_ids"]):
        raise ValueError("Nonempty unambiguous stop classification required")
    if stop["pad_token_id"] is not None and (type(stop["pad_token_id"]) is not int or stop["pad_token_id"] < 0):
        raise ValueError("Padding ID is a nonnegative integer or null")
    if type(value["max_new_tokens"]) is not int or not 1 <= value["max_new_tokens"] <= MAX_RESPONSE:
        raise ValueError("Positive bounded output cap required")
    if type(g["context_window"]) is not int or not 1 <= g["context_window"] <= 131072:
        raise ValueError("Explicit bounded context window required")
    if type(g["samples"]) is not int or not 1 <= g["samples"] <= 64:
        raise ValueError("One to sixty-four declared attempts per item required")
    if g["device"] not in ("cpu", "cuda") or g["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("Supported device/dtype required; no automatic device mapping")
    if g["device"] == "cpu" and g["dtype"] != "float32":
        raise ValueError("CPU adapter uses float32; CUDA bfloat16 is a separate explicit path")
    if type(g["add_special_tokens"]) is not bool:
        raise ValueError("Explicit add_special_tokens required")
    if type(g["max_run_seconds"]) not in (int, float) or not math.isfinite(g["max_run_seconds"]) or not 0 < g["max_run_seconds"] <= 3600:
        raise ValueError("Finite positive run budget up to one hour required")
    if g["scoring_text"] != "decode_without_terminal_stop":
        raise ValueError("The grading text policy must be explicitly frozen")
    if g["input_mode"] == "raw":
        if g["template"] is not None or value["thinking_mode"] != "not-applicable":
            raise ValueError("Raw mode has no chat template or thinking-template toggle")
    elif g["input_mode"] == "chat":
        if not isinstance(g["template"], str) or not g["template"] or len(g["template"]) > 32768 or g["add_special_tokens"]:
            raise ValueError("Chat needs an explicit bounded template and no second special-token insertion")
        if value["thinking_mode"] not in ("template-default", "enabled", "disabled"):
            raise ValueError("Explicit chat thinking-template mode required")
        if value["thinking_mode"] != "template-default" and "enable_thinking" not in g["template"]:
            raise ValueError("The template does not expose the requested enable_thinking keyword")
    else:
        raise ValueError("Input mode must be raw or chat")
    return value


def observed_interface(tokenizer, settings):
    stop = settings["stopping"]
    interface = tokenizer_interface(tokenizer, template=settings["generation"]["template"],
                                    stop_ids=stop["eos_token_ids"] + stop["turn_stop_token_ids"])
    if stop["pad_token_id"] != tokenizer.pad_token_id:
        raise ValueError("Declared padding ID differs from the actual tokenizer")
    return interface


def freeze_local_contract(items, settings, tokenizer):
    """Freeze actual encoding/template semantics before loading any weights."""
    settings = validate_settings(settings, require_interface=False)
    actual = observed_interface(tokenizer, settings)
    if "interface" in settings["generation"]:
        assert_compatible(settings["generation"]["interface"], actual, compare_source=False)
    settings["generation"]["interface"] = actual
    return freeze_contract(items, settings)


def attempt_seed(base_seed, item_id, sample_index):
    if type(base_seed) is not int or type(sample_index) is not int or sample_index < 0:
        raise ValueError("Integer seed and nonnegative sample coordinate required")
    digest = canonical_hash({"base_seed": base_seed, "item_id": item_id,
                             "sample_index": sample_index, "version": ADAPTER_VERSION})
    return int(digest[:16], 16) % (2**63)


def serialize_prompt(tokenizer, item, settings):
    """Only prompt/messages enter the model; references and rubric labels do not."""
    g = settings["generation"]
    if g["input_mode"] == "raw":
        text = item["prompt"]
        ids = tokenizer.encode(text, add_special_tokens=g["add_special_tokens"])
    else:
        messages = item.get("messages", [{"role": "user", "content": item["prompt"]}])
        if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or set(m) != {"role", "content"} or
                m["role"] not in ("system", "user", "assistant") or
                not isinstance(m["content"], str) for m in messages) or messages[-1]["role"] != "user":
            raise ValueError("Chat evaluation needs explicit messages ending in a user turn")
        kwargs = {} if settings["thinking_mode"] == "template-default" else {
            "enable_thinking": settings["thinking_mode"] == "enabled"}
        text = tokenizer.apply_chat_template(messages, chat_template=g["template"],
            tokenize=False, add_generation_prompt=True, **kwargs)
        ids = tokenizer.encode(text, add_special_tokens=False)
        direct = tokenizer.apply_chat_template(messages, chat_template=g["template"],
            tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs)
        if ids != direct:
            raise ValueError("Rendered chat text and direct template tokenization disagree")
    if not ids:
        raise ValueError("An empty tokenized prompt cannot predict a next token")
    vocabulary_ids = set(tokenizer.get_vocab().values())
    if any(type(i) is not int or i not in vocabulary_ids for i in ids):
        raise ValueError("Prompt IDs outside actual tokenizer vocabulary")
    return text, ids


def choose_token(logits, decoding, generator):
    """Return chosen ID plus actual behavior/raw-model log probabilities."""
    import torch
    from dongxi_llms.sampling_likelihood_lab import behavior_distribution
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("Next-token logits must be one finite vocabulary vector")
    logits = logits.detach().to(dtype=torch.float64)
    raw_logp = logits.log_softmax(-1)
    if not bool(torch.isfinite(raw_logp).all()):
        raise ValueError("Raw log probabilities are not representable")
    if decoding["mode"] == "greedy":
        chosen = int(logits.argmax())
        return chosen, 0., float(raw_logp[chosen]), 1
    probabilities, logp, support = behavior_distribution(logits,
        decoding["temperature"], decoding["top_k"], decoding["top_p"])
    chosen = int(torch.multinomial(probabilities, 1, generator=generator))
    return chosen, float(logp[chosen]), float(raw_logp[chosen]), int(support.sum())


def _synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def generate_record(model, tokenizer, item, contract, *, checkpoint_id,
                    identity, sample_index=0, deadline=None, progress=None):
    """One attempt. Preserve partial generation on ordinary errors/interruption.

    A progress callback must persist synchronously. Interrupts are returned as a
    recorded error plus a flag so the invocation can journal and then re-raise.
    """
    import torch
    settings = contract["settings"]
    g, stop = settings["generation"], settings["stopping"]
    seed = attempt_seed(settings["decoding"]["seed"], item["id"], sample_index)
    started = time.perf_counter()
    record = {"schema_version": SCHEMA_VERSION, "adapter_version": ADAPTER_VERSION,
        "contract_id": contract["identity"], "checkpoint_id": checkpoint_id,
        "sample_id": f"sample-{sample_index}-seed-{seed}", "item_id": item["id"],
        "source_group": item["source_group"], "task": item["task"], "split": item["split"],
        "raw_response": "", "response_text": "", "token_ids": [], "prompt_tokens": None,
        "generated_tokens": 0, "stop_reason": "unknown", "truncated": False,
        "error": None, "error_stage": None, "serialized_prompt": None,
        "prompt_token_ids": None, "attempt_seed": seed, "sample_index": sample_index,
        "selected_behavior_log_probabilities": [], "selected_raw_log_probabilities": [],
        "retained_support_sizes": [], "stop_token_id": None,
        "checkpoint_interface_sha256": g["interface"]["interface_sha256"],
        "input_identity_sha256": identity["identity_sha256"],
        "input_sha256": identity["input_sha256"],
        "settings_sha256": canonical_hash(settings),
        "decoding": settings["decoding"], "thinking_mode": settings["thinking_mode"],
        "cost": {"wall_seconds": 0., "generation_tokens": 0, "scoring_tokens": 0,
                 "model_forward_tokens": 0, "attempted_forward_tokens": 0,
                 "forward_calls": 0, "attempted_forward_calls": 0,
                 "forward_seconds": 0., "prefill_seconds": 0., "decode_seconds": 0.},
        "cost_boundary": "Single-sequence full-prefix HF forwards; positions are not FLOPs/billing; no model scoring",
        "timing_boundary": "Device-synchronized wall time" if g["device"] == "cuda" else "CPU host wall time",
        "interrupted": False}
    stage = "serialize_prompt"
    try:
        text, prompt = serialize_prompt(tokenizer, item, settings)
        vocabulary_ids = set(tokenizer.get_vocab().values())
        record.update(serialized_prompt=text, prompt_token_ids=prompt, prompt_tokens=len(prompt))
        generator = torch.Generator(device=g["device"]).manual_seed(seed)
        model.eval()
        for _ in range(settings["max_new_tokens"]):
            prefix = prompt + record["token_ids"]
            if len(prefix) >= g["context_window"]:
                record.update(stop_reason="context_limit", truncated=True)
                break
            if deadline is not None and time.monotonic() >= deadline:
                stage = "deadline_boundary"
                raise TimeoutError("Declared run deadline reached at a forward boundary")
            stage = "model_forward"
            ids = torch.tensor([prefix], dtype=torch.long, device=g["device"])
            record["cost"]["attempted_forward_tokens"] += len(prefix)
            record["cost"]["attempted_forward_calls"] += 1
            _synchronize(g["device"])
            began = time.perf_counter()
            try:
                with torch.inference_mode():
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                _synchronize(g["device"])
            finally:
                elapsed = time.perf_counter() - began
                record["cost"]["forward_seconds"] += elapsed
                boundary = "prefill_seconds" if record["cost"]["attempted_forward_calls"] == 1 else "decode_seconds"
                record["cost"][boundary] += elapsed
            record["cost"]["model_forward_tokens"] += len(prefix)
            record["cost"]["forward_calls"] += 1
            stage = "choose_token"
            chosen, behavior_logp, raw_logp, support_size = choose_token(output.logits[0, -1], settings["decoding"], generator)
            record["token_ids"].append(chosen)
            record["selected_behavior_log_probabilities"].append(behavior_logp)
            record["selected_raw_log_probabilities"].append(raw_logp)
            record["retained_support_sizes"].append(support_size)
            record["generated_tokens"] = len(record["token_ids"])
            record["cost"]["generation_tokens"] = record["generated_tokens"]
            stage = "decode"
            record["raw_response"] = tokenizer.decode(record["token_ids"], skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False)
            record["response_text"] = record["raw_response"]
            if chosen not in vocabulary_ids:
                raise ValueError("The model selected an ID without a tokenizer mapping; full IDs are retained")
            if chosen in stop["eos_token_ids"] or chosen in stop["turn_stop_token_ids"]:
                reason = "eos" if chosen in stop["eos_token_ids"] else "turn_stop"
                record.update(stop_reason=reason, stop_token_id=chosen)
                record["response_text"] = tokenizer.decode(record["token_ids"][:-1],
                    skip_special_tokens=False, clean_up_tokenization_spaces=False)
            record["cost"]["wall_seconds"] = time.perf_counter() - started
            if progress is not None:
                progress(json.loads(json.dumps(record, allow_nan=False)))
            if record["stop_reason"] != "unknown":
                break
        if record["stop_reason"] == "unknown":
            record.update(stop_reason="max_tokens", truncated=True)
    except (Exception, KeyboardInterrupt) as error:
        record.update(stop_reason="error", truncated=False, error=str(error) or type(error).__name__,
                      error_stage=stage, interrupted=isinstance(error, KeyboardInterrupt))
    record["cost"]["wall_seconds"] = time.perf_counter() - started
    record["raw_response_sha256"] = hashlib.sha256(record["raw_response"].encode()).hexdigest()
    record["response_text_sha256"] = hashlib.sha256(record["response_text"].encode()).hexdigest()
    validate_record(record, contract, item)
    return record


class GenerationJournal:
    """Append-only events and response rows in an exclusively created directory."""
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.events = (self.output / "events.jsonl").open("x", encoding="utf-8")
        self.records = (self.output / "responses.jsonl").open("x", encoding="utf-8")

    @staticmethod
    def append(handle, value):
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    def event(self, stage, **fields):
        self.append(self.events, {"stage": stage, "utc": datetime.now(timezone.utc).isoformat(),
                                  **safe_config(fields)})

    def record(self, value):
        self.append(self.records, value)

    def write_new(self, name, value):
        with (self.output / name).open("x", encoding="utf-8") as handle:
            self.append(handle, value)

    def close(self):
        self.events.close()
        self.records.close()


def load_local_tokenizer(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Tokenizer must be an existing local directory, not a remote identifier")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


def run_generation(*, root, items_path, contract_path, checkpoint, output,
                   tokenizer_path=None, environment_lock=None, allow_cuda=False):
    """Run an explicitly invoked local experiment; no download/install pathway."""
    root, checkpoint = Path(root).resolve(), Path(checkpoint).resolve()
    items_path, contract_path = Path(items_path).resolve(), Path(contract_path).resolve()
    tokenizer_path = Path(tokenizer_path or checkpoint).resolve()
    journal = GenerationJournal(output)
    stage, records, identity, planned = "read_contract", [], None, []
    began = time.monotonic()
    try:
        journal.event(stage, items_path=str(items_path), contract_path=str(contract_path), checkpoint=str(checkpoint))
        items, contract = json.loads(items_path.read_text()), json.loads(contract_path.read_text())
        # Contract/suite/version proof, without generating or grading a response.
        replay_records(items, [], contract)
        settings = validate_settings(contract["settings"])
        planned = [{"item_id": item["id"], "sample_index": i} for item in items
                   for i in range(settings["generation"]["samples"])]
        journal.event("planned_coverage", attempts=planned)
        journal.write_new("contract.json", contract)
        if not checkpoint.is_dir() or (checkpoint / "adapter_config.json").exists():
            raise ValueError("Need a local full or explicitly merged HF checkpoint with its saved tokenizer")
        stage = "capture_input_identity"
        checkpoint_hashes = artifact_hashes(checkpoint)
        tokenizer_hashes = artifact_hashes(tokenizer_path, patterns=TOKENIZER_PATTERNS)
        identity = collect_run_identity(root, source_files=[root / p for p in SOURCE_FILES],
            input_files=[items_path, contract_path], checkpoint=checkpoint,
            environment_lock=environment_lock, config=settings,
            interface=settings["generation"]["interface"], device={"mode": settings["generation"]["device"]})
        identity["selected_tokenizer_files"] = tokenizer_hashes
        identity["selected_tokenizer_path"] = str(tokenizer_path)
        identity["interface_evidence"] = "Frozen declaration; completed validation writes observed-interface.json before model loading"
        identity["identity_sha256"] = canonical_hash({k: v for k, v in identity.items() if k != "identity_sha256"})
        journal.write_new("input-identity.json", identity)
        stage = "validate_tokenizer_interface"
        journal.event(stage, identity_sha256=identity["identity_sha256"])
        tokenizer = load_local_tokenizer(tokenizer_path)
        actual = observed_interface(tokenizer, settings)
        assert_compatible(settings["generation"]["interface"], actual, compare_source=False)
        saved = observed_interface(load_local_tokenizer(checkpoint), settings)
        assert_compatible(saved, actual, compare_source=False)
        genealogy = checkpoint / "course-genealogy.json"
        if genealogy.exists():
            trained = json.loads(genealogy.read_text()).get("checkpoint_interface")
            if trained is None:
                raise ValueError("Saved checkpoint genealogy lacks its tokenizer interface")
            validate_interface(trained)
            if trained["tokenizer"] != actual["tokenizer"]:
                raise ValueError("Saved training tokenizer semantics differ from evaluation tokenizer")
        journal.write_new("observed-interface.json", actual)
        if artifact_hashes(checkpoint) != checkpoint_hashes or artifact_hashes(tokenizer_path, patterns=TOKENIZER_PATTERNS) != tokenizer_hashes:
            raise ValueError("Local artifact bytes changed during interface preparation")
        stage = "hardware_preflight"
        journal.event(stage)
        import torch
        device = settings["generation"]["device"]
        if device == "cuda":
            if not allow_cuda or not torch.cuda.is_available():
                raise ValueError("CUDA requires explicit allowance and an available device")
            if not Path("/proc/meminfo").is_file():
                raise ValueError("CUDA path requires the Spark/Linux host reserve check")
            available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:")) / 1024**2
            if available < 25:
                raise ValueError("At least 25 GiB host reserve required before model loading")
        stage = "load_model"
        journal.event(stage)
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, attn_implementation="eager",
            dtype=getattr(torch, settings["generation"]["dtype"]))
        model.to(device).eval()
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("This adapter requires a decoder-only causal LM")
        capacities = [getattr(model.config, key, None) for key in ("max_position_embeddings", "n_positions")]
        capacities = [v for v in capacities if type(v) is int and v > 0]
        if not capacities or settings["generation"]["context_window"] > min(capacities):
            raise ValueError("Declared context window exceeds or lacks an actual model capacity")
        if model.get_input_embeddings().weight.shape[0] <= actual["tokenizer"]["max_token_id"]:
            raise ValueError("Model embeddings cannot address the actual token vocabulary")
        width = model.get_output_embeddings().weight.shape[0]
        if settings["decoding"]["top_k"] is not None and settings["decoding"]["top_k"] > width:
            raise ValueError("top_k exceeds the actual output vocabulary")
        if artifact_hashes(checkpoint) != checkpoint_hashes or artifact_hashes(tokenizer_path, patterns=TOKENIZER_PATTERNS) != tokenizer_hashes:
            raise ValueError("Local artifact bytes changed during model loading")
        checkpoint_id = "local-hf-sha256:" + canonical_hash({"files": checkpoint_hashes,
            "tokenizer": tokenizer_hashes, "interface": actual["interface_sha256"]})
        journal.event("model_loaded", checkpoint_id=checkpoint_id,
            parameter_count=sum(p.numel() for p in model.parameters()), device=device,
            dtype=str(next(model.parameters()).dtype), model_class=type(model).__name__)
        deadline = began + settings["generation"]["max_run_seconds"]
        for item in items:
            for sample_index in range(settings["generation"]["samples"]):
                stage = "generation_attempt"
                journal.event(stage, item_id=item["id"], sample_index=sample_index,
                    seed=attempt_seed(settings["decoding"]["seed"], item["id"], sample_index))
                record = generate_record(model, tokenizer, item, contract, checkpoint_id=checkpoint_id,
                    identity=identity, sample_index=sample_index, deadline=deadline,
                    progress=lambda partial: journal.event("partial_response", record=partial))
                journal.record(record)
                records.append(record)
                if record["interrupted"]:
                    raise KeyboardInterrupt(record["error"])
        stage = "verify_unchanged_inputs"
        journal.event(stage)
        if artifact_hashes(checkpoint) != checkpoint_hashes or artifact_hashes(tokenizer_path, patterns=TOKENIZER_PATTERNS) != tokenizer_hashes or any(
                file_digest(path) != identity["input_sha256"][str(path.relative_to(root)) if path.is_relative_to(root) else str(path)]
                for path in (items_path, contract_path)):
            raise ValueError("Input bytes changed during evaluation; records are retained but invocation is invalid")
        replay = replay_records(items, records, contract)
        journal.write_new("evaluation.json", replay)
        status = "completed_with_errors" if any(r["error"] is not None for r in records) else "completed"
        summary = {"status": status, "adapter_version": ADAPTER_VERSION,
            "checkpoint_id": checkpoint_id, "contract_id": contract["identity"],
            "identity_sha256": identity["identity_sha256"], "record_count": len(records),
            "elapsed_seconds": time.monotonic() - began, "inputs_unchanged": True,
            "limitations": ["Local generation events, not a pretrained capability or human review claim",
                "Full-prefix loop without KV caching; measured times are not optimized inference benchmarks",
                "Deadline checked at forward boundaries, not an external hard-kill supervisor",
                "New invocation only; saved partial events are not an exact-resume guarantee"]}
        journal.write_new("summary.json", summary)
        journal.event(status, summary=summary)
        return summary
    except (Exception, KeyboardInterrupt) as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        journal.event(status, failure={"stage": stage, "type": type(error).__name__, "message": str(error)},
                      persisted_final_records=len(records), identity_sha256=identity.get("identity_sha256") if identity else None)
        journal.write_new("failure.json", {"status": status, "stage": stage,
            "error_type": type(error).__name__, "error": safe_config(str(error)),
            "persisted_final_records": len(records), "elapsed_seconds": time.monotonic() - began,
            "planned_attempts": planned,
            "not_attempted": [p for p in planned if (p["item_id"], p["sample_index"]) not in
                              {(r["item_id"], r["sample_index"]) for r in records}]})
        raise
    finally:
        journal.close()
