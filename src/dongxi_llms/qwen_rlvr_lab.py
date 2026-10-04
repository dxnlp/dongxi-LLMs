"""Optional synchronous HF causal-model RLVR adapter, offline local weights only.

CPU mechanism tests validate update_model with a tiny Torch model. No Qwen
execution, CUDA profiling, framework integration or language capability is implied.
Collection uses temperature1/full support; response-only token masks include EOS.
"""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import torch

from dongxi_llms.grpo_lab import (clipped_objective, exact_kl,
                                  group_advantages, verify_integer)


def model_logits(model, ids):
    value = model(input_ids=ids, use_cache=False)
    return value.logits.float()


def available_gib():
    text = Path("/proc/meminfo").read_text()
    match = re.search(r"^MemAvailable:\s+(\d+) kB$", text, re.MULTILINE)
    if not match:
        raise RuntimeError("Cannot inspect MemAvailable on this platform")
    return int(match.group(1))*1024/2**30


def guard_memory(reserve=25.):
    measured = available_gib()
    if measured < reserve:
        raise RuntimeError(f"MemAvailable {measured:.2f} GiB below {reserve} GiB reserve")
    return measured


def update_model(model, reference, prompt_ids, expected, decode, eos_id,
                 optimizer, generator, group_size=4, max_new_tokens=32,
                 beta=.02, before_stage=None, context_length=None, stop_ids=None):
    """One prompt → G detached rollouts → strict text rewards → one real update.

    model/ref expose HF-style logits [B,T,V]; prompt [1,P]. Callable decode maps
    response IDs to text. No tokenizer or HF package is required by this function.
    before_stage supplies an optional resource guard during generation/backward.
    """
    if group_size < 2 or max_new_tokens < 1 or prompt_ids.shape[0] != 1:
        raise ValueError("Need one prompt, G>=2 and positive generation cap")
    if eos_id is None or not isinstance(eos_id, int) or eos_id < 0:
        raise ValueError("A valid explicit EOS token ID is required")
    if context_length is not None and prompt_ids.shape[1]+max_new_tokens > context_length:
        raise ValueError("Prompt plus response cap exceeds model context length")
    stops_allowed = tuple(sorted(set(stop_ids or (eos_id,))))
    if any(not isinstance(token, int) or token < 0 for token in stops_allowed):
        raise ValueError("Stop IDs must be explicit nonnegative integers")
    check = before_stage or (lambda: None)
    model.eval()  # Disable dropout for behavior/current ratio alignment.
    reference.eval()
    prompts = prompt_ids.repeat(group_size, 1)
    current = prompts.clone()
    finished = torch.zeros(group_size, dtype=torch.bool, device=prompts.device)
    tokens, masks = [], []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            check()
            probabilities = model_logits(model, current)[:, -1].softmax(-1)
            next_token = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
            next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
            tokens.append(next_token)
            masks.append(~finished)
            finished |= torch.isin(next_token, next_token.new_tensor(stops_allowed))
            current = torch.cat((current, next_token[:, None]), -1)
            if bool(finished.all()):
                break
    responses, mask = torch.stack(tokens, -1), torch.stack(masks, -1)
    start = prompts.shape[1]-1
    inputs = torch.cat((prompts, responses), -1)[:, :-1]
    with torch.no_grad():
        old_logits = model_logits(model, inputs)[:, start:]
        old_logp = old_logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
        ref_logits = model_logits(reference, inputs)[:, start:]
    texts, rewards, stops = [], [], []
    for row, valid in zip(responses, mask):
        active = row[valid].tolist()
        stop_positions = [i for i, token in enumerate(active) if token in stops_allowed]
        terminated = bool(stop_positions)
        text = decode(active[:stop_positions[0]] if terminated else active)
        texts.append(text)
        # A length-capped response is explicitly not accepted as a complete answer.
        rewards.append(float(terminated and verify_integer(text, expected)))
        stops.append("eos" if terminated else "token-limit")
    reward = torch.tensor(rewards, device=prompts.device)
    advantage = group_advantages(reward[None]).flatten()
    check()
    logits = model_logits(model, inputs)[:, start:]
    logp = logits.log_softmax(-1).gather(-1, responses[..., None]).squeeze(-1)
    ratio_error = float((logp.detach()-old_logp).abs()[mask].max())
    kl = exact_kl(logits, ref_logits)
    loss = clipped_objective(logp, old_logp, advantage, mask, kl=kl, beta=beta)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
    if not torch.isfinite(loss) or not torch.isfinite(torch.tensor(gradient)):
        raise RuntimeError("Nonfinite loss or gradient")
    check()
    optimizer.step()
    return {"loss": float(loss.detach()), "reward": sum(rewards)/group_size,
            "gradient_norm": gradient, "initial_log_ratio_max_abs": ratio_error,
            "exact_kl_nats_before_update": float((kl.detach()*mask).sum()/mask.sum()),
            "valid_response_tokens": int(mask.sum()), "texts": texts,
            "responses": responses.tolist(), "rewards": rewards, "stops": stops,
            "advantages": advantage.tolist()}


def prompt_contract(tokenizer, mode, genealogy=None, template=None):
    """Freeze raw/base or chat/SFT formatting and stopping identity, reject mismatch."""
    if mode not in ("raw", "chat") or tokenizer.eos_token_id is None:
        raise ValueError("Need a declared prompt mode and EOS identity")
    if template is not None:
        tokenizer.chat_template = template
    if mode == "raw" and genealogy is not None:
        raise ValueError("An SFT genealogy requires its saved chat template; raw prompting is a mismatch")
    stops = [tokenizer.eos_token_id]
    template_hash = None
    if mode == "chat":
        if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
            raise ValueError("Chat mode requires one explicit saved or supplied template")
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        if genealogy is not None and genealogy.get("template_sha256") != template_hash:
            raise ValueError("Template differs from the SFT checkpoint genealogy")
        marker = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(marker) != 1 or marker[0] == tokenizer.unk_token_id:
            raise ValueError("Course chat contract needs the existing single-token im_end marker")
        stops.append(marker[0])
    def encode(prompt):
        if mode == "chat":
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False)
        return torch.tensor([ids], dtype=torch.long)
    return encode, sorted(set(stops)), {"prompt_mode": mode, "template_sha256": template_hash,
                                      "generation_stop_ids": sorted(set(stops)),
                                      "enable_thinking": False if mode == "chat" else None}


@torch.no_grad()
def evaluate_model(model, pairs, encode, decode, stop_ids, max_new_tokens,
                   context_length, device="cpu", before_stage=None, on_row=None):
    """Frozen greedy exact-integer panel; all failures/truncations retained."""
    model.eval()
    check = before_stage or (lambda: None)
    rows = []
    for a, b in pairs:
        prompt = f"Return only the integer answer. {a} + {b} ="
        ids = encode(prompt).to(device)
        if ids.shape[1]+max_new_tokens > context_length:
            raise ValueError("Evaluation prompt plus response cap exceeds model context")
        output, ended = [], False
        for _ in range(max_new_tokens):
            check()
            token = int(model_logits(model, ids)[0, -1].argmax())
            if token in stop_ids:
                ended = True
                break
            output.append(token)
            ids = torch.cat((ids, ids.new_tensor([[token]])), -1)
        text = decode(output)
        rows.append({"prompt": prompt, "expected": a+b, "text": text,
                     "tokens": output, "stop": "declared-stop" if ended else "token-limit",
                     "correct": bool(ended and verify_integer(text, a+b))})
        if on_row is not None:
            on_row(rows[-1])
    return {"mode": "greedy-frozen-original-arithmetic", "n": len(rows),
            "accuracy": sum(row["correct"] for row in rows)/len(rows), "rows": rows}


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunEvidence:
    """Persist progress in one newly created run directory, including failures.

    Each JSON snapshot is replaced atomically. Completed updates also append to
    a JSONL log. This journal is evidence persistence, not checkpoint recovery.
    """
    def __init__(self, output, config):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.data = {"status": "running", "stage": "created", "config": config,
                     "records": [], "initial_heldout_partial": []}
        self.flush()

    def flush(self):
        self.data["journal_seconds"] = time.monotonic()-self.started
        report = self.output/"report.json"
        temporary = self.output/"report.json.tmp"
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False)+"\n")
        temporary.replace(report)
        status = {key: self.data[key] for key in ("status", "stage", "journal_seconds")}
        status["completed_updates"] = len(self.data["records"])
        if "failure" in self.data:
            status["failure"] = self.data["failure"]
        temporary = self.output/"status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2)+"\n")
        temporary.replace(self.output/"status.json")

    def stage(self, name, **fields):
        self.data.update(fields)
        self.data["stage"] = name
        self.flush()

    def baseline_row(self, row):
        self.data["initial_heldout_partial"].append(row)
        self.flush()

    def record(self, row):
        with (self.output/"metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
            handle.flush()
        self.data["records"].append(row)
        self.flush()

    def fail(self, error):
        self.data["status"] = "failed"
        self.data["failure"] = {"type": type(error).__name__, "message": str(error),
                                "stage": self.data["stage"]}
        self.flush()

    def complete(self, manifest):
        self.data.update(manifest)
        self.data.update(status="completed", stage="completed")
        self.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True, help="Recorded immutable upstream commit SHA")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2323)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt-mode", choices=("raw", "chat"), default="chat")
    parser.add_argument("--template", type=Path,
                        help="Explicit template if absent locally; must match a saved SFT genealogy")
    parser.add_argument("--max-seconds", type=float, default=600.,
                        help="Wall-clock cap checked between loading/generation/backward stages")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be a full immutable 40-character commit SHA")
    if not (1 <= args.updates <= 16 and 2 <= args.group_size <= 8 and 1 <= args.max_new_tokens <= 64):
        parser.error("bounded lab: updates1..16, group2..8, generation1..64")
    if not args.model_dir.is_dir() or args.output.exists() or args.lr <= 0 or not 0 < args.max_seconds <= 1800:
        parser.error("need existing local model dir, positive lr and unused output path")
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    journal = RunEvidence(args.output, config)
    try:
        execute_run(args, journal)
    except BaseException as error:
        journal.fail(error)
        raise


def execute_run(args, journal):
    started = time.monotonic()
    minimum_available = float("inf")
    def resource_guard():
        nonlocal minimum_available
        if time.monotonic()-started >= args.max_seconds:
            raise RuntimeError("Bounded RLVR wall-clock budget exhausted")
        measured = guard_memory()
        minimum_available = min(minimum_available, measured)
        return measured
    journal.stage("before-load")
    resource_guard()
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version
    journal.stage("tokenizer-load")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    resource_guard()
    if tokenizer.eos_token_id is None:
        raise ValueError("the supplied tokenizer has no EOS token ID")
    if (args.model_dir/"adapter_config.json").exists():
        raise ValueError("Supply a full or explicitly merged model, not a PEFT adapter directory")
    genealogy_path = args.model_dir/"course-genealogy.json"
    genealogy = json.loads(genealogy_path.read_text()) if genealogy_path.is_file() else None
    encode, stop_ids, interface = prompt_contract(tokenizer, args.prompt_mode, genealogy,
        args.template.read_text() if args.template is not None else None)
    journal.stage("model-load", interface=interface, parent_genealogy=genealogy)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    if args.device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Declared CUDA mode requires verified BF16 support")
        torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
                                                torch_dtype=dtype, attn_implementation="sdpa").to(args.device)
    resource_guard()
    journal.stage("reference-copy")
    reference = deepcopy(model).eval()
    resource_guard()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = ((2, 3), (4, 5), (3, 6), (1, 7))
    heldout_pairs = ((2, 6), (3, 4), (5, 5), (7, 2))
    if set(pairs) & set(heldout_pairs):
        raise RuntimeError("Frozen arithmetic prompt pairs overlap")
    context = getattr(model.config, "max_position_embeddings", None)
    if context is None:
        raise ValueError("cannot determine the supplied model's context bound")
    decode = lambda tokens: tokenizer.decode(tokens, skip_special_tokens=True)
    journal.stage("initial-evaluation", frozen_panel_sha256=hashlib.sha256(repr(heldout_pairs).encode()).hexdigest())
    initial_panel = evaluate_model(model, heldout_pairs, encode, decode, stop_ids,
                                    args.max_new_tokens, int(context), args.device, resource_guard,
                                    on_row=journal.baseline_row)
    journal.stage("ready-for-updates", initial_heldout=initial_panel)
    records = journal.data["records"]
    for update in range(args.updates):
        a, b = pairs[update % len(pairs)]
        prompt = f"Return only the integer answer. {a} + {b} ="
        ids = encode(prompt).to(args.device)
        journal.stage("update", active_update=update+1, active_prompt=prompt,
                      behavior_version=update)
        result = update_model(model, reference, ids, a+b,
                              decode,
                              tokenizer.eos_token_id, optimizer, generator,
                              args.group_size, args.max_new_tokens, before_stage=resource_guard,
                              context_length=int(context), stop_ids=stop_ids)
        result.update({"update": update+1, "behavior_version": update,
                       "policy_version": update+1, "prompt": prompt})
        journal.record(result)
        journal.stage("post-update")
        result["host_available_gib"] = resource_guard()
        journal.flush()
    journal.stage("final-evaluation")
    final_panel = evaluate_model(model, heldout_pairs, encode, decode, stop_ids,
                                  args.max_new_tokens, int(context), args.device, resource_guard)
    journal.stage("source-hashing", final_heldout=final_panel)
    files = [path for path in args.model_dir.iterdir() if path.is_file()
             and path.suffix in (".json", ".safetensors", ".bin", ".model", ".txt")]
    manifest = {"mode": "local-model-bounded-smoke", "upstream_revision": args.revision,
                "revision_evidence": "user-supplied metadata; local file hashes identify actual input bytes",
                "local_source_hashes": {path.name: hash_file(path) for path in files},
                "config": {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()},
                "torch_version": torch.__version__, "transformers_version": transformers_version,
                "python_version": platform.python_version(), "machine": platform.machine(),
                "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                "attention_backend": "Transformers SDPA; actual dispatched kernel requires profiling",
                "dtype": str(dtype), "minimum_sampled_memavailable_gib": minimum_available,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
                "source_command": [sys.executable, *sys.argv],
                "interface": interface, "parent_genealogy": genealogy,
                "seconds": time.monotonic()-started, "records": records,
                "frozen_panel_sha256": hashlib.sha256(repr(heldout_pairs).encode()).hexdigest(),
                "initial_heldout": initial_panel, "final_heldout": final_panel,
                "evaluation": "four original held-out integer prompts; descriptive pilot, no broad reasoning claim",
                "recovery": "final state saved; this runner provides no exact-resume command"}
    journal.stage("checkpoint-export", **manifest)
    resource_guard()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "reference_revision": args.revision, "config": manifest["config"],
                "rollout_generator_state": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                "completed_updates": args.updates, "interface": interface},
               args.output/"checkpoint.pt")
    model.save_pretrained(args.output/"policy", safe_serialization=True)
    tokenizer.save_pretrained(args.output/"policy")
    (args.output/"policy"/"course-genealogy.json").write_text(json.dumps({
        "kind": "full-HF-model", "objective": "course-response-mean-GRPO",
        "parent_local_source_hashes": manifest["local_source_hashes"],
        "upstream_revision_metadata": args.revision, "template_sha256": interface["template_sha256"],
        "evaluation_sha256": manifest["frozen_panel_sha256"],
        "data_sha256": hashlib.sha256(repr(pairs).encode()).hexdigest()}, indent=2)+"\n")
    manifest["seconds"] = time.monotonic()-started
    journal.complete(manifest)
    print(json.dumps({"output": str(args.output), "updates": len(records),
                      "seconds": manifest["seconds"]}))


if __name__ == "__main__":
    main()
