#!/usr/bin/env python3
"""Optional bounded full-parameter DPO on a local SFT checkpoint, no implicit run.

    PYTHONPATH=src python scripts/run_chapter11_spark_dpo.py --help

JSONL pairs: id, prompt (list of role/content messages), chosen, rejected.
Independent generation JSONL: id, prompt (messages), expected (exact text).
This runner deliberately rejects truncation and template-prefix mismatches.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import torch
from dongxi_llms.dpo_lab import model_pair_loss

_OUTPUT = None


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable; use the Linux Spark lane")


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty rows with unique ids")
    return rows


def restore_saved_template(checkpoint):
    """Read current HF Jinja storage, with a legacy JSON-template fallback.

    save_pretrained normally writes chat_template.jinja and removes the template
    from tokenizer_config.json. Reading only the latter breaks SFT -> DPO.
    """
    checkpoint = Path(checkpoint)
    jinja_path = checkpoint / "chat_template.jinja"
    config_path = checkpoint / "tokenizer_config.json"
    if jinja_path.is_file():
        template = jinja_path.read_text()
    elif config_path.is_file():
        template = json.loads(config_path.read_text()).get("chat_template")
    else:
        template = None
    if not isinstance(template, str) or not template:
        raise ValueError("SFT checkpoint has no single audited saved chat template")
    genealogy_path = checkpoint / "course-genealogy.json"
    if genealogy_path.is_file():
        expected = json.loads(genealogy_path.read_text()).get("template_sha256")
        actual = hashlib.sha256(template.encode()).hexdigest()
        if expected != actual:
            raise ValueError("Saved template differs from the checkpoint genealogy")
    return template


def encode_pair(tokenizer, row, limit):
    prompt = row["prompt"]
    if not isinstance(prompt, list) or prompt[-1]["role"] == "assistant":
        raise ValueError("prompt must be messages ending before the assistant response")
    prefix = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    branches = []
    for key in ("chosen", "rejected"):
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError("Both answers must be nonempty strings")
        ids = tokenizer.apply_chat_template(prompt + [{"role": "assistant", "content": row[key]}],
                                             tokenize=True, add_generation_prompt=False, enable_thinking=False)
        if ids[:len(prefix)] != prefix:
            raise ValueError("Template does not preserve generation prefix; inspect the pinned tokenizer")
        if len(ids) > limit or len(ids) <= len(prefix):
            raise ValueError("Sequence exceeds limit or has an empty completion; filter explicitly")
        branches.append((ids, [False] * len(prefix) + [True] * (len(ids) - len(prefix))))
    return branches


def collate(encoded, branch, pad, device):
    records = [item[branch] for item in encoded]
    length = max(len(ids) for ids, _ in records)
    ids, attention, mask = [], [], []
    for tokens, completion in records:
        n = length - len(tokens)
        ids.append(tokens + [pad] * n)
        attention.append([1] * len(tokens) + [0] * n)
        mask.append(completion + [False] * n)
    return (torch.tensor(ids, device=device), torch.tensor(attention, device=device),
            torch.tensor(mask, dtype=torch.bool, device=device))


def main():
    global _OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local HF SFT checkpoint directory")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True, help="Exact 40-character HF commit")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--evaluation", required=True, help="Independent prompt/expected JSONL")
    parser.add_argument("--output", required=True, help="New output directory; must not exist")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1818)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not (len(args.tokenizer_revision) == 40 and all(c in "0123456789abcdef" for c in args.tokenizer_revision)):
        parser.error("tokenizer revision must be an immutable 40-character lowercase commit")
    if not 1 <= args.updates <= 1000 or not 1 <= args.accumulation <= 16 or args.beta <= 0 or args.lr <= 0:
        parser.error("Invalid bounded update/accumulation/beta/lr contract")
    if not 16 <= args.max_length <= 1024 or not 1 <= args.max_new_tokens <= 128:
        parser.error("Length limits exceed this teaching runner's bounds")
    if not Path(args.checkpoint).is_dir() or Path(args.output).exists():
        parser.error("Require existing local checkpoint and new output directory")
    checkpoint = Path(args.checkpoint)
    if (checkpoint / "adapter_config.json").exists() or not (checkpoint / "config.json").is_file():
        parser.error("DPO requires a full or merged HF checkpoint; merge a LoRA adapter explicitly first")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("pytorch_model*.bin")):
        parser.error("Full HF model weights are missing from the checkpoint")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or host_available() < 25 * 1024**3:
        raise RuntimeError("Require BF16-capable CUDA Spark and at least 25 GiB available host memory")
    output = Path(args.output)
    output.mkdir(parents=True)
    _OUTPUT = output
    start = time.monotonic()
    minimum_available = host_available()
    (output / "status.json").write_text(json.dumps({"status": "running", "config": vars(args)}))
    def guard():
        nonlocal minimum_available
        minimum_available = min(minimum_available, host_available())
        if minimum_available < 25 * 1024**3 or time.monotonic() - start > 1800:
            raise RuntimeError("Sampled host memory reserve or 30-minute whole-run wall cap crossed")
    guard()
    train, valid, evaluation = [load_rows(p) for p in (args.train, args.validation, args.evaluation)]
    if len(train) > 4096 or len(valid) > 128 or len(evaluation) > 64:
        raise ValueError("Teaching runner limits: 4096 train, 128 validation, 64 independent prompts")
    sets = [set(row["id"] for row in rows) for rows in (train, valid, evaluation)]
    if any(sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("IDs overlap across train, validation and independent evaluation")
    prompts = [set(json.dumps(row["prompt"], sort_keys=True) for row in rows) for rows in (train, valid, evaluation)]
    if any(prompts[a] & prompts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Exact rendered-message prompts overlap across splits")
    from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision,
                                             local_files_only=not args.allow_download)
    tokenizer.chat_template = restore_saved_template(checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = [tokenizer.eos_token_id]
    turn_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if turn_end is not None and turn_end != tokenizer.unk_token_id and turn_end not in stop_ids:
        stop_ids.append(turn_end)
    encoded_train = [encode_pair(tokenizer, r, args.max_length) for r in train]
    encoded_valid = [encode_pair(tokenizer, r, args.max_length) for r in valid]
    guard()
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
             torch_dtype=torch.float32, attn_implementation="sdpa").cuda()
    guard()
    reference = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
             torch_dtype=torch.float32, attn_implementation="sdpa").cuda().eval().requires_grad_(False)
    guard()
    model.config.use_cache = False
    reference.config.use_cache = False
    model.gradient_checkpointing_enable()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
    def wrapper(network):
        return lambda ids, attention: network(input_ids=ids, attention_mask=attention, use_cache=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    rng = torch.Generator().manual_seed(args.seed)
    identity = {**vars(args), "torch": torch.__version__, "transformers": transformers_version,
                "python": platform.python_version(), "device": torch.cuda.get_device_name(),
                "cuda_runtime": torch.version.cuda,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "gpu_driver": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version",
                                                       "--format=csv,noheader"], text=True).strip(),
                "source_sha256": {str(Path(__file__).name): file_digest(__file__),
                                  "dpo_lab.py": file_digest(Path(__file__).resolve().parents[1] / "src/dongxi_llms/dpo_lab.py")},
                "input_sha256": {p: file_digest(p)
                                  for p in (args.train, args.validation, args.evaluation)},
                "checkpoint_files": {p.name: file_digest(p)
                      for p in Path(args.checkpoint).glob("*") if p.is_file() and p.suffix in (".json", ".safetensors", ".bin")},
                "template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()}
    identity["generation_stop_token_ids"] = stop_ids
    (output / "identity.json").write_text(json.dumps(identity, indent=2))
    guard()
    def independent_evaluate():
        model.eval()
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in evaluation:
                guard()
                prefix = tokenizer.apply_chat_template(row["prompt"], tokenize=True,
                       add_generation_prompt=True, enable_thinking=False)
                if len(prefix) + args.max_new_tokens > args.max_length:
                    raise ValueError("Independent prompt plus generation exceeds the declared context")
                ids = torch.tensor([prefix], device="cuda")
                sampled = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                          max_new_tokens=args.max_new_tokens, use_cache=False, pad_token_id=tokenizer.pad_token_id,
                          eos_token_id=stop_ids)
                text = tokenizer.decode(sampled[0, len(prefix):], skip_special_tokens=True)
                rows.append({"id": row["id"], "generated": text, "expected": row["expected"],
                             "exact_match": text.strip() == row["expected"].strip()})
                guard()
        return rows
    before = independent_evaluate()
    (output / "evaluation-before.json").write_text(json.dumps(before, indent=2))
    metrics = []
    model.train()
    for step in range(args.updates):
        guard()
        optimizer.zero_grad(set_to_none=True)
        losses, margins = [], []
        for _ in range(args.accumulation):
            guard()
            i = int(torch.randint(len(train), (), generator=rng))
            encoded = [encoded_train[i]]
            branches = [collate(encoded, b, tokenizer.pad_token_id, "cuda") for b in (0, 1)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, observations = model_pair_loss(wrapper(model), wrapper(reference), *branches, beta=args.beta)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            (loss / args.accumulation).backward()
            losses.append(float(loss.detach()))
            margins.append(float(observations["margin"].mean()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        item = {"update": step + 1, "loss": sum(losses) / len(losses),
                "margin": sum(margins) / len(margins), "gradient_norm": float(norm),
                "mem_available_bytes": host_available(), "elapsed_seconds": time.monotonic() - start}
        metrics.append(item)
        with (output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(item) + "\n")
    model.eval()
    validation = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for encoded in encoded_valid:
            guard()
            branches = [collate([encoded], b, tokenizer.pad_token_id, "cuda") for b in (0, 1)]
            loss, observations = model_pair_loss(wrapper(model), wrapper(reference), *branches, beta=args.beta)
            validation.append({"loss": float(loss), "margin": float(observations["margin"].mean())})
    after = independent_evaluate()
    guard()
    model.save_pretrained(output / "policy", safe_serialization=True)
    tokenizer.save_pretrained(output / "policy")
    (output / "policy" / "course-genealogy.json").write_text(json.dumps({
         "kind": "full-HF-DPO-policy", "parent_checkpoint": str(checkpoint.resolve()),
         "parent_checkpoint_sha256": identity["checkpoint_files"],
         "tokenizer": args.tokenizer, "tokenizer_revision": args.tokenizer_revision,
         "template_sha256": identity["template_sha256"],
         "training_input_sha256": identity["input_sha256"]}, indent=2))
    torch.save({"optimizer": optimizer.state_dict(), "rng": rng.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "updates": args.updates}, output / "training-state.pt")
    (output / "result.json").write_text(json.dumps({"before": before, "after": after,
               "validation": validation, "minimum_sampled_mem_available_bytes": minimum_available,
               "wall_seconds": time.monotonic() - start, "reference_has_gradients": any(p.grad is not None for p in reference.parameters())}, indent=2))
    (output / "status.json").write_text(json.dumps({"status": "completed", "updates": args.updates,
                                                  "wall_seconds": time.monotonic() - start}))
    print(json.dumps({"output": str(output), "updates": args.updates, "status": "completed"}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _OUTPUT is not None:
            (_OUTPUT / "status.json").write_text(json.dumps({"status": "failed",
                "error_type": type(error).__name__, "error": str(error), "partial_outputs_retained": True}))
        raise
