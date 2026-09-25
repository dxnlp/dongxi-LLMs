"""Single-device story training with auditable state and a bounded command line.

Uses the course decoder's parameter layout, not pretrained model weights.
The transparent notebook implementation remains unchanged.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from dongxi_llms.decoder_lab import DecoderConfig, MultiHeadAttention, TinyDecoder, rope
from dongxi_llms.pretraining_lab import WindowStream, learning_rate
from dongxi_llms.stories_data import EOS, IGNORE, Windows, digest


class SDPAAttention(MultiHeadAttention):
    def forward(self, x):
        b, t, _ = x.shape
        def split(layer):
            return layer(x).reshape(b, t, self.cfg.heads, self.cfg.head_dim).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        # Trigonometry in FP32 under autocast; retain FP64 for reference audits.
        compute = torch.float64 if q.dtype == torch.float64 else torch.float32
        positions = torch.arange(t, device=x.device)
        q = rope(q.to(compute), positions).to(v.dtype)
        k = rope(k.to(compute), positions).to(v.dtype)
        mixed = F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=True)
        return self.out(mixed.transpose(1, 2).contiguous().reshape(b, t, -1))


class StoriesDecoder(TinyDecoder):
    def __init__(self, cfg, activation_checkpointing=False):
        if not cfg.modern or cfg.qk_norm or cfg.heads != cfg.kv_heads:
            raise ValueError("Stories path requires modern MHA without QK norm")
        super().__init__(cfg)
        for block in self.blocks:
            replacement = SDPAAttention(cfg)
            replacement.load_state_dict(block.attn.state_dict())
            block.attn = replacement
        self.activation_checkpointing = activation_checkpointing

    @staticmethod
    def apply_block(block, x):
        x = x + block.attn(block.norm1(x))
        return x + block.mlp(block.norm2(x))

    def features(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.cfg.max_length:
            raise ValueError("Expected nonempty IDs [B,T] within context budget")
        x = self.token(ids)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = checkpoint(self.apply_block, block, x, use_reentrant=False)
            else:
                x = self.apply_block(block, x)
        return self.final_norm(x)

    def forward(self, ids):
        return self.lm_head(self.features(ids))


def baseline(context=1024):
    return DecoderConfig(vocab=50257, width=512, heads=8, kv_heads=8,
                         head_dim=64, layers=12, hidden=1536, max_length=context,
                         modern=True, qk_norm=False, tied=True)


@dataclass(frozen=True)
class Recipe:
    total_updates: int = 3
    warmup: int = 1
    peak_lr: float = 3e-4
    floor_lr: float = 3e-5
    microbatch: int = 1
    accumulation: int = 1
    weight_decay: float = .1
    clip: float = 1.
    seed: int = 909
    bf16: bool = False
    activation_checkpointing: bool = True

    def __post_init__(self):
        if not 0 < self.warmup < self.total_updates:
            raise ValueError("Warmup must lie within the update horizon")
        if min(self.microbatch, self.accumulation) <= 0:
            raise ValueError("Batch dimensions must be positive")
        if not all(math.isfinite(v) for v in (self.peak_lr, self.floor_lr, self.weight_decay, self.clip)):
            raise ValueError("Recipe values must be finite")
        if not 0 <= self.floor_lr <= self.peak_lr or self.peak_lr <= 0 or self.clip <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings")


class Session:
    def __init__(self, cfg, recipe, train, device="cpu"):
        self.recipe, self.train_data = recipe, train
        self.device = torch.device(device)
        if recipe.bf16 and (self.device.type != "cuda" or not torch.cuda.is_bf16_supported()):
            raise ValueError("BF16 recipe requires a verified CUDA device")
        torch.manual_seed(recipe.seed)
        self.model = StoriesDecoder(cfg, recipe.activation_checkpointing).to(self.device)
        decay, no_decay = [], []
        for parameter in self.model.parameters():
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": recipe.weight_decay},
             {"params": no_decay, "weight_decay": 0.}],
            lr=recipe.peak_lr, betas=(.9, .95), eps=1e-8, foreach=False)
        self.stream = WindowStream(len(train), recipe.seed + 1)
        self.step, self.tokens = 0, 0
        self.contract = dict(schema=1, model=asdict(cfg), recipe=asdict(recipe),
                             data=train.identity, device=str(self.device),
                             torch=str(torch.__version__), attention="pytorch-sdpa-auto-causal",
                             implementation={name: digest(Path(__file__).with_name(name))
                                 for name in ("stories_training.py", "stories_data.py",
                                              "decoder_lab.py", "pretraining_lab.py")})

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.recipe.bf16 else nullcontext()

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def summed_loss(self, x, y):
        with self.autocast():
            logits = self.model(x.to(self.device))
            # Explicit FP32 CE and token-weighted reductions, never mean-of-means.
            return F.cross_entropy(logits.float().flatten(0, 1), y.to(self.device).flatten(),
                                   ignore_index=IGNORE, reduction="sum")

    def update(self):
        if self.step >= self.recipe.total_updates:
            raise ValueError("Fixed schedule exhausted; specify a new experiment")
        self.model.train()
        self.sync()
        start = time.monotonic()
        batches = [self.train_data.batch(self.stream.take(self.recipe.microbatch))
                   for _ in range(self.recipe.accumulation)]
        count = sum(int((y != IGNORE).sum()) for _, y in batches)
        if count == 0:
            raise ValueError("No valid training targets")
        lr = learning_rate(self.step, self.recipe.total_updates, self.recipe.warmup,
                           self.recipe.peak_lr, self.recipe.floor_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = 0.
        for x, y in batches:
            loss = self.summed_loss(x, y) / count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            loss_total += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip,
                                             error_if_nonfinite=True)
        if float(norm) == 0:
            raise FloatingPointError("Zero gradient norm")
        probe = self.model.blocks[0].attn.q.weight
        before = probe.detach().clone()
        self.optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise FloatingPointError("Nonfinite updated parameters")
        self.step += 1
        self.tokens += count
        self.sync()
        elapsed = time.monotonic() - start
        return dict(update=self.step, loss=loss_total, lr=lr, gradient_norm=float(norm),
                    valid_targets=count, cumulative_targets=self.tokens, seconds=elapsed,
                    valid_targets_per_second=count / elapsed,
                    first_q_max_parameter_change=float((probe.detach()-before).abs().max()),
                    processed_positions=self.recipe.microbatch*self.recipe.accumulation*self.model.cfg.max_length)

    @torch.no_grad()
    def evaluate(self, valid, max_windows=None):
        was_training = self.model.training
        self.model.eval()
        total, count = 0., 0
        stop = len(valid) if max_windows is None else min(len(valid), max_windows)
        if stop < 1:
            raise ValueError("Validation must contain at least one window")
        selection = (torch.randperm(len(valid), generator=torch.Generator().manual_seed(909))[:stop].tolist()
                     if stop < len(valid) else list(range(stop)))
        try:
            for start in range(0, stop, self.recipe.microbatch):
                x, y = valid.batch(selection[start:start+self.recipe.microbatch])
                total += float(self.summed_loss(x, y))
                count += int((y != IGNORE).sum())
        finally:
            self.model.train(was_training)
        return dict(loss=total / count, valid_targets=count, windows=stop,
                    complete_prepared_split=stop == len(valid), selection_seed=909)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        state = dict(contract=self.contract, model=self.model.state_dict(),
                     optimizer=self.optimizer.state_dict(), stream=self.stream.state_dict(),
                     step=self.step, tokens=self.tokens, rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [])
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)

    def restore(self, path):
        # Tensor/primitive-only checkpoint; never enable unrestricted pickle loading.
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["contract"] != self.contract:
            raise ValueError("Checkpoint model/data/recipe/environment contract changed")
        if not 0 <= state["step"] <= self.recipe.total_updates or state["tokens"] < 0:
            raise ValueError("Invalid checkpoint counters")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.stream.load_state_dict(state["stream"])
        self.step, self.tokens = state["step"], state["tokens"]
        torch.set_rng_state(state["rng"])
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])


PROMPTS = (
    "Once upon a time, a little rabbit lived near a forest.",
    "Lily put her red ball in a box. Then she went outside.",
    "Tom wanted to fly his kite, but there was no wind.",
)


@torch.no_grad()
def samples(session, tokenizer, max_new=32):
    previous = session.model.training
    session.model.eval()
    result = []
    try:
        for prompt in PROMPTS:
            prefix = [EOS, *tokenizer.encode(prompt, add_special_tokens=False).ids]
            for method in ("greedy", "sample"):
                generator = torch.Generator(device=session.device).manual_seed(909)
                ids = torch.tensor([prefix], device=session.device)
                generated = []
                ended = False
                for _ in range(min(max_new, session.model.cfg.max_length-len(prefix))):
                    with session.autocast():
                        logits = session.model.lm_head(session.model.features(ids)[:, -1]).float()
                    if method == "greedy":
                        token = int(logits.argmax(-1).item())
                    else:
                        # Fixed temperature .8, no top-k/p filtering in this baseline panel.
                        token = int(torch.multinomial((logits/.8).softmax(-1), 1, generator=generator).item())
                    generated.append(token)
                    ids = torch.cat((ids, torch.tensor([[token]], device=session.device)), 1)
                    if token == EOS:
                        ended = True
                        break
                result.append(dict(prompt=prompt, method=method, seed=909,
                                   temperature=None if method == "greedy" else .8,
                                   token_ids=generated, continuation=tokenizer.decode(generated),
                                   ended_with_eos=ended, max_new_tokens=max_new))
    finally:
        session.model.train(previous)
    return result


@torch.no_grad()
def activation_summary(session, valid):
    """Fixed first validation window; summarize real positions, excluding padding."""
    ids, labels = valid.batch([0])
    mask = (labels != IGNORE).to(session.device)
    result = []
    def record(name, value):
        selected = value[mask].float()
        result.append(dict(component=name, rms=float(selected.square().mean().sqrt()),
                           max_abs=float(selected.abs().max())))
    with session.autocast():
        x = session.model.token(ids.to(session.device))
        record("embedding", x)
        for i, block in enumerate(session.model.blocks):
            x = session.model.apply_block(block, x)
            record(f"residual_after_block_{i}", x)
        record("final_norm", session.model.final_norm(x))
    if not all(math.isfinite(row[metric]) for row in result for metric in ("rms", "max_abs")):
        raise FloatingPointError("Nonfinite observed activations")
    return result


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    raise RuntimeError("Cannot read Linux MemAvailable")


class MemoryMonitor:
    def __init__(self, root, reserve=25., seconds=600.):
        self.root, self.reserve, self.seconds = Path(root), reserve, seconds
        self.stop = threading.Event()
        self.minimum = available_gib()
        self.reason = None

    def __enter__(self):
        if self.minimum < self.reserve:
            raise RuntimeError("Insufficient host memory reserve before launch")
        def watch():
            start = time.monotonic()
            with (self.root / "host-memory.jsonl").open("x") as log:
                while not self.stop.is_set():
                    value = available_gib()
                    elapsed = time.monotonic()-start
                    self.minimum = min(self.minimum, value)
                    log.write(json.dumps(dict(seconds=elapsed, available_gib=value)) + "\n")
                    log.flush()
                    if value < self.reserve or elapsed > self.seconds:
                        self.reason = "memory reserve" if value < self.reserve else "wall-clock limit"
                        os.kill(os.getpid(), signal.SIGINT)
                        return
                    self.stop.wait(.2)
        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        summary = dict(minimum_sampled_available_gib=self.minimum, abort_reason=self.reason,
                       sampling_seconds=.2, reserve_gib=self.reserve)
        if torch.cuda.is_initialized():
            summary.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                peak = Path("/sys/fs/cgroup") / line[3:].lstrip("/") / "memory.peak"
                if peak.exists():
                    summary["cgroup_peak_bytes"] = int(peak.read_text())
        (self.root / "memory-summary.json").write_text(json.dumps(summary, indent=2)+"\n")


def run(data, output, device="cuda", total=3, stop_after=None, resume=None,
        microbatch=1, accumulation=1, warmup=1, valid_windows=2, sample_tokens=16,
        checkpoint_every=2, max_seconds=600., peak_lr=3e-4, floor_lr=3e-5):
    from tokenizers import Tokenizer
    root = Path(output)
    if checkpoint_every < 1 or max_seconds <= 0 or sample_tokens < 1 or valid_windows < 1:
        raise ValueError("Positive output/evaluation/time bounds required")
    recipe = Recipe(total_updates=total, warmup=warmup, microbatch=microbatch,
                    accumulation=accumulation, bf16=device.startswith("cuda"),
                    peak_lr=peak_lr, floor_lr=floor_lr)
    stop_after = total if stop_after is None else stop_after
    if not 0 < stop_after <= total:
        raise ValueError("Stop must lie within the fixed schedule")
    root.mkdir(parents=True, exist_ok=False)
    with MemoryMonitor(root, seconds=max_seconds):
        run_start = time.monotonic()
        train, valid = Windows(data, "train"), Windows(data, "valid")
        session = Session(baseline(train.length), recipe, train, device)
        if resume:
            session.restore(resume)
        if session.step >= stop_after:
            raise ValueError("Resume has already reached requested stop")
        identity = dict(contract=session.contract, stop_after=stop_after,
                        resume=str(resume) if resume else None, valid_windows=valid_windows,
                        sample_tokens=sample_tokens,
                        validation_selection_seed=909,
                        sources={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                        data_manifest=train.manifest)
        (root / "run.json").write_text(json.dumps(identity, indent=2)+"\n")
        tok = Tokenizer.from_file(str(Path(data) / "tokenizer.json"))
        def observe(label):
            observation = dict(update=session.step, validation=session.evaluate(valid, valid_windows),
                               samples=samples(session, tok, sample_tokens),
                               activations=activation_summary(session, valid))
            (root / f"{label}.json").write_text(json.dumps(observation, indent=2)+"\n")
            print(json.dumps({"observation": label, "update": session.step,
                              "validation": observation["validation"]}), flush=True)
        observe("initial")
        with (root / "metrics.jsonl").open("x") as log:
            while session.step < stop_after:
                # Reserve shutdown time before the independent hard deadline.
                if time.monotonic()-run_start > max_seconds-min(120., max_seconds*.1):
                    break
                record = session.update()
                log.write(json.dumps(record)+"\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if session.step % checkpoint_every == 0 or session.step == stop_after:
                    session.save(root / f"update-{session.step:06d}.pt")
                    if session.step != stop_after:
                        observe(f"update-{session.step:06d}")
        last_checkpoint = root / f"update-{session.step:06d}.pt"
        if not last_checkpoint.exists():
            session.save(last_checkpoint)
        if time.monotonic()-run_start < max_seconds-60:
            observe("final")
        completion = dict(completed_updates=session.step, cumulative_targets=session.tokens,
                          schedule_complete=session.step == total,
                          requested_stop_reached=session.step == stop_after,
                          stopped_for_time_budget=session.step < stop_after,
                          seconds=time.monotonic()-run_start)
        (root / "completion.json").write_text(json.dumps(completion, indent=2)+"\n")
        print(json.dumps(completion), flush=True)
        return session.step
