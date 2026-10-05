"""Chapter 6 CPU teaching contracts; not a production data loader or trainer."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math

import torch
from torch.nn import functional as F

from dongxi_llms.decoder_lab import DecoderConfig, TinyDecoder

BOS, EOS, IGNORE = 257, 256, -100
TOKENIZER = "utf8-bytes-v1:bytes=0..255,eos=256,bos=257"
# Original teaching sentences authored for this repository, not a benchmark.
TRAIN = (
    ("train-01", "The cat rests beside the window."),
    ("train-02", "The dog waits beside the door."),
    ("train-03", "A river carries water toward the sea."),
    ("train-04", "A road connects the village to the town."),
    ("train-05", "We measure loss before changing the recipe."),
    ("train-06", "A checkpoint remembers more than model weights."),
    ("train-07", "Small batches can contribute to one update."),
    ("train-08", "Validation asks about text outside the training split."),
)
VALID = (
    ("valid-01", "The cat waits beside the door."),
    ("valid-02", "We measure loss on held out text."),
)


def fingerprint(documents):
    return hashlib.sha256(repr(tuple(documents)).encode()).hexdigest()


def check_splits(train, valid):
    """Reject ID overlap and exact normalized text overlap, NOT near duplicates."""
    norm = lambda text: " ".join(text.casefold().split())
    if not train or not valid:
        raise ValueError("Both splits must be nonempty")
    all_ids = [key for key, _ in (*train, *valid)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Document IDs must be unique across splits")
    if {norm(t) for _, t in train} & {norm(t) for _, t in valid}:
        raise ValueError("Normalized text overlaps the held-out split")


def make_windows(documents, length=16):
    """Shift ONCE within each document; right-pad tails; restart context per window."""
    if length <= 0 or not documents:
        raise ValueError("Need positive length and nonempty documents")
    xs, ys, owners = [], [], []
    for key, text in documents:
        sequence = [BOS, *text.encode("utf-8"), EOS]
        inputs, labels = sequence[:-1], sequence[1:]
        for start in range(0, len(labels), length):
            x, y = inputs[start:start+length], labels[start:start+length]
            xs.append(x + [EOS] * (length-len(x)))
            ys.append(y + [IGNORE] * (length-len(y)))
            owners.append(key)
    return {"x": torch.tensor(xs), "y": torch.tensor(ys), "documents": owners}


def fixture(length=16):
    check_splits(TRAIN, VALID)
    return make_windows(TRAIN, length), make_windows(VALID, length)


def make_model(seed=808, dtype=torch.float32):
    cfg = DecoderConfig(vocab=258, width=16, heads=4, kv_heads=2, head_dim=4,
                        layers=2, hidden=32, max_length=64, modern=True,
                        qk_norm=True, tied=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(cfg).to(dtype=dtype)


def loss_sum(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten(),
                           ignore_index=IGNORE, reduction="sum")


def flat_grad(model):
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def accumulation_audit():
    """Different valid lengths expose the mean-of-microbatch-means bug."""
    windows, _ = fixture()
    lengths = (windows["y"] != IGNORE).sum(1)
    indices = [int(lengths.argmax()), int(lengths.argmin())]
    x, y = windows["x"][indices], windows["y"][indices]
    total = int((y != IGNORE).sum())
    grads = []
    for mode in ("full", "correct", "wrong"):
        model = make_model(dtype=torch.float64)
        if mode == "full":
            (loss_sum(model, x, y) / total).backward()
        else:
            for i in range(len(x)):
                denominator = total if mode == "correct" else int((y[i] != IGNORE).sum()) * len(x)
                (loss_sum(model, x[i:i+1], y[i:i+1]) / denominator).backward()
        grads.append(flat_grad(model))
    return {"counts": (y != IGNORE).sum(1).tolist(),
            "correct_error": float((grads[0]-grads[1]).abs().max()),
            "wrong_error": float((grads[0]-grads[2]).abs().max()),
            "x": x, "y": y}


def learning_rate(step, total=24, warmup=3, peak=.01, floor=.001):
    """step is zero-based COMPLETED updates before this update; floor at last update."""
    if not (0 < warmup < total and 0 <= step < total and 0 <= floor <= peak):
        raise ValueError("Invalid schedule bounds")
    if step < warmup:
        return peak * (step+1) / warmup
    progress = (step-warmup+1) / (total-warmup)
    return floor + .5*(peak-floor)*(1+math.cos(math.pi*progress))


def adamw_reference(parameter, gradient, m, v, step, lr, betas=(.9, .999), eps=1e-8, decay=.01):
    """Unfused AdamW recurrence; step is one-based; no AMSGrad/maximize."""
    b1, b2 = betas
    m = b1*m + (1-b1)*gradient
    v = b2*v + (1-b2)*gradient.square()
    update = (m/(1-b1**step)) / ((v/(1-b2**step)).sqrt()+eps)
    return parameter*(1-lr*decay)-lr*update, m, v


def optimizer_audit():
    model = make_model(dtype=torch.float64)
    windows, _ = fixture()
    y = windows["y"][:2]
    (loss_sum(model, windows["x"][:2], y) / (y != IGNORE).sum()).backward()
    parameter = model.token.weight
    before, gradient = parameter.detach().clone(), parameter.grad.detach().clone()
    expected, _, _ = adamw_reference(before, gradient, torch.zeros_like(before),
                                     torch.zeros_like(before), 1, .01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01, foreach=False)
    optimizer.step()
    return {"error": float((parameter.detach()-expected).abs().max()),
            "gradient": gradient.flatten()[:16],
            "sgd_delta": -.01*gradient.flatten()[:16],
            "adamw_delta": (parameter.detach()-before).flatten()[:16]}


def clipping_audit(limit=1.):
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Clipping limit must be positive and finite")
    model = make_model()
    windows, _ = fixture()
    y = windows["y"][:2]
    (50*loss_sum(model, windows["x"][:2], y)/(y != IGNORE).sum()).backward()
    before = flat_grad(model).clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), limit, error_if_nonfinite=True)
    after = flat_grad(model)
    return {"before": float(before.norm()), "after": float(after.norm()),
            "cosine": float(F.cosine_similarity(before, after, dim=0))}


class WindowStream:
    """Single-process shuffle with explicit cursor; no workers/prefetch/distributed state."""
    def __init__(self, size, seed=808):
        if size <= 0:
            raise ValueError("Empty stream")
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor, self.epoch = 0, 0

    def take(self, count):
        if count <= 0:
            raise ValueError("Batch count must be positive")
        selected = []
        for _ in range(count):
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor, self.epoch = 0, self.epoch+1
            selected.append(int(self.order[self.cursor]))
            self.cursor += 1
        return selected

    def state_dict(self):
        return {"size": self.size, "order": self.order.clone(), "cursor": self.cursor,
                "epoch": self.epoch, "rng": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Stream size changed")
        if sorted(state["order"].tolist()) != list(range(self.size)) or not 0 <= state["cursor"] <= self.size:
            raise ValueError("Invalid stream permutation/cursor")
        self.order = state["order"].clone()
        self.cursor, self.epoch = state["cursor"], state["epoch"]
        self.generator.set_state(state["rng"])


@torch.no_grad()
def evaluate(model, windows, batch=3):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0., 0
    try:
        for start in range(0, len(windows["x"]), batch):
            x, y = windows["x"][start:start+batch], windows["y"][start:start+batch]
            total_loss += float(loss_sum(model, x, y))
            total_tokens += int((y != IGNORE).sum())
    finally:
        model.train(was_training)
    return total_loss/total_tokens


class TrainingSession:
    """Bounded float32 CPU fixture. Checkpoints only at completed update boundaries."""
    def __init__(self, total=24):
        if total <= 3:
            raise ValueError("Need more updates than warmup")
        self.train, self.valid = fixture()
        self.model = make_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01,
                                          weight_decay=.01, foreach=False)
        self.stream = WindowStream(len(self.train["x"]))
        self.step, self.tokens, self.total = 0, 0, total
        self.contract = {"format": 1, "tokenizer": TOKENIZER,
                         "train_sha256": fingerprint(TRAIN), "valid_sha256": fingerprint(VALID),
                         "model": asdict(self.model.cfg), "length": 16,
                         "total": total, "warmup": 3, "peak": .01, "floor": .001,
                         "microbatch": 1, "accumulation": 2, "clip": 1.,
                         "precision": "cpu-float32", "seed": 808,
                         "optimizer": "AdamW:betas=.9,.999;eps=1e-8;decay=.01;all-parameters"}

    def update(self):
        lr = learning_rate(self.step, total=self.total)
        indices = self.stream.take(2)
        x, y = self.train["x"][indices], self.train["y"][indices]
        count = int((y != IGNORE).sum())
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        summed = 0.
        for i in range(2):
            loss = loss_sum(self.model, x[i:i+1], y[i:i+1]) / count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; no optimizer update applied")
            loss.backward()
            summed += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        self.step += 1
        self.tokens += count
        return {"step": self.step, "loss": summed, "lr": lr, "grad_norm": float(norm),
                "tokens": self.tokens, "indices": indices}

    def checkpoint(self):
        return deepcopy({"contract": self.contract, "model": self.model.state_dict(),
                         "optimizer": self.optimizer.state_dict(), "stream": self.stream.state_dict(),
                         "step": self.step, "tokens": self.tokens, "torch_rng": torch.get_rng_state()})

    def restore(self, state, *, optimizer=True, stream=True):
        # Optional omissions are intentional negative controls, not supported resume modes.
        if state["contract"] != self.contract:
            raise ValueError("Checkpoint experiment contract changed")
        self.model.load_state_dict(state["model"])
        if optimizer:
            self.optimizer.load_state_dict(deepcopy(state["optimizer"]))
        if stream:
            self.stream.load_state_dict(state["stream"])
        self.step, self.tokens = state["step"], state["tokens"]
        torch.set_rng_state(state["torch_rng"])


def recovery_audit(path, total=24):
    """Round-trip a trusted local checkpoint, then compare three continuation branches."""
    reference = TrainingSession(total)
    history = []
    for _ in range(total//2):
        history.append(reference.update())
    torch.save(reference.checkpoint(), path)
    tail = [reference.update() for _ in range(total-total//2)]
    history += tail
    result = {"history": history, "reference_validation": evaluate(reference.model, reference.valid)}
    for name, opt, data in (("complete", True, True), ("no_optimizer", False, True), ("no_cursor", True, False)):
        restored = TrainingSession(total)
        restored.restore(torch.load(path, weights_only=True, map_location="cpu"), optimizer=opt, stream=data)
        replay = [restored.update() for _ in tail]
        error = max(float((a.detach()-b.detach()).abs().max()) for a, b in zip(reference.model.parameters(), restored.model.parameters()))
        result[name] = {"parameter_error": error, "same_history": replay == tail,
                        "same_batches": [r["indices"] for r in replay] == [r["indices"] for r in tail],
                        "validation": evaluate(restored.model, restored.valid)}
    return result
