"""Original tiny causal TEXT reward model, terminal/process labels and frozen export.

This is not the known-feature shortcut model, a pretrained evaluator or a
human-feedback dataset. All fitting/scoring is bounded CPU work on authored text.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from .run_identity import canonical_hash, file_digest, tokenizer_interface, assert_compatible

SPECIALS = ("<pad>", "<unk>", "<sep>", "<step>", "<eos>")
SCHEMA = "dongxi-text-reward-v1"
TOKEN_PATTERN = r"\w+|[^\w\s]"
TEMPLATE = "prompt <sep> completion; optional inclusive <eos> endpoint"


class TextVocabulary:
    """Exact original encoding rules; unknown handling is explicit and serialized."""
    pad_token_id, unk_token_id, sep_token_id, eos_token_id = 0, 1, 2, 4
    additional_special_tokens_ids = [3]

    def __init__(self, tokens, unknown_policy="unk"):
        self.tokens = list(tokens)
        if self.tokens[:5] != list(SPECIALS) or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary special IDs/order and unique token meanings are required")
        if not 5 <= len(self.tokens) <= 10_000 or any(not isinstance(t, str) or not t for t in self.tokens):
            raise ValueError("Invalid or oversized original vocabulary")
        if unknown_policy not in {"unk", "error"}:
            raise ValueError("Unknown policy must be unk or error")
        self.unknown_policy = unknown_policy
        self.mapping = {token: i for i, token in enumerate(self.tokens)}
        self.backend_tokenizer = self

    @staticmethod
    def pieces(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be nonempty")
        return re.findall(TOKEN_PATTERN, text.casefold(), flags=re.UNICODE)

    @classmethod
    def fit(cls, texts):
        pieces = {piece for text in texts for piece in cls.pieces(text)}
        return cls([*SPECIALS, *sorted(pieces - set(SPECIALS))])

    def get_vocab(self):
        return dict(self.mapping)

    def to_str(self):
        return json.dumps({"model": {"type": "OriginalWordLookup", "vocab": self.mapping},
                           "normalizer": "unicode-casefold", "pre_tokenizer": TOKEN_PATTERN,
                           "unknown_policy": self.unknown_policy,
                           "special_tokens": dict(zip(SPECIALS, range(5)))}, sort_keys=True)

    def encode(self, text):
        pieces = self.pieces(text)
        unknown = [piece for piece in pieces if piece not in self.mapping]
        if unknown and self.unknown_policy == "error":
            raise ValueError(f"Unknown tokens under frozen vocabulary: {unknown}")
        return [self.mapping.get(piece, self.unk_token_id) for piece in pieces], unknown

    def interface(self, include_eos=True):
        template = TEMPLATE + ("; include EOS" if include_eos else "; exclude EOS")
        return tokenizer_interface(self, template=template, stop_ids=[self.eos_token_id],
                                   tokenizer_id="original-course-word-vocabulary", tokenizer_revision=None)


def last_valid_indices(mask):
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] < 1 or not mask.any(1).all():
        raise ValueError("Boolean mask must have at least one valid position per row")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(1).values


def _pad(sequences, unknowns, *, padding, max_positions, step_targets=None):
    if padding not in {"left", "right"} or not sequences:
        raise ValueError("Nonempty batch and explicit left/right padding required")
    if any(not row for row in sequences) or max(map(len, sequences)) > max_positions:
        raise ValueError("Empty or overlength sequence; no silent truncation")
    width = max(map(len, sequences))
    ids = torch.zeros(len(sequences), width, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    targets = torch.full(ids.shape, -100., dtype=torch.float64)
    for i, row in enumerate(sequences):
        start = 0 if padding == "right" else width - len(row)
        ids[i, start:start + len(row)] = torch.tensor(row)
        mask[i, start:start + len(row)] = True
        if step_targets is not None:
            for position, value in step_targets[i]:
                targets[i, start + position] = float(value)
    return {"ids": ids, "mask": mask, "endpoints": last_valid_indices(mask),
            "unknown_tokens": unknowns, "step_targets": targets}


def text_batch(vocabulary, texts, *, padding="right", include_eos=True, max_positions=96):
    sequences, unknowns = [], []
    for prompt, completion in texts:
        x, unknown_x = vocabulary.encode(prompt)
        y, unknown_y = vocabulary.encode(completion)
        sequences.append(x + [2] + y + ([4] if include_eos else []))
        unknowns.append(unknown_x + unknown_y)
    return _pad(sequences, unknowns, padding=padding, max_positions=max_positions)


def process_batch(vocabulary, records, *, padding="right", include_eos=True, max_positions=96):
    sequences, unknowns, labels = [], [], []
    for row in records:
        x, unknown = vocabulary.encode(row["prompt"])
        sequence, boundaries = x + [2], []
        for step in row["steps"]:
            ids, missing = vocabulary.encode(step["text"])
            sequence += ids + [3]
            unknown += missing
            boundaries.append((len(sequence) - 1, int(step["valid"])))
        final, missing = vocabulary.encode(row["final"])
        sequence += final + ([4] if include_eos else [])
        sequences.append(sequence)
        unknowns.append(unknown + missing)
        labels.append(boundaries)
    return _pad(sequences, unknowns, padding=padding, max_positions=max_positions, step_targets=labels)


@dataclass(frozen=True)
class RewardConfig:
    vocab_size: int
    width: int = 24
    heads: int = 2
    max_positions: int = 96
    include_eos: bool = True

    def __post_init__(self):
        if any(type(v) is not int for v in (self.vocab_size, self.width, self.heads, self.max_positions)):
            raise ValueError("Model dimensions must be integers")
        if not 5 <= self.vocab_size <= 10_000 or not 4 <= self.width <= 128 or not 1 <= self.heads <= self.width:
            raise ValueError("Model dimensions outside tiny CPU limits")
        if self.width % self.heads or not 8 <= self.max_positions <= 256 or type(self.include_eos) is not bool:
            raise ValueError("Invalid head geometry, position cap or EOS contract")


class TinyTextReward(nn.Module):
    """One causal decoder block and scalar head; dropout disabled, float64 CPU."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.width
        self.embedding = nn.Embedding(config.vocab_size, d, padding_idx=0)
        self.position = nn.Embedding(config.max_positions, d)
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.output = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.final_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.double().cpu()

    def hidden(self, ids, mask):
        if ids.ndim != 2 or ids.shape != mask.shape or ids.dtype != torch.long:
            raise ValueError("Token IDs must be a two-dimensional long tensor matching the mask")
        if ids.device.type != "cpu" or mask.device.type != "cpu":
            raise ValueError("This microscope is CPU-only")
        last_valid_indices(mask)
        if ids.shape[1] > self.config.max_positions or (ids < 0).any() or (ids >= self.config.vocab_size).any():
            raise ValueError("Token IDs or length outside frozen model bounds")
        if (ids[~mask] != 0).any():
            raise ValueError("Invalid positions must contain the declared pad token")
        b, t = ids.shape
        d, h = self.config.width, self.config.heads
        # Actual token positions, not array indices: left/right padding agree.
        positions = (mask.long().cumsum(1) - 1).clamp_min(0)
        x = (self.embedding(ids) + self.position(positions)) * mask.unsqueeze(-1)
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q, k, v = [tensor.reshape(b, t, h, d // h).transpose(1, 2) for tensor in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(d // h)
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        allowed = causal[None, None] & mask[:, None, None, :] & mask[:, None, :, None]
        # Finite sentinel keeps all-masked padded queries from producing NaNs;
        # zeroing their outputs ensures they contribute no content or gradient.
        attention = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min).softmax(-1)
        context = (attention @ (v * mask[:, None, :, None])).transpose(1, 2).reshape(b, t, d)
        x = (x + self.output(context)) * mask.unsqueeze(-1)
        x = (x + self.mlp(self.norm2(x))) * mask.unsqueeze(-1)
        return self.final_norm(x) * mask.unsqueeze(-1)

    def token_scores(self, ids, mask):
        return self.head(self.hidden(ids, mask)).squeeze(-1)

    def forward(self, ids, mask):
        scores = self.token_scores(ids, mask)
        return scores.gather(1, last_valid_indices(mask)[:, None]).squeeze(1)


def validate_fixture(obj):
    if obj.get("schema") != SCHEMA:
        raise ValueError("Unknown text-reward schema")
    splits, seen = {}, set()
    for kind, rows in (("pair", obj["pairs"]), ("trace", obj["processes"])):
        if not rows:
            raise ValueError("Empty fixture branch")
        for row in rows:
            identifier = row["pair_id"] if kind == "pair" else row["record_id"]
            if not identifier or identifier in seen:
                raise ValueError("Duplicate or missing record ID")
            seen.add(identifier)
            group, split = row["source_group_id"], row["split"]
            if not group or split not in {"train", "calibration", "test"} or splits.setdefault(group, split) != split:
                raise ValueError("Source-group split collision or invalid identity")
            if row["provenance"] != "authored":
                raise ValueError("This original fixture requires explicit authored provenance")
            TextVocabulary.pieces(row["prompt"])
            if kind == "pair":
                if type(row["q_left"]) is not int or row["q_left"] not in (0, 1):
                    raise ValueError("Original pair labels are binary authored choices")
                TextVocabulary.pieces(row["left"])
                TextVocabulary.pieces(row["right"])
            else:
                if type(row["outcome"]) is not int or row["outcome"] not in (0, 1) or not row["steps"]:
                    raise ValueError("Trace requires binary terminal label and explicit steps")
                TextVocabulary.pieces(row["final"])
                for step in row["steps"]:
                    TextVocabulary.pieces(step["text"])
                    if type(step["valid"]) is not bool or not step["justification"]:
                        raise ValueError("Step needs authored validity and justification")
    return obj


def load_fixture(path):
    return validate_fixture(json.loads(Path(path).read_text()))


def fixture_vocabulary(obj):
    texts = []
    for row in obj["pairs"]:
        if row["split"] == "train":
            texts += [row["prompt"], row["left"], row["right"]]
    for row in obj["processes"]:
        if row["split"] == "train":
            texts += [row["prompt"], row["final"], *(step["text"] for step in row["steps"])]
    # Declared formatting-control alphabet, independent of held-out labels.
    return TextVocabulary.fit(texts + ["Answer : ** In summary , this is a clear answer ."])


def nuisance_pairs(rows):
    result = []
    for row in rows:
        result.append(dict(row))
        for condition in ("matched-heading", "longer-incorrect", "same-substance-longer", "same-substance-heading"):
            changed = dict(row, pair_id=row["pair_id"] + "-" + condition, condition=condition)
            if condition == "matched-heading":
                changed["left"], changed["right"] = "** Answer : " + row["left"] + " **", "** Answer : " + row["right"] + " **"
            elif condition == "longer-incorrect":
                wrong = "right" if row["q_left"] else "left"
                changed[wrong] += " In summary , this is a clear answer ."
            else:
                correct = row["left"] if row["q_left"] else row["right"]
                changed["left"], changed["right"], changed["q_left"] = correct, correct, .5
                changed["right"] = correct + " In summary , this is a clear answer ." if condition.endswith("longer") else "** Answer : " + correct + " **"
            result.append(changed)
    return result


def probability_metrics(probabilities, targets, bins=4):
    p = torch.as_tensor(probabilities, dtype=torch.float64)
    q = torch.as_tensor(targets, dtype=torch.float64)
    if p.ndim != 1 or p.shape != q.shape or not len(p) or not torch.isfinite(p).all() or not torch.isfinite(q).all():
        raise ValueError("Probability/label vectors must match and be finite")
    if ((p < 0) | (p > 1) | (q < 0) | (q > 1)).any() or type(bins) is not int or bins < 1:
        raise ValueError("Invalid probabilities or bins")
    clipped = p.clamp(1e-12, 1 - 1e-12)
    reliability, ece = [], 0.
    for index in range(bins):
        valid = (p >= index / bins) & (p <= 1 if index == bins - 1 else p < (index + 1) / bins)
        if valid.any():
            count = int(valid.sum())
            prediction, observed = float(p[valid].mean()), float(q[valid].mean())
            reliability.append({"count": count, "predicted": prediction, "observed": observed})
            ece += count / len(p) * abs(prediction - observed)
    decisive = q != .5
    return {"n": len(p), "decisive_n": int(decisive.sum()),
            "ranking_accuracy": float(((p[decisive] > .5) == (q[decisive] > .5)).double().mean()) if decisive.any() else None,
            "nll": float(-(q * clipped.log() + (1 - q) * (1 - clipped).log()).mean()),
            "brier": float((q * (1 - p).square() + (1 - q) * p.square()).mean()),
            "brier_convention": "Observed binary score for hard labels; expected Bernoulli(q) score for declared soft fixture targets",
            "ece": ece, "reliability": reliability}


def group_mean_loss(losses, groups):
    if losses.ndim != 1 or len(losses) != len(groups) or not len(groups):
        raise ValueError("One loss per declared source observation required")
    unique = sorted(set(groups))
    return torch.stack([losses[torch.tensor([g == group for g in groups])].mean() for group in unique]).mean()


def fit_text_model(obj, vocabulary, *, seed=1601, objective="preference", steps=None):
    if objective not in {"preference", "outcome", "process"}:
        raise ValueError("Unknown text reward objective")
    steps = (120 if objective == "preference" else 80) if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= 400:
        raise ValueError("Fitting budget outside bounded CPU recipe")
    rows = [row for row in obj["pairs" if objective == "preference" else "processes"] if row["split"] == "train"]
    config = RewardConfig(len(vocabulary.tokens))
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = TinyTextReward(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
    if objective == "preference":
        batch = text_batch(vocabulary, [(r["prompt"], r[side]) for side in ("left", "right") for r in rows])
        labels = torch.tensor([r["q_left"] for r in rows], dtype=torch.float64)
    else:
        batch = process_batch(vocabulary, rows)
        labels = torch.tensor([r["outcome"] for r in rows], dtype=torch.float64)
    history, gradient_evidence = [], None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if objective == "preference":
            scores = model(batch["ids"], batch["mask"])
            logits = scores[:len(rows)] - scores[len(rows):]
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = group_mean_loss(losses, [r["source_group_id"] for r in rows])
        elif objective == "outcome":
            logits = model(batch["ids"], batch["mask"])
            losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
            loss = group_mean_loss(losses, [r["source_group_id"] for r in rows])
        else:
            scores = model.token_scores(batch["ids"], batch["mask"])
            supervised = batch["step_targets"] != -100
            losses = F.binary_cross_entropy_with_logits(scores[supervised], batch["step_targets"][supervised], reduction="none")
            groups = [r["source_group_id"] for r, mask in zip(rows, supervised) for _ in range(int(mask.sum()))]
            loss = group_mean_loss(losses, groups)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite text reward objective")
        loss.backward()
        if step == 0:
            gradient_evidence = {"embedding_norm": float(model.embedding.weight.grad.norm()),
                                 "attention_qkv_norm": float(model.qkv.weight.grad.norm()),
                                 "head_norm": float(model.head.weight.grad.norm())}
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Nonfinite reward gradients")
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    return model, {"seed": seed, "objective": objective, "steps": steps,
                   "history": history, "first_backward": gradient_evidence,
                   "train_record_ids": [r.get("pair_id", r.get("record_id")) for r in rows]}


@torch.no_grad()
def evaluate_pairs(model, vocabulary, rows, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    batch = text_batch(vocabulary, [(r["prompt"], r[side]) for side in ("left", "right") for r in rows], include_eos=model.config.include_eos)
    scores = model(batch["ids"], batch["mask"])
    margins = scores[:len(rows)] - scores[len(rows):]
    p = (margins / temperature).sigmoid()
    predictions = [{**r, "margin": float(margin), "p_left": float(probability),
                    "left_oov": batch["unknown_tokens"][i], "right_oov": batch["unknown_tokens"][len(rows) + i]}
                   for i, (r, margin, probability) in enumerate(zip(rows, margins, p))]
    conditions = sorted({r["condition"] for r in rows})
    return {"temperature": temperature, "predictions": predictions,
            "slices": {condition: probability_metrics([r["p_left"] for r in predictions if r["condition"] == condition],
                                                       [r["q_left"] for r in predictions if r["condition"] == condition])
                       for condition in conditions}}


@torch.no_grad()
def evaluate_traces(outcome_model, process_model, vocabulary, rows):
    batch = process_batch(vocabulary, rows)
    terminal = outcome_model(batch["ids"], batch["mask"]).sigmoid()
    step_scores = process_model.token_scores(batch["ids"], batch["mask"]).sigmoid()
    predictions, probabilities, targets = [], [], []
    for i, row in enumerate(rows):
        positions = (batch["step_targets"][i] != -100).nonzero().flatten()
        ps = step_scores[i, positions].tolist()
        qs = batch["step_targets"][i, positions].tolist()
        predictions.append({**row, "outcome_probability": float(terminal[i]),
                            "step_probabilities": ps, "step_labels": qs,
                            "step_positions": positions.tolist(), "oov": batch["unknown_tokens"][i]})
        probabilities += ps
        targets += qs
    return {"predictions": predictions,
            "outcome_metrics": probability_metrics(terminal.tolist(), [r["outcome"] for r in rows]),
            "process_metrics": probability_metrics(probabilities, targets)}


def export_frozen(model, vocabulary, path, *, fixture_path, source_groups, seed, objective="preference"):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new frozen reward export; preserve earlier evidence")
    if objective != "preference":
        raise ValueError("Linked policy export requires a preference reward, not a process head")
    body = {"schema": SCHEMA, "dtype": "float64", "objective": objective,
            "config": asdict(model.config), "tokens": vocabulary.tokens,
            "unknown_policy": vocabulary.unknown_policy,
            "checkpoint_interface": vocabulary.interface(model.config.include_eos),
            "endpoint_semantics": "final nonpadding EOS" if model.config.include_eos else "final completion token; EOS excluded",
            "seed": seed, "selection": f"predeclared seed{seed}; no held-out selection",
            "fixture_sha256": file_digest(fixture_path), "source_groups": source_groups,
            "source_sha256": file_digest(Path(__file__)),
            "state": {key: tensor.detach().cpu().tolist() for key, tensor in model.state_dict().items()}}
    body["payload_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return {"path": str(path), "file_sha256": file_digest(path), "payload_sha256": body["payload_sha256"]}


class FrozenTextReward:
    def __init__(self, model, vocabulary, identity):
        self.model, self.vocabulary, self.identity = model.eval().requires_grad_(False), vocabulary, identity

    @torch.no_grad()
    def score_many(self, texts):
        batch = text_batch(self.vocabulary, texts, include_eos=self.model.config.include_eos,
                           max_positions=self.model.config.max_positions)
        return self.model(batch["ids"], batch["mask"]).detach().cpu()

    def score(self, prompt, completion):
        return float(self.score_many([(prompt, completion)])[0])


def load_frozen(path, *, expected_interface=None):
    path = Path(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Frozen tiny reward file exceeds bounded JSON loader")
    body = json.loads(path.read_text())
    if not isinstance(body, dict):
        raise ValueError("Frozen reward must be a JSON object")
    digest = body.pop("payload_sha256", None)
    if body.get("schema") != SCHEMA or digest != canonical_hash(body) or body.get("dtype") != "float64":
        raise ValueError("Frozen reward schema, dtype or payload hash mismatch")
    required = {"schema", "dtype", "objective", "config", "tokens", "unknown_policy",
                "checkpoint_interface", "endpoint_semantics", "seed", "selection",
                "fixture_sha256", "source_groups", "source_sha256", "state"}
    if set(body) != required or body["objective"] != "preference":
        raise ValueError("Frozen preference payload is incomplete or has unexpected fields/objective")
    if any(not isinstance(body[key], str) or not re.fullmatch(r"[0-9a-f]{64}", body[key])
           for key in ("fixture_sha256", "source_sha256")):
        raise ValueError("Invalid fixture/source hash identity")
    groups = body["source_groups"]
    if not isinstance(groups, dict) or set(groups) != {"train", "calibration", "test"}:
        raise ValueError("Frozen reward source splits are missing")
    if any(not isinstance(split, list) or not split
           or any(not isinstance(group, str) or not group.strip() for group in split)
           or len(set(split)) != len(split) for split in groups.values()):
        raise ValueError("Frozen source splits must contain nonempty lists of unique nonempty strings")
    flat = [group for split in groups.values() for group in split]
    if len(set(flat)) != len(flat) or any(not isinstance(group, str) or not group for group in flat):
        raise ValueError("Frozen source groups collide or lack identity")
    config = RewardConfig(**body["config"])
    vocabulary = TextVocabulary(body["tokens"], body["unknown_policy"])
    if len(vocabulary.tokens) != config.vocab_size:
        raise ValueError("Vocabulary geometry differs from model config")
    actual = vocabulary.interface(config.include_eos)
    assert_compatible(body["checkpoint_interface"], actual)
    if expected_interface is not None:
        assert_compatible(expected_interface, actual)
    expected_endpoint = "final nonpadding EOS" if config.include_eos else "final completion token; EOS excluded"
    if body["endpoint_semantics"] != expected_endpoint:
        raise ValueError("Endpoint convention differs from encoding interface")
    with torch.random.fork_rng():
        model = TinyTextReward(config)
    expected = model.state_dict()
    if set(body["state"]) != set(expected):
        raise ValueError("Frozen reward state keys are incomplete or unexpected")
    restored = {}
    for key, values in body["state"].items():
        try:
            tensor = torch.tensor(values, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Malformed numeric reward state") from error
        if tensor.shape != expected[key].shape or not torch.isfinite(tensor).all():
            raise ValueError("Frozen reward state shape or numeric value mismatch")
        restored[key] = tensor
    model.load_state_dict(restored, strict=True)
    return FrozenTextReward(model, vocabulary, {**body, "payload_sha256": digest, "file_sha256": file_digest(path)})


def run_reference(fixture_path, export_path=None):
    obj = load_fixture(fixture_path)
    vocabulary = fixture_vocabulary(obj)
    result = {"schema": SCHEMA, "scope": "Original authored text/labels; tiny CPU-trained reward models, not human/pretrained evidence",
              "fixture_sha256": file_digest(fixture_path), "vocabulary": vocabulary.tokens,
              "unknown_policy": vocabulary.unknown_policy, "seeds": [],
              "source_splits": {split: sorted({r["source_group_id"] for rows in (obj["pairs"], obj["processes"])
                                               for r in rows if r["split"] == split}) for split in ("train", "calibration", "test")}}
    calibration_rows = [r for r in obj["pairs"] if r["split"] == "calibration"]
    test_rows = nuisance_pairs([r for r in obj["pairs"] if r["split"] == "test"])
    for seed in (1601, 1602, 1603):
        start = time.perf_counter()
        preference, preference_fit = fit_text_model(obj, vocabulary, seed=seed)
        calibration = {tau: evaluate_pairs(preference, vocabulary, calibration_rows, tau) for tau in (.5, 1., 2., 4.)}
        temperature = min(calibration, key=lambda tau: calibration[tau]["slices"]["matched-length-format"]["nll"])
        outcome, outcome_fit = fit_text_model(obj, vocabulary, seed=seed, objective="outcome")
        process, process_fit = fit_text_model(obj, vocabulary, seed=seed, objective="process")
        row = {"seed": seed, "preference_fit": preference_fit, "outcome_fit": outcome_fit, "process_fit": process_fit,
               "temperature_selection": {"calibration_only": True, "selected": temperature,
                    "grid_nll": {str(tau): calibration[tau]["slices"]["matched-length-format"]["nll"] for tau in calibration}},
               "calibration": calibration[temperature], "test_raw": evaluate_pairs(preference, vocabulary, test_rows),
               "test_calibrated": evaluate_pairs(preference, vocabulary, test_rows, temperature),
               "traces": {split: evaluate_traces(outcome, process, vocabulary, [r for r in obj["processes"] if r["split"] == split])
                          for split in ("train", "calibration", "test")}, "seconds": time.perf_counter() - start}
        if seed == 1601 and export_path is not None:
            row["frozen_export"] = export_frozen(preference, vocabulary, export_path, fixture_path=fixture_path,
                source_groups=result["source_splits"], seed=seed)
            frozen = load_frozen(export_path, expected_interface=vocabulary.interface())
            texts = [(r["prompt"], r["left"]) for r in calibration_rows]
            batch = text_batch(vocabulary, texts)
            with torch.no_grad():
                expected = preference(batch["ids"], batch["mask"])
            row["frozen_reload_exact"] = torch.equal(expected, frozen.score_many(texts))
            row["frozen_parameters_have_gradients_enabled"] = any(p.requires_grad for p in frozen.model.parameters())
        result["seeds"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frozen-export", type=Path)
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError("Choose a new experiment output; retain prior results")
    torch.set_num_threads(1)
    result = run_reference(args.fixture, args.frozen_export)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(serialized)
        print(json.dumps({"output": str(args.output), "seeds": [r["seed"] for r in result["seeds"]],
                          "frozen_export": str(args.frozen_export)}))
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
