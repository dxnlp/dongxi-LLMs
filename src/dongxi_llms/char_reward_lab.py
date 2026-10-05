"""Encoding-disjoint ASCII text reward intervention; original bounded CPU code.

This additive arm reuses the actual causal decoder, not known quality features.
Historical word-token artifacts are never rewritten. No work runs on import.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import re
import sys
import time

import torch
from torch.nn import functional as F

from .run_identity import canonical_hash, file_digest, tokenizer_interface, assert_compatible
from .text_reward_lab import (
    SPECIALS, TextVocabulary, RewardConfig, TinyTextReward, FrozenTextReward,
    text_batch as word_text_batch, process_batch as word_process_batch,
    validate_fixture, load_fixture, nuisance_pairs, probability_metrics,
    group_mean_loss,
)

SCHEMA = "dongxi-char-text-reward-v1"
MAX_POSITIONS = 160
SEEDS = (1611, 1612, 1613)
TEMPERATURES = (.5, 1., 2., 4.)
ALPHABET = tuple(chr(i) for i in range(32, 127))
TOKENS = (*SPECIALS, *ALPHABET)
PROTOCOL = "experiments/specs/2026-10-04-text-reward-vocabulary-intervention.md"


class ASCIICharacterVocabulary(TextVocabulary):
    """Fixed printable ASCII alphabet; no fitted vocabulary or unknown fallback."""
    def __init__(self, tokens=None, unknown_policy="error"):
        if unknown_policy != "error" or (tokens is not None and list(tokens) != list(TOKENS)):
            raise ValueError("Character contract requires the exact fixed alphabet and explicit rejection")
        super().__init__(list(TOKENS), unknown_policy="error")

    @staticmethod
    def pieces(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Character input must be nonblank text")
        if any(not 32 <= ord(char) <= 126 for char in text):
            raise ValueError("Unsupported character: only printable ASCII is accepted before casefolding")
        return list(text.casefold())

    @classmethod
    def fit(cls, texts):
        raise ValueError("Fixed character vocabulary is not trained or refitted")

    def encode(self, text):
        return [self.mapping[char] for char in self.pieces(text)], []

    def to_str(self):
        return json.dumps({"model": {"type": "OriginalFixedASCIICharacters", "vocab": self.mapping},
                           "normalizer": "validate-printable-ASCII-before-casefold",
                           "pre_tokenizer": "one-character-including-space",
                           "unknown_policy": "reject", "literal_special_text": "characters-only",
                           "special_tokens": dict(zip(SPECIALS, range(5)))}, sort_keys=True)

    def interface(self, include_eos=True):
        return tokenizer_interface(self,
            template="ASCII(prompt) + SEP + ASCII(completion)" + (" + EOS" if include_eos else "; EOS excluded"),
            stop_ids=[4], tokenizer_id="original-course-fixed-printable-ASCII", tokenizer_revision=None)


def char_config(*, include_eos=True):
    return RewardConfig(len(TOKENS), max_positions=MAX_POSITIONS, include_eos=include_eos)


def char_text_batch(vocabulary, texts, *, padding="right", include_eos=True, max_positions=MAX_POSITIONS):
    if not isinstance(vocabulary, ASCIICharacterVocabulary):
        raise ValueError("Character collator requires the fixed character interface")
    return word_text_batch(vocabulary, texts, padding=padding,
                           include_eos=include_eos, max_positions=max_positions)


def char_process_batch(vocabulary, rows, *, padding="right", include_eos=True, max_positions=MAX_POSITIONS):
    if not isinstance(vocabulary, ASCIICharacterVocabulary):
        raise ValueError("Character process collator requires the fixed character interface")
    return word_process_batch(vocabulary, rows, padding=padding,
                              include_eos=include_eos, max_positions=max_positions)


def _valid_ids(batch, index):
    return tuple(batch["ids"][index, batch["mask"][index]].tolist())


def encoding_audit(obj, vocabulary=None):
    """Audit complete inputs and supervised causal prefixes before any fitting.

    Same-source repeated prefixes are expected. Arbitrary short lexical prefixes
    are not leakage; this checks the entire prompt/response or STEP endpoint.
    Every collision retains identities and labels, including candidate swaps.
    """
    vocabulary = vocabulary or ASCIICharacterVocabulary()
    records, issues, owners, lengths = [], [], {}, {"pair": [], "trace": []}
    try:
        validate_fixture(obj)
    except (ValueError, TypeError, KeyError) as error:
        return {"passed": False, "issues": [{"kind": "raw-source-contract", "error": repr(error)}],
                "records": [], "maximum_positions": MAX_POSITIONS}

    def register(kind, signature, row, label=None):
        key = kind, canonical_hash(signature)
        identity = {"record_id": row.get("pair_id", row.get("record_id")),
                    "source_group_id": row["source_group_id"], "split": row["split"], "label": label}
        prior = owners.setdefault(key, [])
        for old in prior:
            if old["source_group_id"] != identity["source_group_id"] or old["split"] != identity["split"]:
                issues.append({"kind": kind + "-source-or-split-collision", "encoding_sha256": key[1],
                               "first": old, "second": identity})
            elif label is not None and old["label"] != label:
                issues.append({"kind": kind + "-conflicting-label", "encoding_sha256": key[1],
                               "first": old, "second": identity})
        prior.append(identity)
        return key[1]

    # Audit the baseline and exact evaluation perturbations, not newly invented sources.
    pairs = [r for r in obj["pairs"] if r["split"] != "test"]
    pairs += nuisance_pairs([r for r in obj["pairs"] if r["split"] == "test"])
    for row in pairs:
        try:
            batch = char_text_batch(vocabulary, [(row["prompt"], row["left"]), (row["prompt"], row["right"])])
            left, right = _valid_ids(batch, 0), _valid_ids(batch, 1)
            prompt = tuple(vocabulary.encode(row["prompt"])[0])
            ordered = (left, right) if left <= right else (right, left)
            label = row["q_left"] if left <= right else 1 - row["q_left"]
            pair_hash = register("unordered-pair", ordered, row, label)
            prompt_hash = register("full-prompt", prompt, row)
            left_hash, right_hash = [register("prompt-completion", ids, row) for ids in (left, right)]
            lengths["pair"] += [len(left), len(right)]
            records.append({"kind": "pair", "record_id": row["pair_id"], "source_group_id": row["source_group_id"],
                            "split": row["split"], "condition": row["condition"], "q_left": row["q_left"],
                            "prompt_sha256": prompt_hash, "unordered_pair_sha256": pair_hash,
                            "left_input_sha256": left_hash, "right_input_sha256": right_hash,
                            "left_positions": len(left), "right_positions": len(right)})
        except (ValueError, TypeError, KeyError) as error:
            issues.append({"kind": "pair-encoding", "record_id": row["pair_id"], "error": repr(error)})
    for row in obj["processes"]:
        try:
            batch = char_process_batch(vocabulary, [row])
            ids = _valid_ids(batch, 0)
            terminal_hash = register("terminal-trace", ids, row, row["outcome"])
            prompt_hash = register("full-prompt", tuple(vocabulary.encode(row["prompt"])[0]), row)
            boundaries = (batch["step_targets"][0] != -100).nonzero().flatten().tolist()
            prefixes = [register("step-prefix", ids[:position + 1], row,
                                 int(batch["step_targets"][0, position])) for position in boundaries]
            lengths["trace"].append(len(ids))
            records.append({"kind": "trace", "record_id": row["record_id"], "source_group_id": row["source_group_id"],
                            "split": row["split"], "positions": len(ids), "prompt_sha256": prompt_hash,
                            "terminal_input_sha256": terminal_hash, "step_positions": boundaries,
                            "step_prefix_sha256": prefixes, "step_labels": [int(s["valid"]) for s in row["steps"]]})
        except (ValueError, TypeError, KeyError) as error:
            issues.append({"kind": "trace-encoding", "record_id": row["record_id"], "error": repr(error)})
    counts = {split: {"pairs": len({r["unordered_pair_sha256"] for r in records
                                   if r["kind"] == "pair" and r["split"] == split}),
                      "step_prefixes": len({p for r in records if r["kind"] == "trace" and r["split"] == split
                                            for p in r["step_prefix_sha256"]})}
              for split in ("train", "calibration", "test")}
    return {"passed": not issues, "issues": issues, "records": records,
            "maximum_positions": MAX_POSITIONS,
            "maximum_pair_positions": max(lengths["pair"], default=0),
            "maximum_trace_positions": max(lengths["trace"], default=0),
            "unique_encoded_counts": counts,
            "source_splits": {split: sorted({r["source_group_id"] for branch in (obj["pairs"], obj["processes"])
                                              for r in branch if r["split"] == split})
                              for split in ("train", "calibration", "test")}}


class FitFailure(RuntimeError):
    def __init__(self, error, evidence):
        super().__init__(str(error))
        self.evidence = evidence


def fit_char_model(obj, vocabulary=None, *, seed=1611, objective="preference", steps=None):
    """Train the real shared text decoder/head on training records only."""
    vocabulary = vocabulary or ASCIICharacterVocabulary()
    audit = encoding_audit(obj, vocabulary)
    if not audit["passed"]:
        raise ValueError("Encoded source/prefix gate failed: " + json.dumps(audit["issues"]))
    if objective not in {"preference", "outcome", "process"}:
        raise ValueError("Unknown character reward objective")
    steps = (120 if objective == "preference" else 80) if steps is None else steps
    if type(seed) is not int or type(steps) is not int or not 1 <= steps <= 400:
        raise ValueError("Invalid seed or bounded fitting budget")
    rows = [r for r in obj["pairs" if objective == "preference" else "processes"] if r["split"] == "train"]
    if not rows:
        raise ValueError("Character objective has no training rows")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = TinyTextReward(char_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.02, weight_decay=.01)
    if objective == "preference":
        batch = char_text_batch(vocabulary, [(r["prompt"], r[side]) for side in ("left", "right") for r in rows])
    else:
        batch = char_process_batch(vocabulary, rows)
    labels = torch.tensor([r["q_left"] if objective == "preference" else r["outcome"] for r in rows], dtype=torch.float64)
    evidence = {"seed": seed, "objective": objective, "planned_steps": steps, "history": [],
                "first_backward": None, "train_record_ids": [r.get("pair_id", r.get("record_id")) for r in rows]}
    try:
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            if objective == "process":
                logits = model.token_scores(batch["ids"], batch["mask"])
                selected = batch["step_targets"] != -100
                losses = F.binary_cross_entropy_with_logits(logits[selected], batch["step_targets"][selected], reduction="none")
                groups = [r["source_group_id"] for r, mask in zip(rows, selected) for _ in range(int(mask.sum()))]
            else:
                scores = model(batch["ids"], batch["mask"])
                logits = scores[:len(rows)] - scores[len(rows):] if objective == "preference" else scores
                losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                groups = [r["source_group_id"] for r in rows]
            loss = group_mean_loss(losses, groups)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite character reward loss")
            loss.backward()
            if step == 0:
                evidence["first_backward"] = {"embedding_norm": float(model.embedding.weight.grad.norm()),
                    "attention_qkv_norm": float(model.qkv.weight.grad.norm()), "head_norm": float(model.head.weight.grad.norm())}
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite character reward gradient")
            optimizer.step()
            evidence["history"].append(float(loss.detach()))
    except Exception as error:
        raise FitFailure(error, evidence) from error
    evidence["completed_steps"] = len(evidence["history"])
    return model.eval(), evidence


@torch.no_grad()
def evaluate_char_pairs(model, vocabulary, rows, temperature=1.):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    batch = char_text_batch(vocabulary, [(r["prompt"], r[side]) for side in ("left", "right") for r in rows],
                            include_eos=model.config.include_eos, max_positions=model.config.max_positions)
    scores = model(batch["ids"], batch["mask"])
    margins = scores[:len(rows)] - scores[len(rows):]
    probabilities = (margins / temperature).sigmoid()
    predictions = [{**r, "margin": float(m), "p_left": float(p),
                    "left_input_sha256": canonical_hash(_valid_ids(batch, i)),
                    "right_input_sha256": canonical_hash(_valid_ids(batch, i + len(rows))),
                    "left_oov": batch["unknown_tokens"][i], "right_oov": batch["unknown_tokens"][i + len(rows)]}
                   for i, (r, m, p) in enumerate(zip(rows, margins, probabilities))]
    return {"temperature": temperature, "predictions": predictions,
            "slices": {c: probability_metrics([r["p_left"] for r in predictions if r["condition"] == c],
                                               [r["q_left"] for r in predictions if r["condition"] == c])
                       for c in sorted({r["condition"] for r in rows})}}


@torch.no_grad()
def evaluate_char_traces(outcome_model, process_model, vocabulary, rows):
    batch = char_process_batch(vocabulary, rows)
    terminal = outcome_model(batch["ids"], batch["mask"]).sigmoid()
    steps = process_model.token_scores(batch["ids"], batch["mask"]).sigmoid()
    predictions, ps, qs = [], [], []
    for i, r in enumerate(rows):
        positions = (batch["step_targets"][i] != -100).nonzero().flatten()
        p, q = steps[i, positions].tolist(), batch["step_targets"][i, positions].tolist()
        ids = _valid_ids(batch, i)
        predictions.append({**r, "outcome_probability": float(terminal[i]), "step_probabilities": p,
                            "step_labels": q, "step_positions": positions.tolist(),
                            "terminal_input_sha256": canonical_hash(ids),
                            "step_prefix_sha256": [canonical_hash(ids[:int(pos) + 1]) for pos in positions],
                            "oov": batch["unknown_tokens"][i]})
        ps += p
        qs += q
    return {"predictions": predictions,
            "outcome_metrics": probability_metrics(terminal.tolist(), [r["outcome"] for r in rows]),
            "process_metrics": probability_metrics(ps, qs)}


def export_char_reward(model, vocabulary, path, *, fixture_path, source_groups, seed, protocol_path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("Choose a new character export; preserve historical evidence")
    if (not isinstance(vocabulary, ASCIICharacterVocabulary) or model.config.vocab_size != len(TOKENS)
            or model.config.max_positions != MAX_POSITIONS or type(seed) is not int):
        raise ValueError("Character export needs exact token meanings and geometry")
    if any(v.dtype != torch.float64 or v.device.type != "cpu" or not torch.isfinite(v).all()
           for v in model.state_dict().values()):
        raise ValueError("Character export requires finite float64 CPU weights")
    audit = encoding_audit(load_fixture(fixture_path), vocabulary)
    if not audit["passed"] or source_groups != audit["source_splits"]:
        raise ValueError("Character export source/encoding contract differs from fixture")
    body = {"schema": SCHEMA, "dtype": "float64", "objective": "preference",
            "config": asdict(model.config), "tokens": list(vocabulary.tokens), "unknown_policy": "error",
            "checkpoint_interface": vocabulary.interface(model.config.include_eos),
            "endpoint_semantics": "final nonpadding EOS" if model.config.include_eos else "final completion token; EOS excluded",
            "seed": seed, "selection": f"predeclared seed {seed}; no heldout selection",
            "fixture_sha256": file_digest(fixture_path), "protocol_sha256": file_digest(protocol_path),
            "source_groups": source_groups, "encoding_audit_sha256": canonical_hash(audit),
            "source_sha256": {"char_reward_lab.py": file_digest(Path(__file__)),
                              "text_reward_lab.py": file_digest(Path(__file__).with_name("text_reward_lab.py")),
                              "run_identity.py": file_digest(Path(__file__).with_name("run_identity.py"))},
            "state": {key: value.detach().cpu().tolist() for key, value in model.state_dict().items()}}
    body["payload_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"path": str(path), "file_sha256": file_digest(path), "payload_sha256": body["payload_sha256"]}


def _numeric_tree(value):
    if isinstance(value, list):
        return all(_numeric_tree(v) for v in value)
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate character reward JSON field")
        result[key] = value
    return result


def _bounded_json_structure(body):
    stack, nodes = [(body, 0)], 0
    while stack:
        value, depth = stack.pop()
        nodes += 1
        if depth > 32 or nodes > 100_000:
            raise ValueError("Character reward JSON nesting or element bound exceeded")
        if isinstance(value, dict):
            stack.extend((v, depth + 1) for v in value.values())
        elif isinstance(value, list):
            stack.extend((v, depth + 1) for v in value)


def load_char_reward(path, *, expected_interface=None, expected_file_sha256=None):
    path = Path(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Character reward exceeds bounded JSON loader")
    file_hash = file_digest(path)
    if expected_file_sha256 is not None and file_hash != expected_file_sha256:
        raise ValueError("Character reward file differs from expected external identity")
    try:
        body = json.loads(path.read_text(), object_pairs_hook=_unique_fields)
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid bounded character reward JSON") from error
    if not isinstance(body, dict):
        raise ValueError("Character reward must be an object")
    _bounded_json_structure(body)
    digest = body.pop("payload_sha256", None)
    fields = {"schema", "dtype", "objective", "config", "tokens", "unknown_policy", "checkpoint_interface",
              "endpoint_semantics", "seed", "selection", "fixture_sha256", "protocol_sha256",
              "source_groups", "encoding_audit_sha256", "source_sha256", "state"}
    if set(body) != fields or body["schema"] != SCHEMA or body["dtype"] != "float64" or body["objective"] != "preference":
        raise ValueError("Unknown or incomplete character reward schema")
    if canonical_hash(body) != digest:
        raise ValueError("Character reward payload hash mismatch")
    groups = body["source_groups"]
    if not isinstance(groups, dict) or set(groups) != {"train", "calibration", "test"}:
        raise ValueError("Character reward source splits are missing")
    if any(not isinstance(v, list) or not v or any(not isinstance(g, str) or not g.strip() for g in v)
           or len(set(v)) != len(v) for v in groups.values()):
        raise ValueError("Character source splits require unique nonblank IDs in nonempty lists")
    flat = [g for values in groups.values() for g in values]
    if len(set(flat)) != len(flat):
        raise ValueError("Character source groups cross splits")
    def digest_ok(value):
        return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    sources = body["source_sha256"]
    if (not all(digest_ok(body[key]) for key in ("fixture_sha256", "protocol_sha256", "encoding_audit_sha256"))
            or not isinstance(sources, dict) or set(sources) != {"char_reward_lab.py", "text_reward_lab.py", "run_identity.py"}
            or not all(digest_ok(v) for v in sources.values()) or type(body["seed"]) is not int
            or body["selection"] != f"predeclared seed {body['seed']}; no heldout selection"):
        raise ValueError("Malformed character reward source/protocol/selection identity")
    config = RewardConfig(**body["config"])
    vocabulary = ASCIICharacterVocabulary(body["tokens"], body["unknown_policy"])
    if config.vocab_size != len(TOKENS) or config.max_positions != MAX_POSITIONS:
        raise ValueError("Character reward geometry differs from fixed protocol")
    actual = vocabulary.interface(config.include_eos)
    assert_compatible(body["checkpoint_interface"], actual)
    if expected_interface is not None:
        assert_compatible(expected_interface, actual)
    expected_endpoint = "final nonpadding EOS" if config.include_eos else "final completion token; EOS excluded"
    if body["endpoint_semantics"] != expected_endpoint:
        raise ValueError("Character endpoint convention differs")
    with torch.random.fork_rng():
        model = TinyTextReward(config)
    expected = model.state_dict()
    state = body["state"]
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Character state keys are incomplete or unexpected")
    restored = {}
    for key, value in state.items():
        if not _numeric_tree(value):
            raise ValueError("Character state must contain finite numeric weights, not booleans/strings")
        try:
            tensor = torch.tensor(value, dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Malformed character numeric tensor") from error
        if tensor.shape != expected[key].shape:
            raise ValueError("Character state shape mismatch")
        restored[key] = tensor
    model.load_state_dict(restored, strict=True)
    return FrozenTextReward(model, vocabulary, {**body, "payload_sha256": digest, "file_sha256": file_hash})


def run_character_reference(fixture_path, protocol_path, export_path=None):
    vocabulary = ASCIICharacterVocabulary()
    try:
        obj = json.loads(Path(fixture_path).read_text())
        if not isinstance(obj, dict):
            raise ValueError("Fixture must be an object")
    except (ValueError, OSError, RecursionError) as error:
        return {"schema": SCHEMA, "status": "failed", "seeds": [],
                "encoding_audit": {"passed": False, "issues": [{"kind": "fixture-read", "error": repr(error)}]},
                "failures": [{"stage": "fixture-read", "error": repr(error)}]}
    audit = encoding_audit(obj, vocabulary)
    result = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Original authored CPU character text models; not pretrained/human/arithmetic capability evidence",
              "protocol_sha256": file_digest(protocol_path), "fixture_sha256": file_digest(fixture_path),
              "encoding_audit": audit, "config": asdict(char_config()), "vocabulary": vocabulary.tokens,
              "source_splits": audit.get("source_splits"), "status": "failed", "seeds": [], "failures": [],
              "environment": {"interpreter": sys.executable, "prefix": sys.prefix, "python": platform.python_version(),
                              "torch": torch.__version__, "platform": platform.platform(), "device": "cpu"}}
    if not audit["passed"]:
        result["failures"].append({"stage": "pre-fit-encoding-gates", "issues": audit["issues"]})
        return result
    calibration = [r for r in obj["pairs"] if r["split"] == "calibration"]
    test = nuisance_pairs([r for r in obj["pairs"] if r["split"] == "test"])
    for seed in SEEDS:
        started, stage = time.perf_counter(), "preference-fit"
        row = {"seed": seed, "status": "failed"}
        try:
            preference, row["preference_fit"] = fit_char_model(obj, vocabulary, seed=seed)
            stage = "calibration-selection"
            panels = {t: evaluate_char_pairs(preference, vocabulary, calibration, t) for t in TEMPERATURES}
            temperature = min(TEMPERATURES, key=lambda t: panels[t]["slices"]["matched-length-format"]["nll"])
            row["temperature_selection"] = {"calibration_only": True, "tie_rule": "first ascending temperature",
                "selected": temperature, "grid_nll": {str(t): panels[t]["slices"]["matched-length-format"]["nll"] for t in TEMPERATURES}}
            row["calibration"] = panels[temperature]
            # Export the declared seed before evaluating its test panel.
            if seed == 1611 and export_path is not None:
                stage = "predeclared-export"
                row["frozen_export"] = export_char_reward(preference, vocabulary, export_path,
                    fixture_path=fixture_path, protocol_path=protocol_path, source_groups=audit["source_splits"], seed=seed)
                frozen = load_char_reward(export_path, expected_interface=vocabulary.interface(),
                                           expected_file_sha256=row["frozen_export"]["file_sha256"])
                texts = [(r["prompt"], r["left"]) for r in calibration]
                b = char_text_batch(vocabulary, texts)
                with torch.no_grad():
                    row["frozen_reload_exact"] = torch.equal(preference(b["ids"], b["mask"]), frozen.score_many(texts))
                row["frozen_parameters_have_gradients_enabled"] = any(p.requires_grad for p in frozen.model.parameters())
            stage = "outcome-fit"
            outcome, row["outcome_fit"] = fit_char_model(obj, vocabulary, seed=seed, objective="outcome")
            stage = "process-fit"
            process, row["process_fit"] = fit_char_model(obj, vocabulary, seed=seed, objective="process")
            stage = "fixed-test-evaluation"
            row.update(test_raw=evaluate_char_pairs(preference, vocabulary, test),
                       test_calibrated=evaluate_char_pairs(preference, vocabulary, test, temperature),
                       traces={s: evaluate_char_traces(outcome, process, vocabulary,
                                  [r for r in obj["processes"] if r["split"] == s])
                               for s in ("train", "calibration", "test")}, status="completed")
        except Exception as error:
            failure = {"seed": seed, "stage": stage, "error": repr(error)}
            if isinstance(error, FitFailure):
                failure["partial_fit"] = error.evidence
            row["failure"] = failure
            result["failures"].append(failure)
        row["seconds"] = time.perf_counter() - started
        result["seeds"].append(row)
    result["status"] = "completed" if not result["failures"] else "failed"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen-export", type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.frozen_export is not None and args.frozen_export.exists()):
        raise FileExistsError("Choose new character measurement/export paths; preserve old evidence")
    torch.set_num_threads(1)
    result = run_character_reference(args.fixture, args.protocol, args.frozen_export)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": result["status"], "seeds": [r["seed"] for r in result["seeds"]],
                      "encoding_gate_passed": result["encoding_audit"]["passed"]}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
