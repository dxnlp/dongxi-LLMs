"""Offline, bounded response replay for Chapter 7.

This is an original teaching instrument, not a general symbolic verifier. Its
grammar is finite exact rational arithmetic, finite sets and real intervals.
Unsupported algebra stays UNSUPPORTED. It never evaluates Python/model text,
imports SymPy, calls an API or loads a model. Units are matched, not converted.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
import random
import re

from dongxi_llms.evaluation_lab import canonical_hash

SCHEMA_VERSION = "dongxi-response-record-v1"
PARSER_VERSION = "bounded-rational-set-interval-v1"
RUBRIC_VERSION = "explicit-marker-fixture-v1"
MAX_RESPONSE = 16384
MAX_ANSWER = 2048
MAX_TOKENS = 256
MAX_DEPTH = 12
MAX_ELEMENTS = 64
MAX_DIGITS = 64
MAX_BITS = 4096
UNITS = {"m": "m", "cm": "cm", "kg": "kg", "g": "g", "s": "s",
         "second": "s", "seconds": "s"}


class AnswerError(ValueError):
    def __init__(self, status, reason):
        super().__init__(reason)
        self.status, self.reason = status, reason


def _reject(status, reason):
    raise AnswerError(status, reason)


def extract_answer(text, policy="boxed_or_final"):
    """One balanced box, one explicit Final answer line, or a whole response.

    Multiple boxes/final lines are ambiguous even if they agree. A response with
    both a box and a Final answer line is also ambiguous; no last-number rescue.
    """
    if not isinstance(text, str) or len(text) > MAX_RESPONSE:
        return {"status": "INVALID", "reason": "response type/length bound"}
    if policy not in ("boxed_or_final", "whole"):
        raise ValueError("Unknown extraction policy")
    if policy == "whole":
        return {"status": "EXTRACTED", "answer": text.strip(), "mechanism": "whole"}
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    finals = re.findall(r"^\s*Final answer:\s*([^\n]+)\s*$", text, re.MULTILINE)
    if len(starts) + len(finals) > 1:
        return {"status": "AMBIGUOUS", "reason": "multiple explicit answer markers"}
    if starts:
        begin = starts[0].end()
        depth = 1
        for index in range(begin, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            if depth > MAX_DEPTH:
                return {"status": "INVALID", "reason": "box nesting bound"}
            if depth == 0:
                return {"status": "EXTRACTED", "answer": text[begin:index].strip(),
                        "mechanism": "boxed"}
        return {"status": "INVALID", "reason": "unclosed answer box"}
    if finals:
        return {"status": "EXTRACTED", "answer": finals[0].strip(), "mechanism": "final-line"}
    return {"status": "EXTRACTED", "answer": text.strip(), "mechanism": "whole"}


class _RationalParser:
    """Hand-written arithmetic parser; no eval/exec/symbolic fallback."""
    pattern = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)|\\(?:d?frac)|[+*/(){}-]")

    def __init__(self, text):
        compact = re.sub(r"\s+", "", text)
        # Tokenize before deleting whitespace: "1 2" must not become "12".
        self.tokens = self.pattern.findall(text)
        if "".join(self.tokens) != compact:
            _reject("UNSUPPORTED", "outside exact rational grammar")
        if not self.tokens or len(self.tokens) > MAX_TOKENS:
            _reject("INVALID", "empty expression or token bound")
        self.index = 0

    def take(self, wanted=None):
        if self.index >= len(self.tokens):
            _reject("INVALID", "incomplete arithmetic")
        value = self.tokens[self.index]
        if wanted is not None and value != wanted:
            _reject("INVALID", "unbalanced arithmetic delimiter")
        self.index += 1
        return value

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    @staticmethod
    def bounded(value):
        if value.numerator.bit_length() > MAX_BITS or value.denominator.bit_length() > MAX_BITS:
            _reject("INVALID", "rational magnitude bound")
        return value

    def expression(self, depth=0):
        if depth > MAX_DEPTH:
            _reject("INVALID", "arithmetic nesting bound")
        value = self.term(depth)
        while self.peek() in ("+", "-"):
            operation = self.take()
            other = self.term(depth)
            value = self.bounded(value + other if operation == "+" else value - other)
        return value

    def term(self, depth):
        value = self.atom(depth)
        while self.peek() in ("*", "/"):
            operation = self.take()
            other = self.atom(depth)
            if operation == "/" and other == 0:
                _reject("INVALID", "zero denominator")
            value = self.bounded(value * other if operation == "*" else value / other)
        return value

    def atom(self, depth):
        if depth > MAX_DEPTH:
            _reject("INVALID", "arithmetic nesting bound")
        token = self.take()
        if token in ("+", "-"):
            value = self.atom(depth + 1)
            return value if token == "+" else -value
        if token in ("(", "{"):
            value = self.expression(depth + 1)
            self.take(")" if token == "(" else "}")
            return value
        if token in (r"\frac", r"\dfrac"):
            self.take("{")
            numerator = self.expression(depth + 1)
            self.take("}")
            self.take("{")
            denominator = self.expression(depth + 1)
            self.take("}")
            if denominator == 0:
                _reject("INVALID", "zero denominator")
            return self.bounded(numerator / denominator)
        if re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", token):
            if len(token.replace(".", "")) > MAX_DIGITS:
                _reject("INVALID", "numeric digit bound")
            return self.bounded(Fraction(token))
        _reject("INVALID", "expected rational atom")

    def parse(self):
        result = self.expression()
        if self.peek() is not None:
            _reject("INVALID", "trailing arithmetic tokens")
        return result


def _fraction(text):
    value = _RationalParser(text).parse()
    return [value.numerator, value.denominator]


def _split_top_level(text):
    parts, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        if depth < 0 or depth > MAX_DEPTH:
            _reject("INVALID", "collection delimiter/depth bound")
        if char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1
    if depth != 0:
        _reject("INVALID", "unclosed collection delimiter")
    parts.append(text[start:])
    if len(parts) > MAX_ELEMENTS:
        _reject("INVALID", "collection element bound")
    return parts


def canonicalize_answer(text):
    """Exact constants/finite sets/intervals; preserve unit and boundary identity."""
    try:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_ANSWER:
            _reject("INVALID", "answer type/length bound")
        value = text.strip()
        if value.startswith("$") and value.endswith("$"):
            value = value[1:-1].strip()
        value = value.replace(r"\left", "").replace(r"\right", "")
        value = value.replace(r"\{", "{").replace(r"\}", "}")
        value = value.replace(r"\,", " ").replace(r"\!", "")
        value = re.sub(r"\\(?:text|mathrm)\{([A-Za-z]+)\}\s*$", r" \1", value)
        unit = None
        suffix = re.search(r"\s+([A-Za-z]+)\s*$", value)
        if suffix:
            if suffix[1] not in UNITS:
                _reject("UNSUPPORTED", "unknown unit; no implicit conversion")
            unit = UNITS[suffix[1]]
            value = value[:suffix.start()].strip()
        if value.startswith("{") and value.endswith("}"):
            inner = value[1:-1].strip()
            elements = [] if not inner else [_fraction(part) for part in _split_top_level(inner)]
            elements = sorted(set(tuple(element) for element in elements), key=lambda pair: Fraction(*pair))
            canonical = {"kind": "set", "elements": [list(pair) for pair in elements], "unit": unit}
        elif value[:1] in ("[", "(") and value[-1:] in ("]", ")") and "," in value:
            parts = _split_top_level(value[1:-1])
            if len(parts) != 2:
                _reject("UNSUPPORTED", "only one-dimensional intervals supported")
            def endpoint(part):
                normalized = part.strip().replace(r"\infty", "inf")
                if normalized in ("inf", "+inf", "-inf"):
                    return "-inf" if normalized == "-inf" else "+inf"
                return _fraction(part)
            left, right = map(endpoint, parts)
            closed = [value[0] == "[", value[-1] == "]"]
            if left == "+inf" or right == "-inf" or (left == "-inf" and closed[0]) or (right == "+inf" and closed[1]):
                _reject("INVALID", "invalid infinite interval endpoint")
            if isinstance(left, list) and isinstance(right, list):
                if Fraction(*left) > Fraction(*right) or (left == right and not all(closed)):
                    _reject("INVALID", "reversed or empty interval unsupported")
            canonical = {"kind": "interval", "left": left, "right": right, "closed": closed, "unit": unit}
        else:
            canonical = {"kind": "scalar", "value": _fraction(value), "unit": unit}
        return {"status": "SUPPORTED", "canonical": canonical}
    except AnswerError as error:
        return {"status": error.status, "reason": error.reason}


def _bounded_json(text):
    if len(text) > MAX_ANSWER:
        _reject("INVALID", "JSON length bound")
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                _reject("INVALID", "duplicate JSON key")
            result[key] = value
        return result
    try:
        parsed = json.loads(text, object_pairs_hook=pairs,
                            parse_constant=lambda _: _reject("INVALID", "nonfinite JSON constant"))
    except (ValueError, RecursionError) as error:
        _reject("INVALID", str(error))
    def inspect(value, depth=0):
        if depth > MAX_DEPTH:
            _reject("INVALID", "JSON nesting bound")
        if isinstance(value, dict):
            if len(value) > MAX_ELEMENTS:
                _reject("INVALID", "JSON object size bound")
            for child in value.values():
                inspect(child, depth + 1)
        elif isinstance(value, list):
            if len(value) > MAX_ELEMENTS:
                _reject("INVALID", "JSON array size bound")
            for child in value:
                inspect(child, depth + 1)
        elif isinstance(value, float) and not math.isfinite(value):
            _reject("INVALID", "nonfinite JSON number")
    inspect(parsed)
    if not isinstance(parsed, dict):
        _reject("INVALID", "JSON object required")
    return parsed


def freeze_contract(items, settings):
    """Serialize a complete offline scoring contract; no mutation or model work."""
    required = ("template_id", "thinking_mode", "decoding", "stopping", "max_new_tokens")
    if any(key not in settings for key in required):
        raise ValueError("Template, thinking, decoding, stopping and cap must be declared")
    if type(settings["max_new_tokens"]) is not int or not 1 <= settings["max_new_tokens"] <= MAX_RESPONSE:
        raise ValueError("Positive bounded response cap required")
    if not items:
        raise ValueError("Nonempty frozen suite required")
    identifiers, groups = set(), {}
    for item in items:
        for key in ("id", "source_group", "split", "task", "prompt", "kind", "reference"):
            if key not in item:
                raise ValueError(f"Missing item field: {key}")
        for key in ("id", "source_group", "split", "task", "prompt"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError("Nonempty string item identities/prompt required")
        if item["id"] in identifiers:
            raise ValueError("Duplicate item ID")
        identifiers.add(item["id"])
        prior = groups.setdefault(item["source_group"], item["split"])
        if prior != item["split"]:
            raise ValueError("Source group crosses evaluation splits")
        if item["kind"] not in ("math", "text", "json", "rubric"):
            raise ValueError("Unsupported task kind")
        if item["kind"] == "math" and canonicalize_answer(item["reference"])["status"] != "SUPPORTED":
            raise ValueError("Author reference outside supported math grammar")
        if item["kind"] in ("text", "json") and not isinstance(item["reference"], str):
            raise ValueError("Text/JSON reference must be a string")
        if item["kind"] == "json":
            try:
                _bounded_json(item["reference"])
            except AnswerError as error:
                raise ValueError("Invalid author JSON reference") from error
        if item["kind"] == "rubric":
            rubric = item["reference"]
            if not isinstance(rubric, dict) or not (rubric.get("required_all") or rubric.get("required_any_groups")):
                raise ValueError("Rubric requires explicit positive criteria")
            flat = rubric.get("required_all", []) + rubric.get("forbidden_any", [])
            alternatives = rubric.get("required_any_groups", [])
            if any(not isinstance(group, list) or not group for group in alternatives):
                raise ValueError("Rubric alternative groups must be nonempty lists")
            flat += [marker for group in alternatives for marker in group]
            if any(not isinstance(marker, str) or not marker for marker in flat):
                raise ValueError("Rubric markers must be nonempty strings")
    contract = {"schema_version": SCHEMA_VERSION, "parser_version": PARSER_VERSION,
                "rubric_version": RUBRIC_VERSION, "suite_sha256": canonical_hash(items),
                "settings": json.loads(json.dumps(settings, allow_nan=False))}
    contract["identity"] = canonical_hash(contract)
    return contract


def validate_record(record, contract, item):
    required = ("schema_version", "contract_id", "checkpoint_id", "sample_id", "item_id",
                "source_group", "task", "split", "raw_response", "token_ids", "prompt_tokens",
                "generated_tokens", "stop_reason", "truncated", "error", "cost")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing response fields: {missing}")
    if record["schema_version"] != SCHEMA_VERSION or record["contract_id"] != contract["identity"]:
        raise ValueError("Response schema/contract identity mismatch")
    for key, item_key in (("item_id", "id"), ("source_group", "source_group"), ("task", "task"), ("split", "split")):
        if record[key] != item[item_key]:
            raise ValueError("Response item/source/task/split mismatch")
    if not isinstance(record["checkpoint_id"], str) or not record["checkpoint_id"]:
        raise ValueError("Nonempty checkpoint identity required")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError("Nonempty sample identity required")
    # Storage preserves full output. The extraction/grammar bounds are separate
    # and return INVALID rather than deleting an oversized model response.
    if not isinstance(record["raw_response"], str):
        raise ValueError("Raw response must remain a string")
    if record["stop_reason"] not in ("eos", "turn_stop", "max_tokens", "context_limit", "error", "unknown") or type(record["truncated"]) is not bool:
        raise ValueError("Explicit stopping/truncation required")
    if record["truncated"] != (record["stop_reason"] in ("max_tokens", "context_limit")):
        raise ValueError("Truncation must agree with the recorded output/context boundary")
    if record["error"] is not None and not isinstance(record["error"], str):
        raise ValueError("Error is a string or explicit null")
    if (record["error"] is not None) != (record["stop_reason"] == "error"):
        raise ValueError("Error and stop reason must agree")
    for key in ("prompt_tokens", "generated_tokens"):
        value = record[key]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Token count must be a nonnegative integer or unknown null")
    token_ids = record["token_ids"]
    if token_ids is not None:
        if not isinstance(token_ids, list) or len(token_ids) > MAX_RESPONSE or any(type(token) is not int or token < 0 for token in token_ids):
            raise ValueError("Token IDs must be integer list or explicitly unknown")
        if record["generated_tokens"] != len(token_ids):
            raise ValueError("Generated token count disagrees with recorded token IDs")
    if record["generated_tokens"] is not None and record["generated_tokens"] > contract["settings"]["max_new_tokens"]:
        raise ValueError("Response exceeds frozen generation cap")
    if record["stop_reason"] == "max_tokens" and record["generated_tokens"] is not None:
        if record["generated_tokens"] != contract["settings"]["max_new_tokens"]:
            raise ValueError("Known max-token termination must reach the declared cap")
    costs = record["cost"]
    if not isinstance(costs, dict) or any(key not in costs for key in ("wall_seconds", "generation_tokens", "scoring_tokens")):
        raise ValueError("All cost boundaries required, with null for unknown")
    for key, value in costs.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            raise ValueError("Costs must be finite nonnegative numbers or explicit null")
        if key in ("generation_tokens", "scoring_tokens") and value is not None and type(value) is not int:
            raise ValueError("Token costs must be integer counts or explicit null")
    if costs["generation_tokens"] != record["generated_tokens"]:
        raise ValueError("Generation cost and token count disagree")
    generation = contract["settings"].get("generation")
    if generation is not None:
        if record.get("adapter_version") != "local-hf-response-v1" or not isinstance(record.get("response_text"), str):
            raise ValueError("Generated contracts require the actual adapter and grading-text record")
        for key in ("raw_response", "response_text"):
            if record.get(key + "_sha256") != hashlib.sha256(record[key].encode()).hexdigest():
                raise ValueError("Retained response text fingerprint changed")
        if record.get("checkpoint_interface_sha256") != generation["interface"]["interface_sha256"]:
            raise ValueError("Generated response interface differs from its frozen contract")
        from dongxi_llms.run_identity import canonical_hash as identity_hash
        if record.get("settings_sha256") != identity_hash(contract["settings"]):
            raise ValueError("Generated response settings fingerprint changed")
        if re.fullmatch(r"[0-9a-f]{64}", record.get("input_identity_sha256", "")) is None:
            raise ValueError("Actual invocation input identity required")
        prompt_ids = record.get("prompt_token_ids")
        if prompt_ids is not None and (not isinstance(prompt_ids, list) or any(type(t) is not int or t < 0 for t in prompt_ids) or len(prompt_ids) != record["prompt_tokens"]):
            raise ValueError("Actual prompt IDs/count mismatch")
        if token_ids is None or record["generated_tokens"] is None:
            raise ValueError("Actual generated token IDs/counts cannot be unknown")
        for key in ("selected_behavior_log_probabilities", "selected_raw_log_probabilities", "retained_support_sizes"):
            values = record.get(key)
            if not isinstance(values, list) or len(values) != len(token_ids) or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Per-action likelihood/support records must match actual IDs")
        if record["stop_reason"] in ("eos", "turn_stop"):
            category = "eos_token_ids" if record["stop_reason"] == "eos" else "turn_stop_token_ids"
            all_stops = contract["settings"]["stopping"]["eos_token_ids"] + contract["settings"]["stopping"]["turn_stop_token_ids"]
            if not token_ids or token_ids[-1] not in contract["settings"]["stopping"][category] or record.get("stop_token_id") != token_ids[-1] or any(t in all_stops for t in token_ids[:-1]):
                raise ValueError("Claimed natural stop does not match the first actual generated stop")
        if record["stop_reason"] == "context_limit" and (record["prompt_tokens"] is None or record["prompt_tokens"] + len(token_ids) < generation["context_window"]):
            raise ValueError("Claimed context boundary was not reached")
        for key in ("model_forward_tokens", "attempted_forward_tokens", "forward_calls", "attempted_forward_calls"):
            if type(costs.get(key)) is not int or costs[key] < 0:
                raise ValueError("Measured forward counts must be nonnegative integers")
        if costs["attempted_forward_tokens"] < costs["model_forward_tokens"] or costs["attempted_forward_calls"] < costs["forward_calls"]:
            raise ValueError("Completed forwards cannot exceed attempted forward work")


def candidate_view(record):
    """Non-oracle selection input: raw generation metadata, never reference/grade."""
    allowed = ("schema_version", "contract_id", "checkpoint_id", "sample_id", "item_id",
               "source_group", "task", "split", "raw_response", "token_ids", "prompt_tokens",
               "generated_tokens", "stop_reason", "truncated", "error", "cost")
    optional = ("response_text", "adapter_version", "prompt_token_ids", "attempt_seed",
                "sample_index", "checkpoint_interface_sha256", "input_identity_sha256",
                "decoding", "thinking_mode")
    return {key: record[key] for key in allowed + optional if key in record}


def grade_response(item, record):
    """Separate answer correctness, required format and termination evidence."""
    text = record.get("response_text", record["raw_response"])
    extracted = extract_answer(text, item.get("extraction", "boxed_or_final"))
    result = dict(record, extraction=extracted, format_valid=False, task_success=False,
                  natural_termination=record["stop_reason"] in ("eos", "turn_stop"), correct=False)
    if record["error"] is not None:
        return dict(result, status="ERROR", reason=record["error"])
    raw = text.strip()
    policy = item.get("format_policy", "any")
    if policy not in ("any", "single_integer", "one_word", "json_object"):
        raise ValueError("Unknown format policy")
    try:
        # Grader coverage is not a format constraint. An unsupported square root
        # may obey an unconstrained interface even though we cannot grade it.
        formatted = (policy == "any" or
                     policy == "single_integer" and re.fullmatch(r"[+-]?[0-9]+", raw, flags=re.ASCII) is not None or
                     policy == "one_word" and re.fullmatch(r"[A-Za-z]+", raw) is not None or
                     policy == "json_object" and isinstance(_bounded_json(raw), dict))
    except AnswerError:
        formatted = False
    result["format_valid"] = bool(formatted)
    if extracted["status"] != "EXTRACTED":
        return dict(result, status=extracted["status"], reason=extracted["reason"])
    answer = extracted["answer"]
    try:
        if item["kind"] == "math":
            parsed = canonicalize_answer(answer)
            result["canonicalization"] = parsed
            if parsed["status"] != "SUPPORTED":
                return dict(result, status=parsed["status"], reason=parsed["reason"])
            reference = canonicalize_answer(item["reference"])
            correct = parsed["canonical"] == reference["canonical"]
        elif item["kind"] == "json":
            parsed = _bounded_json(raw)
            reference = _bounded_json(item["reference"])
            correct = canonical_hash(parsed) == canonical_hash(reference)
        elif item["kind"] == "rubric":
            rubric = item["reference"]
            lowered = raw.casefold()
            correct = (all(marker.casefold() in lowered for marker in rubric.get("required_all", []))
                       and all(any(marker.casefold() in lowered for marker in alternatives)
                               for alternatives in rubric.get("required_any_groups", []))
                       and not any(marker.casefold() in lowered for marker in rubric.get("forbidden_any", [])))
        else:
            correct = " ".join(answer.casefold().split()) == " ".join(item["reference"].casefold().split())
        return dict(result, status="CORRECT" if correct else "INCORRECT", correct=bool(correct),
                    format_valid=bool(formatted), task_success=bool(correct and formatted))
    except AnswerError as error:
        return dict(result, status=error.status, reason=error.reason, task_success=False)


def _summary(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {"n": n, "accuracy": sum(row["correct"] for row in rows)/n,
            "task_success": sum(row.get("task_success", False) for row in rows)/n,
            "format_valid": sum(row["format_valid"] for row in rows)/n,
            "natural_termination": sum(row["natural_termination"] for row in rows)/n,
            "truncation": sum(row["truncated"] for row in rows)/n,
            "stop_reasons": dict(Counter(row["stop_reason"] for row in rows)),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "cost": {key: {"known_total": sum(row["cost"][key] or 0 for row in rows),
                           "unknown_rows": sum(row["cost"][key] is None for row in rows)}
                     for key in ("wall_seconds", "generation_tokens", "scoring_tokens")}}


def replay_records(items, records, contract):
    """Grade every retained response. Invalid schema fails instead of dropping rows."""
    declared = {key: value for key, value in contract.items() if key != "identity"}
    if contract.get("identity") != canonical_hash(declared) or contract["suite_sha256"] != canonical_hash(items):
        raise ValueError("Frozen contract or suite hash mismatch")
    if contract["schema_version"] != SCHEMA_VERSION or contract["parser_version"] != PARSER_VERSION or contract["rubric_version"] != RUBRIC_VERSION:
        raise ValueError("Scoring version mismatch; named migration required")
    mapping = {item["id"]: item for item in items}
    seen, rows = set(), []
    for record in records:
        if record.get("item_id") not in mapping:
            raise ValueError("Unknown evaluation item")
        key = (record["checkpoint_id"], record["item_id"], record["sample_id"])
        if key in seen:
            raise ValueError("Duplicate checkpoint/item/sample response")
        seen.add(key)
        item = mapping[record["item_id"]]
        validate_record(record, contract, item)
        rows.append(grade_response(item, record))
    by_checkpoint = defaultdict(list)
    for row in rows:
        by_checkpoint[row["checkpoint_id"]].append(row)
    checkpoints = {}
    for checkpoint, checkpoint_rows in sorted(by_checkpoint.items()):
        slices = {}
        for key in ("task", "source_group", "split"):
            groups = defaultdict(list)
            for row in checkpoint_rows:
                groups[row[key]].append(row)
            slices[key] = {name: _summary(group) for name, group in sorted(groups.items())}
        missing_items = sorted(set(mapping)-{row["item_id"] for row in checkpoint_rows})
        checkpoints[checkpoint] = {"summary": _summary(checkpoint_rows), "slices": slices,
                                   "missing_item_ids": missing_items}
    return {"contract": contract, "mode": "offline saved-response replay", "rows": rows,
            "checkpoints": checkpoints, "limits": "Authored fixtures or supplied records; no generation or broad safety/capability result."}


def paired_group_bootstrap(a, b, draws=2000, seed=1010, metric="task_success"):
    """Paired source-cluster resampling; estimand is micro-average B minus A.

    Each bootstrap draw samples source groups with replacement, retaining every
    item/sample within each group. Unequal group sizes remain unequal. Few groups
    give descriptive, fragile intervals, not proof of population improvement.
    """
    if not a or draws < 1 or metric not in ("correct", "task_success", "format_valid"):
        raise ValueError("Nonempty rows, positive draws and declared metric required")
    def index(rows):
        out = {}
        for row in rows:
            key = (row["item_id"], row["sample_id"])
            if key in out:
                raise ValueError("Repeated paired ID")
            out[key] = row
        return out
    left, right = index(a), index(b)
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical item/sample coverage")
    contracts = {row["contract_id"] for row in a+b}
    if len(contracts) != 1:
        raise ValueError("Paired comparison contract mismatch")
    groups = defaultdict(list)
    for key in sorted(left):
        if left[key]["source_group"] != right[key]["source_group"]:
            raise ValueError("Paired source group mismatch")
        groups[left[key]["source_group"]].append(float(right[key].get(metric, False))-float(left[key].get(metric, False)))
    names = sorted(groups)
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        differences = [difference for name in rng.choices(names, k=len(names)) for difference in groups[name]]
        samples.append(sum(differences)/len(differences))
    ordered = sorted(samples)
    return {"metric": metric, "delta": sum(sum(values) for values in groups.values())/len(left),
            "interval": [ordered[int(.025*(draws-1))], ordered[int(.975*(draws-1))]],
            "n_items_samples": len(left), "n_source_groups": len(names), "unit": "source_group",
            "draws": draws, "seed": seed, "samples": samples,
            "limits": "Percentile cluster bootstrap; a small authored panel is not a generalization benchmark."}
