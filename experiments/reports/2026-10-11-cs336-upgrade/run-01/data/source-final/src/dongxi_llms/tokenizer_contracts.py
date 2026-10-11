"""Explicit boundary contracts around the existing teaching ByteBPE.

The regex below is an original, small segmentation rule, not a GPT-2 replica.
Chunk framing buffers one complete record; it is not a constant-memory encoder
for an arbitrary unbounded stream. No pretrained tokenizers or downloads.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import unicodedata

from .course_foundations import ByteBPE


class TokenizerContracts:
    def __init__(self, model, *, segmentation="whole", normalization="none",
                 special_tokens=(), special_policy="recognize", decode_errors="strict"):
        if segmentation not in ("whole", "words") or normalization not in ("none", "NFC"):
            raise ValueError("Unsupported segmentation or normalization")
        if special_policy not in ("recognize", "reject", "ordinary") or decode_errors not in ("strict", "replace"):
            raise ValueError("Unsupported admission or decoding policy")
        if any(not isinstance(s, str) or not s for s in special_tokens) or len(set(special_tokens)) != len(special_tokens):
            raise ValueError("Special markers must be unique nonempty strings")
        self.model, self.segmentation, self.normalization = model, segmentation, normalization
        self.special_tokens, self.special_policy = tuple(special_tokens), special_policy
        self.decode_errors = decode_errors
        self._marker = re.compile("|".join(re.escape(s) for s in sorted(special_tokens, key=lambda s: (-len(s), s)))) if special_tokens else None
        self.special_ids = {s: len(model.pieces)+i for i, s in enumerate(special_tokens)}

    def _normalize(self, text):
        if not isinstance(text, str):
            raise TypeError("Tokenizer records must be Unicode strings")
        return unicodedata.normalize("NFC", text) if self.normalization == "NFC" else text

    def segments(self, text):
        """Return (text, is_control) pieces before any learned merge."""
        text = self._normalize(text)
        blocks, start = [], 0
        if self._marker and self.special_policy != "ordinary":
            for match in self._marker.finditer(text):
                if self.special_policy == "reject":
                    raise ValueError("A reserved marker appeared in ordinary input")
                blocks.extend(self._ordinary(text[start:match.start()]))
                blocks.append((match.group(), True)); start = match.end()
        blocks.extend(self._ordinary(text[start:]))
        return blocks

    def _ordinary(self, text):
        if not text:
            return []
        chunks = [text] if self.segmentation == "whole" else re.findall(r"\s+|[^\W_]+|[^\w\s]+|_+", text, flags=re.UNICODE)
        return [(s, False) for s in chunks]

    @classmethod
    def train(cls, documents, merge_count=16, **policies):
        # Segmentation and control admission wrap, rather than duplicate, BPE.
        adapter = cls(ByteBPE([bytes([i]) for i in range(256)], [], []), **policies)
        segments = [s for text in documents for s, control in adapter.segments(text) if not control]
        model = ByteBPE.train(segments, merge_count) if segments else adapter.model
        return cls(model, **policies)

    def encode(self, text):
        return [i for segment, control in self.segments(text)
                for i in ([self.special_ids[segment]] if control else self.model.encode(segment))]

    def decode(self, ids):
        pieces = self.model.pieces + [s.encode("utf-8") for s in self.special_tokens]
        ids = list(ids)
        if any(not isinstance(i, int) or not 0 <= i < len(pieces) for i in ids):
            raise ValueError("Token ID outside the declared vocabulary")
        return b"".join(pieces[i] for i in ids).decode("utf-8", errors=self.decode_errors)

    def encode_iterable(self, iterable, *, framing="records", max_record_chars=None):
        """Records encode independently; chunks jointly form exactly one record.

        No implicit EOS is inserted. Records need not equal encode(concatenation).
        Chunk mode uses O(record length) storage, guarded by an optional limit.
        """
        if framing not in ("records", "chunks") or (max_record_chars is not None and max_record_chars <= 0):
            raise ValueError("Explicit records/chunks framing and positive optional limit required")
        pending, count = [], 0
        for text in iterable:
            if not isinstance(text, str):
                raise TypeError("Iterable elements must be Unicode strings")
            if framing == "records":
                if max_record_chars is not None and len(text) > max_record_chars:
                    raise ValueError("Record exceeds declared character limit")
                yield from self.encode(text)
            else:
                count += len(text)
                if max_record_chars is not None and count > max_record_chars:
                    raise ValueError("Buffered record exceeds declared character limit")
                pending.append(text)
        if framing == "chunks":
            yield from self.encode("".join(pending))

    def _payload(self):
        return {"format": 1, "pieces": [p.hex() for p in self.model.pieces],
                "merges": [[list(pair), new] for pair, new in self.model.merges],
                "counts": self.model.counts, "segmentation": self.segmentation,
                "normalization": self.normalization, "special_tokens": list(self.special_tokens),
                "special_policy": self.special_policy, "decode_errors": self.decode_errors}

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self._payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_json(self):
        return json.dumps({"identity": self.identity, "tokenizer": self._payload()}, sort_keys=True)

    @classmethod
    def from_json(cls, text):
        envelope = json.loads(text); payload = envelope["tokenizer"]
        if payload["format"] != 1:
            raise ValueError("Unsupported serialization format")
        pieces = [bytes.fromhex(p) for p in payload["pieces"]]
        merges = [(tuple(pair), new) for pair, new in payload["merges"]]
        if pieces[:256] != [bytes([i]) for i in range(256)] or len(pieces) != 256+len(merges) or len(payload["counts"]) != len(merges):
            raise ValueError("Invalid byte vocabulary/merge geometry")
        for step, (pair, new) in enumerate(merges, 256):
            if len(pair) != 2 or new != step or any(not isinstance(i, int) or not 0 <= i < step for i in pair) or pieces[new] != pieces[pair[0]]+pieces[pair[1]]:
                raise ValueError("Merge references or piece bytes are inconsistent")
        obj = cls(ByteBPE(pieces, merges, payload["counts"]),
                  **{k: payload[k] for k in ("segmentation", "normalization", "special_tokens", "special_policy", "decode_errors")})
        if obj.identity != envelope["identity"]:
            raise ValueError("Serialized identity does not match content")
        return obj


@dataclass(frozen=True)
class ByteScoringContract:
    normalization: str = "none"
    context: str = "document-local; fixed prefix"
    mask: str = "all text bytes"
    eos: str = "excluded from NLL and byte denominator"


@dataclass(frozen=True)
class ByteScore:
    """scored_text is the exact raw span whose likelihood entered nll_sum."""
    nll_sum: float
    scored_text: str
    contract: ByteScoringContract = ByteScoringContract()


def aggregate_byte_scores(records):
    records = list(records)
    if not records or len({r.contract for r in records}) != 1:
        raise ValueError("Nonempty records with compatible scoring contracts required")
    if any(not math.isfinite(r.nll_sum) or r.nll_sum < 0 or not isinstance(r.scored_text, str) for r in records):
        raise ValueError("Finite nonnegative summed NLL and explicit scored text required")
    lengths = [len(r.scored_text.encode("utf-8")) for r in records]
    if any(n == 0 for n in lengths):
        raise ValueError("Each record must correspond to scored raw bytes")
    total, count = math.fsum(r.nll_sum for r in records), sum(lengths)
    return {"summed_nll": total, "scored_bytes": count,
            "nats_per_byte": total/count, "bits_per_byte": total/(count*math.log(2)),
            "contract": asdict(records[0].contract)}
