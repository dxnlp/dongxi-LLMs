"""Offline evidence, byte-BPE and budget microscopes for Chapters 1, 2 and 6.

ByteBPE is an educational whole-document byte encoder. It deliberately omits
production regex pretokenization, normalization and special-token handling.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math


def fingerprint(configuration):
    """Hash canonical JSON; any changed experiment field changes its identity."""
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assess_smoke(exit_code, losses, minimum_available_gib, reserve_gib):
    """Three independent acceptance conditions; no capability inference."""
    if reserve_gib <= 0:
        raise ValueError("reserve must be positive")
    checks = {
        "successful_exit": exit_code == 0,
        "finite_observed_losses": bool(losses) and all(math.isfinite(x) for x in losses),
        "sampled_reserve_preserved": minimum_available_gib is not None
        and math.isfinite(minimum_available_gib) and minimum_available_gib >= reserve_gib,
    }
    return {"criteria": checks, "accepted": all(checks.values()),
            "claim": "The stated smoke criteria passed." if all(checks.values())
            else "At least one smoke criterion failed or was unobserved."}


@dataclass
class ByteBPE:
    pieces: list
    merges: list
    counts: list

    @classmethod
    def train(cls, documents, merge_count=16):
        if merge_count < 0 or not documents or any(not isinstance(x, str) for x in documents):
            raise ValueError("nonempty text corpus and nonnegative merge count required")
        sequences = [list(x.encode("utf-8")) for x in documents]
        pieces, merges, counts = [bytes([b]) for b in range(256)], [], []
        for _ in range(merge_count):
            pairs = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not pairs:
                break
            # Frequency tie has a declared deterministic lexicographic rule.
            pair = min(pairs, key=lambda p: (-pairs[p], p))
            new_id = len(pieces)
            pieces.append(pieces[pair[0]] + pieces[pair[1]])
            merges.append((pair, new_id)); counts.append(pairs[pair])
            sequences = [cls._merge(seq, pair, new_id) for seq in sequences]
        return cls(pieces, merges, counts)

    @staticmethod
    def _merge(sequence, pair, new_id):
        out, i = [], 0
        while i < len(sequence):
            if i + 1 < len(sequence) and tuple(sequence[i:i+2]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(sequence[i]); i += 1
        return out

    def encode(self, text):
        seq = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            seq = self._merge(seq, pair, new_id)
        return seq

    def decode(self, ids):
        return b"".join(self.pieces[i] for i in ids).decode("utf-8")


def budget_summary(evidence):
    m, c = evidence["metrics_summary"], evidence["completion"]
    return {
        "valid_fraction": m["valid_targets"] / m["processed_positions"],
        "update_targets_per_second": m["valid_targets"] / m["sum_update_seconds"],
        "run_targets_per_second": m["valid_targets"] / c["seconds"],
        "exposure_over_prepared": m["valid_targets"] / evidence["data_manifest"]["train"]["valid_targets"],
    }


def visible_sources(document_ids):
    """Document-isolated causal [T,T] mask for an explicitly packed toy sequence."""
    import torch
    ids = torch.as_tensor(document_ids)
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("nonempty document ID vector required")
    causal = torch.arange(ids.numel())[:, None] >= torch.arange(ids.numel())[None, :]
    return causal & (ids[:, None] == ids[None, :])
