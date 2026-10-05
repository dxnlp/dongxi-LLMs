"""Pinned TinyStories preparation; bounded streaming, explicit document windows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

DATA_REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EOS = 50256
IGNORE = -100


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokenizer():
    path = hf_hub_download("openai-community/gpt2", "tokenizer.json", revision=TOKENIZER_REV)
    tok = Tokenizer.from_file(path)
    if tok.get_vocab_size() != 50257 or tok.token_to_id("<|endoftext|>") != EOS:
        raise ValueError("Unexpected tokenizer contract")
    return tok, path


def documents(lines, verified_eof=False):
    """Delimiter must occupy its own line; preserve document text/whitespace.

    The final line ending before a delimiter is format framing, not story text.
    Fail on an unterminated document (e.g. a truncated network response).
    """
    pending = []
    for line in lines:
        if line.rstrip("\r\n") == "<|endoftext|>":
            text = "".join(pending)
            if text.endswith("\n"):
                text = text[:-1]
                if text.endswith("\r"):
                    text = text[:-1]
            if text.strip():
                yield text
            pending = []
        else:
            pending.append(line)
    if "".join(pending).strip():
        if verified_eof:
            yield "".join(pending)
            return
        raise ValueError("Source ends with an unterminated document")


def remote_documents(split):
    name = f"TinyStories-{'train' if split == 'train' else 'valid'}.txt"
    url = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{DATA_REV}/{name}"
    # Do not log headers or credentials; these files are public.
    with requests.get(url, stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        yield from documents(line + "\n" for line in response.iter_lines(decode_unicode=True))


def write_split(root, split, texts, encode, length, limit, train_hashes=None,
                filter_overlap=False, deduplicate=False):
    """Store uint32 tokens and (offset, target_count) windows; no cross-doc packing."""
    if length < 1 or limit < 0:
        raise ValueError("Positive context and nonnegative document limit required")
    root = Path(root)
    hashes, windows, lengths = set(), [], []
    duplicates = 0
    overlaps = 0
    position = 0
    source_hash = hashlib.sha256()
    with (root / f"{split}.bin").open("xb") as f:
        for text in texts:
            normalized = " ".join(text.casefold().split())
            h = hashlib.sha256(normalized.encode()).hexdigest()
            if train_hashes is not None and h in train_hashes:
                if filter_overlap:
                    overlaps += 1
                    continue
                raise ValueError("Normalized exact train/validation overlap")
            duplicates += h in hashes
            if deduplicate and h in hashes:
                continue
            hashes.add(h)
            # Length framing makes this digest unambiguous across document boundaries.
            raw = text.encode("utf-8")
            source_hash.update(len(raw).to_bytes(8, "little") + raw)
            ids = encode(text)
            if not ids or min(ids) < 0 or max(ids) >= EOS:
                raise ValueError("Empty tokenization or special token embedded in story")
            seq = np.asarray([EOS, *ids, EOS], dtype="<u4")
            f.write(seq.tobytes())
            targets = len(seq) - 1
            lengths.append(targets)
            for start in range(0, targets, length):
                windows.append((position + start, min(length, targets - start)))
            position += len(seq)
            if len(lengths) % 10000 == 0:
                print(json.dumps(dict(split=split, prepared_documents=len(lengths),
                                      duplicate_documents=duplicates, excluded_overlap=overlaps)), flush=True)
            if limit and len(lengths) >= limit:
                break
    if not lengths:
        raise ValueError("Empty split")
    np.save(root / f"{split}.windows.npy", np.asarray(windows, dtype=np.int64))
    return dict(documents=len(lengths), windows=len(windows), valid_targets=sum(lengths),
                long_documents=sum(n > length for n in lengths), duplicate_documents=duplicates,
                excluded_overlap=overlaps, deduplicated=deduplicate,
                target_length_quantiles=np.percentile(lengths, [0, 50, 90, 99, 100]).tolist(),
                framed_text_sha256=source_hash.hexdigest(),
                files={f"{split}.bin": digest(root / f"{split}.bin"),
                       f"{split}.windows.npy": digest(root / f"{split}.windows.npy")}), hashes


def prepare(output, length=1024, train_limit=1024, valid_limit=128):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    tok, path = tokenizer()
    manifest = dict(schema=1, dataset="roneneldan/TinyStories", dataset_revision=DATA_REV,
                    tokenizer_revision=TOKENIZER_REV, tokenizer_sha256=digest(path),
                    vocab=50257, eos=EOS, context=length,
                    subset=bool(train_limit or valid_limit),
                    limits={"train": train_limit, "valid": valid_limit},
                    policy="separate-docs; target windows; reset context; positional padding mask")
    (root / "tokenizer.json").write_bytes(Path(path).read_bytes())
    hashes = None
    for split, limit in (("train", train_limit), ("valid", valid_limit)):
        stream = remote_documents(split)
        try:
            stats, observed = write_split(root, split, stream,
                lambda s: tok.encode(s, add_special_tokens=False).ids,
                length, limit, hashes if split == "valid" else None)
        finally:
            stream.close()
        manifest[split] = stats
        if split == "train":
            hashes = observed
        print(json.dumps({"split": split, **stats}), flush=True)
    # Manifest appears only after both splits are complete and audited.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_full(output, length=1024):
    """Keep validation fixed first; filter exact overlaps from TRAIN, never vice versa."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    tok, path = tokenizer()
    manifest = dict(schema=1, dataset="roneneldan/TinyStories", dataset_revision=DATA_REV,
                    tokenizer_revision=TOKENIZER_REV, tokenizer_sha256=digest(path),
                    vocab=50257, eos=EOS, context=length, subset=False,
                    policy="full; dedup normalized exact; validation-first exclusion from train; separate-document windows")
    (root/"tokenizer.json").write_bytes(Path(path).read_bytes())
    validation_hashes = None
    manifest["raw_files"] = {}
    raw_sha256 = {
        "valid": "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4",
        "train": "c5cf5e22ff13614e830afbe61a99fbcbe8bcb7dd72252b989fa1117a368d401f",
    }
    for split in ("valid", "train"):
        name = f"TinyStories-{split}.txt"
        raw_path = Path(hf_hub_download("roneneldan/TinyStories", name,
                                      repo_type="dataset", revision=DATA_REV))
        sha = digest(raw_path)
        if sha != raw_sha256[split]:
            raise ValueError("Raw source SHA256 does not match pinned HF LFS identity")
        manifest["raw_files"][name] = dict(sha256=sha, bytes=raw_path.stat().st_size)
        # The complete official files may omit a delimiter after their final story.
        # EOF is accepted only after validating the COMPLETE raw-file SHA256.
        raw_file = raw_path.open(encoding="utf-8")
        stream = documents(raw_file, verified_eof=True)
        try:
            stats, hashes = write_split(root, split, stream,
                lambda s: tok.encode(s, add_special_tokens=False).ids,
                length, 0, validation_hashes, filter_overlap=True, deduplicate=True)
        finally:
            stream.close()
            raw_file.close()
        manifest[split] = stats
        if split == "valid":
            validation_hashes = hashes
        print(json.dumps(dict(completed_split=split, **stats)), flush=True)
    (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


class Windows:
    def __init__(self, root, split, verify=True):
        import torch
        self.torch = torch
        root = Path(root)
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.identity = digest(root / "manifest.json")
        self.length = self.manifest["context"]
        if verify:
            for name, expected in self.manifest[split]["files"].items():
                if digest(root / name) != expected:
                    raise ValueError(f"Corrupted data file: {name}")
            if digest(root / "tokenizer.json") != self.manifest["tokenizer_sha256"]:
                raise ValueError("Tokenizer hash mismatch")
        self.tokens = np.memmap(root / f"{split}.bin", dtype="<u4", mode="r")
        self.windows = np.load(root / f"{split}.windows.npy", mmap_mode="r")

    def __len__(self):
        return len(self.windows)

    def batch(self, indices):
        x = np.full((len(indices), self.length), EOS, dtype=np.int64)
        y = np.full_like(x, IGNORE)
        for row, i in enumerate(indices):
            offset, n = self.windows[i]
            x[row, :n] = self.tokens[offset:offset+n]
            y[row, :n] = self.tokens[offset+1:offset+n+1]
        return self.torch.from_numpy(x), self.torch.from_numpy(y)
