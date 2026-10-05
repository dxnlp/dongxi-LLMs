# Cooperative artifact budget verification

This protocol precedes measurements for a new artifact ledger and an optional
shared snapshot hook. Read alongside `docs/PRODUCTION_SUPERVISOR_PLAN.md`.
Only original bounded filesystem and tensor-only CPU fixtures may run. No
filesystem quota, cgroup, service, installation, pretrained model, GPU, Git or
runner recipe changes are authorized. Existing recovery reports remain historical.

## Contract and scope

The ledger owns one exclusively created, private, UID-owned directory containing
direct regular-file artifacts. Absolute paths outside that root, nested paths,
symlinks, special files, unexpected entries and ownership changes are refused.
An open ledger has one cooperative writer under a nonblocking advisory lock.
This is not protection from hostile same-UID processes or arbitrary library and
descendant writes. A caller requiring a physical aggregate quota must fail closed
because this implementation has no verified physical backend.

Immutable identity includes campaign ID, root device/inode, a unique ledger ID,
maximum logical pathname bytes, maximum simultaneous reserved/file entries and
fixed journal byte capacity. The journal consumes that capacity and one entry
upfront, so accounting records cannot grow outside their own bound. Metadata
storage is conservative: a hard-linked staging/header pair counts twice by
pathname until the staging name is removed. This is not an ext4 allocated-block
or physical inode quota.

Every allocation is a fsynced append-only hash-chain event before file creation.
Reserve a whole bundle before opening any member; immutable caps include old
artifacts plus new payload, final marker and header staging coexistence. Bounded
exclusive streams refuse a crossing write or seek before touching excess bytes.
Successful close seals actual bytes/digest. Failed writes, fsyncs or journal
commits retain files and conservative reservations. Uncreated reservations may
be released only after proving absence; staging release follows exact owned
unlink and absence validation. Partial files are never silently deleted or freed.

Restore requires an independently retained identity and journal-prefix receipt,
validates the hash chain and physical inventory, then replays the whole current
journal rather than truncating to an older checkpoint. Changed caps/root/campaign,
rollback before the retained prefix, tail corruption, concurrent access and
untracked artifacts fail closed. New output names continue consuming the same
ledger. This cannot detect rollback to an old independently supplied expectation
or make creation of unrelated new campaign ledgers an aggregate quota.

## Shared snapshot hook

Keep save/load/inspect schemas and all no-hook numerical behavior unchanged.
`save_snapshot(..., artifact_budget=budget)` reserves the payload's `max_bytes`,
two `HEADER_LIMIT` capacities and three entries before its first artifact open.
Use only the ledger's private root and pre-reserved exclusive stream, link and
staging removal operations. On success seal actual final bytes and release the
removed staging name. On failure retain all known artifacts and reservations;
prior snapshots are never overwritten. The budget receipt is separate from the
unchanged checkpoint header and must be retained by a caller. There is no claim
that SFT/DPO/RLVR, logs or HF exports already call this optional hook.

## Frozen CPU controls

Use the existing isolated CPU interpreter and one thread with HF offline and
CUDA hidden. Pure byte fixtures use deterministic lengths of 0, 1, 8, 16, 64
and 256 bytes, not model data. Test small immutable caps with 16 KiB journal
capacity; snapshot fixtures use one original six-value FP32 tensor, `max_bytes`
4096, journal capacity 32 KiB, aggregate capacity 200000 bytes and entry cap 16.
Do not increase a failed test's declared production cap to hide refusal; separate
authored test envelopes may target a distinct boundary.

Require exact-cap admission and crossing-bundle refusal before any file or
serializer invocation; cumulative old/new/staging coexistence; entry refusal;
stream/seek refusal; partial-write and flush/fsync/link/journal failure retention;
restored reservations with a new output name; non-resettable caps; concurrent
lock refusal; symlink/ancestor/root replacement, hard-link, special-file,
unknown-entry, ownership, malformed JSON/schema/hash/duplicate-key and rollback
gates. Exercise the actual optional snapshot function, not only a counter class.
Compare its tensor/header semantics to no-hook save/load, preserve old shared
snapshot failure tests, and keep the final source frozen during collection.

## Evidence

Record every measured command, actual exit, stdout/stderr, source/spec/test/lock
hashes before and after, interpreter/platform, quantitative byte/entry receipts
and injected failure observations. Retain initial failures without rewriting
their evidence. The source gate establishes cooperative outputs covered by this
writer only. Pretrained Spark execution, total runner artifact coverage and
physical backend provisioning remain pending; all external outcomes stay null.
