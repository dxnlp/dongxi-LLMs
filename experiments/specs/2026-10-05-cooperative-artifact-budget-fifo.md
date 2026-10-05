# Nonblocking regular file refusal

Saved before follow-up measurements. A source review found that the snapshot
reader and artifact digest helper opened a path read-only before checking its
type. A FIFO could therefore wait for a peer before the intended refusal. The
earlier 67 and 69 test evidence did not cover this open-before-stat condition
and remains unchanged.

Add `O_NONBLOCK` to snapshot regular-file opens, artifact journal opens and
artifact digest opens. Ordinary regular-file semantics and snapshot numerical
recipes remain unchanged. Test authored FIFO header and payload paths, a FIFO
journal and a FIFO digest target in child processes capped at three seconds.
Require explicit regular-file refusal without a peer or deserialization. Keep
all subprocess commands/exits and the fixed byte/entry controls. This creates
no quota, cgroup, service, pretrained model, GPU work, install or Git operation.
