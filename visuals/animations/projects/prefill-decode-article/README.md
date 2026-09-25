# Prefill and Decode article loops

Approved `ANIM-PD-001`, Mac-only production of seven reviewed article concepts.
Same visual system as the KV article; no production labels inside the scenes.
Seven independent loops: lifecycle, parallel positions, first-token cache boundary,
two-step decode feedback, workload volume, chunk scheduling/placement, KV handoff.

```sh
uv run --project visuals/animations python visuals/animations/projects/prefill-decode-article/render.py --preview
uv run --project visuals/animations python visuals/animations/projects/prefill-decode-article/render.py
```

The exporter is reused read-only from `kv-article-loops/render.py`. Our validator
checks clip-specific events and article numerical invariants. Each output folder
contains 1080p MP4 (ignored by Git), 960px GIF, poster, half-size still and contact
sheet. Source/output hashes, commands and dependency versions are in metadata.

Frames express dependency and ordering, never measured compute/network durations.
Append operations preserve existing cache entries; fade-out is an editorial loop
reset. Handoff includes a transfer tail and assumes compatible generation state.
No Spark experiment, X upload, publication, commit or push is part of this task.
