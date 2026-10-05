# Nonblocking regular file input for DPO work limits

This source-hardening protocol precedes its new controls. The earlier cumulative
work-budget run remains intact with its original source identity. No training
recipe, objective, mask, budget dimension, normal cap or recovery contract is
retuned. The only mechanism change is the initial production CLI limits-file
read, which previously used a blocking Path.open before inspecting file type.

Open the final cap-file component with O_NONBLOCK and O_NOFOLLOW. Walk directory
ancestors using retained no-follow directory descriptors so a symlinked ancestor
is also rejected. Require a regular file before reading, inspect its declared
size and use a bounded read of at most 64 KiB plus one refusal byte. Close every
descriptor on acceptance and rejection. Do not consume a FIFO, device or socket
as if it were a JSON limits file. This is a trusted-local preflight gate, not a
claim of hostile filesystem containment for all later provenance-file accesses.

Add original focused controls for a normal file, an oversized regular file,
directory, final symlink, ancestor symlink and a FIFO with no peer. Exercise the
actual CLI FIFO path in a separate process with a ten-second test deadline; it
must reject before output creation, hardware checking or model work. Retain raw
exit/output. The existing frozen 61-test panel must still pass, plus the new
controls. Run a new exclusive cumulative-work collector directory and compare
source identities before and after. Preserve run-01 and its Markdown report;
write a separate hardening report rather than changing historical hashes.

The numerical acceptance remains exactly the original CPU FP32 random seed-1818
six-update DPO recipe, actual activation checkpointing, same original reference,
Adam/RNG/history and update-3 fresh-process recovery with later spent work.
Pretrained/CUDA/GPU/physical-quota/other-runner/campaign outcomes remain outside
this bounded task. No acquisitions, installs, services or Git actions occur.
