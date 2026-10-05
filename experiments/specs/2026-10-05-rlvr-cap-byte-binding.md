# Bind RLVR limits to the bytes that were parsed

Declared before the follow-up test on2026-10-05. Preserve the completed37-test
run03 and its source identities. Source review found that initial bounded cap
parsing preceded identity collection, whose later hash could describe a file
changed after that parse. The numerical recipe and all23 caps remain unchanged.

Bind the SHA256 of the exact bounded initially parsed bytes, retain it in the
invocation evidence and compare a second bounded no-follow regular-file read
before tokenizer/model loading. Reject a change rather than adopting new caps.
The scientific input role must use that parsed-byte digest; final source/input
rehashing still applies. This is trusted-local change detection, not protection
against an adversarial writer that controls the process and its expectations.

Add a regression that changes the cap file during mocked identity collection
and proves refusal before tokenizer/model allocation while the opened evidence
journal retains the known parsed identity. Save a new exclusive run04 containing
this check, the37 existing controls and original failures. No API/model/GPU
acquisition, shared-ledger edit or numerical intervention is authorized here.
