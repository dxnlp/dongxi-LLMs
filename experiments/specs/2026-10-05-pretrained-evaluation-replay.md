# Real checkpoint evaluation through the frozen response ledger

This addresses DXI-02's missing genuine pretrained responses and independent
behavioral review. Use the existing15-item development grading/regression
instrument unchanged, not the campaign's20-item reasoning publication panel or
assistant120-item held-out test. It includes fractions, sets, intervals, units,
JSON, exact text, benign response, disagreement, formatting and unsupported
output categories. This small instrument is not a broad capability/safety test.

Freeze one actual tokenizer/interface contract before inference, with the exact
saved native instruction template, template-default thinking semantics, no
extra special-token insertion, deterministic greedy decoding, seed1010,
64new-token cap, context512 and one attempt per item. Separate151643 EOS from
151645 message-end;151643 is padding. Score decoded text without the terminal
stop, retaining actual raw IDs/stop/likelihood/support/cost/error records.

Evaluate the acquired pinned0.6B Base and the disposable20-update full-SFT
export on the same contract. Each generation child has a900-second independent
deadline and25GiB sampled host reserve, local-only BF16/eager execution and no
download. Maximum15attempts/960new-token slots per child; the actual uncached
forward work and elapsed cost are separately retained. Exports/checkpoint bytes
are hashed by the existing adapter; do not substitute declaration for identity.

Prediction: the adapter produces complete auditable coverage, but neither
checkpoint is assumed to pass task/termination/regression criteria. Failed and
truncated records stay in the ledger. Replay grading without a model, compare
source-group uncertainty and export evidence-limited cards. An independent
reviewer checks every raw answer and explicitly audits benign/disagreement
rubrics for marker-based false positives/negatives. Negative quality can satisfy
a valid evaluation; missing records, changed interfaces or invented review cannot.

This does not designate the20-update profile as the selected400-update parent,
backfill campaign publication rows, train on these cases, or certify a general
assistant. Goal-aligned routine evaluation follows the approved measured profile;
learner Day9 remains unchanged. No external API/service, Git publication or
Mac/hosted result is authorized or claimed.
