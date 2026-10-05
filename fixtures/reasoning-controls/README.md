# Original reasoning-control fixtures

Authored for DXI-04; no external corpus or copied benchmark items.

- `control_items.json`:16 structured arithmetic-predicate prompts. Only the
  four sum-positive training rows enter sampled decoder or lookup updates.
- `math_items.json`:20 original natural-language arithmetic, linear-equation
  and two-step quantity prompts. Exact references are independently validated;
  this is a frozen evaluation instrument, not text consumed by the tiny decoder.
- `protocol.json`:unexecuted base/raw and instruct/chat/thinking matrix and
  predeclared output-budget interventions for separately approved local models.

Each mathematical problem has one source group. Splits do not share groups or
problem identities. Unseen-template rows also have new source problems, so they
do not isolate a pure template effect. Multi-step math and odd-sum controls are
entirely withheld task families. Correct final answers do not verify rationales.

The CPU decoder consumes `[BOS, instruction, numeral_a, numeral_b]`, not this
English prose. It is a transparent constructive control, not a natural-language
reasoning benchmark. Training permits answer0/1 at its first response position,
then answer0/1/EOS; the latter stopping action is sampled, not supplied free.
Every evaluated path retains its raw tokens, grading and stopping. All training
updates retain metrics; only the first and last rollout groups additionally
archive every sampled token, actual grammar-conditional likelihood and advantage.
The intermediate training paths are reproducible from seeds, not fully archived.
