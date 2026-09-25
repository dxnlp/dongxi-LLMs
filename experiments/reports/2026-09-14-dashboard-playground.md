# Final-model dashboard playground verification

User explicitly requested loading the trained model for custom prompts.
Loaded `outputs/day09-learning-01/update-014000.pt` into a separate inference
model, using tensor-only CPU-mapped loading and the hash-matched GPT-2 tokenizer.
No optimizer created, checkpoint changed, new training run, installation,
public upload, commit or push.

Service: `dongxigpt-playground-20260914.service`,127.0.0.1:8765,8G cgroup cap,
no cgroup swap. The known log-only service was stopped before binding the same
port. CUDA was idle before model loading; host available memory was118GiB.
Generation preserves a25GiB reserve, checks a60-second deadline between tokens,
permits only one request at a time and caps output at256 new tokens.

Verification:97 regression tests passed (exit0). Added request-bound and HTTP
JSON/custom-header/origin tests. JavaScript syntax parsed in V8. The live API
reported model loaded, checkpoint14000, context1024. Browser visual/interaction
QA was not performed; the updated page is opened for learner review.

Real API smoke request: "Once upon a time, a little rabbit found a golden key
in the forest." Temperature.8,seed909,max128 new tokens. Response:114 generated
tokens,stop_reason=eos,1.0193s. It began "He was very persistent and he wanted to
open it" and ended "The end." It also included semantic/grammatical errors
(opening a key; "It was enter a deep cave"). This demonstrates usable inference,
not reliable narrative coherence or a general latency benchmark.

The dashboard's chart selector does not change the inference checkpoint: the
prompt form explicitly labels its fixed run/update. User prompts/completions are
not written to logs by this service. It is intentionally local-only and must
be stopped before any new GPU training campaign.
