"""Original programmatic teacher journals, frozen selection and tiny sequence SFT.

No model/data acquisition or service. Wrong well-formed candidates survive the
format filter. Selection never reads held-out references or teacher mode labels.
"""
import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import torch
from torch import nn

from .decoder_lab import DecoderConfig, TinyDecoder
from .instruction_data_lab import encode_messages, collate, SPECIAL, VOCAB, ID_TO_WORD
from .run_identity import canonical_hash, file_digest
from .sft_lab import token_loss_sum

MODES = ["correct", "wrong_same_length", "wrong_verbose", "empty", "overlength",
         "unsupported", "no_end", "error_then_duplicate"]
COLORS = ["red", "blue", "green", "yellow"]


def derived_seed(*parts):
    return int(canonical_hash(list(parts))[:15], 16)


def parse_task(prompt):
    words = prompt.split()
    if words and words[0] == "please":
        words = words[1:]
    if len(words) not in (3, 4) or words[0] not in ("copy", "reverse") or any(w not in COLORS for w in words[1:]):
        raise ValueError("Only declared two/three-color copy/reverse tasks")
    return words[0], words[1:]


def task_answer(prompt):
    operation, words = parse_task(prompt)
    return " ".join(words if operation == "copy" else words[::-1])


def prefix_ids(item):
    conversation = encode_messages([
        {"role": "system", "content": "short answer"},
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": "red"}])
    return conversation.ids[:-2]


def validate_protocol(obj):
    obj = copy.deepcopy(obj)
    if obj.get("schema") != "dongxi-teacher-data-v1" or not obj.get("content_terms"):
        raise ValueError("Original schema and content terms required")
    teacher, student = obj["teacher"], obj["student"]
    if teacher["kind"] != "programmatic" or teacher["modes"] != MODES or teacher["samples_per_prompt"] != 8 or teacher["max_error_retries"] != 1:
        raise ValueError("Frozen teacher schedule required")
    if obj["filter"]["require_correctness"] is not False or obj["filter"]["require_end"] is not True or obj["filter"]["max_final_words"] != 6:
        raise ValueError("Filter is format/stop, not correctness")
    if not 1 <= student["updates"] <= 200 or student["dtype"] != "float64" or student["max_new_tokens"] != 6 or student["temperature"] != 1.0:
        raise ValueError("Bounded frozen CPU student settings required")
    ids, prompts, encodings, groups = set(), set(), set(), {}
    for item in obj["items"]:
        if not item["id"] or item["id"] in ids or item["split"] not in ("train", "dev", "test", "control"):
            raise ValueError("Unique item IDs and declared splits required")
        ids.add(item["id"])
        normalized = " ".join(item["prompt"].split())
        encoded = tuple(prefix_ids(item))
        if normalized in prompts or encoded in encodings:
            raise ValueError("Normalized prompt / actual student-ID collision")
        prompts.add(normalized); encodings.add(encoded)
        operation, words = parse_task(item["prompt"])
        group = canonical_hash([operation, words])
        if group in groups and groups[group] != item["split"]:
            raise ValueError("Underlying source group crosses splits")
        groups[group] = item["split"]
        item["source_group"] = group
        item["family"] = operation
        if item["reference"] != task_answer(item["prompt"]) or item["difficulty"] != ("two-word" if len(words) == 2 else "three-word"):
            raise ValueError("Authored reference/difficulty inconsistent with task")
    if not any(i["split"] == "train" for i in obj["items"]):
        raise ValueError("Training sources required")
    return obj


def load_protocol(root):
    return validate_protocol(json.loads((Path(root)/"fixtures/teacher-data/protocol.json").read_text()))


def freeze_contract(root, protocol):
    protocol = validate_protocol(protocol)
    files = ["src/dongxi_llms/teacher_data_lab.py", "src/dongxi_llms/sft_lab.py",
             "src/dongxi_llms/instruction_data_lab.py", "src/dongxi_llms/decoder_lab.py",
             "src/dongxi_llms/run_identity.py", "fixtures/teacher-data/protocol.json",
             "experiments/specs/2026-10-04-teacher-data.md"]
    contract = {"schema": "dongxi-teacher-attempt-contract-v1", "protocol": protocol,
                "source_sha256": {p: file_digest(Path(root)/p) for p in files},
                "interface": {"vocabulary": VOCAB, "special_ids": SPECIAL,
                    "serialization": "BOS SYSTEM short answer END USER prompt END ASSISTANT final END",
                    "supervision": "assistant body plus END; exactly one causal shift"},
                "teacher_identity": protocol["teacher"], "sampler_identity": "fixed slots; item/sample/retry keyed-v1",
                "verifier_identity": protocol["selection"]["version"],
                "content_terms": protocol["content_terms"]}
    contract["identity"] = canonical_hash(contract)
    return contract


def attempt_identity(contract, item_id, sample, retry):
    return canonical_hash([contract["identity"], item_id, sample, retry])


def attach_digest(row):
    row = copy.deepcopy(row)
    row["payload_sha256"] = canonical_hash(row)
    return row


def validate_digest(row):
    if not isinstance(row, dict) or row.get("payload_sha256") != canonical_hash({k:v for k,v in row.items() if k != "payload_sha256"}):
        raise ValueError("Changed/corrupt journal payload")


class TeacherJournal:
    """Single-writer append journal; no duplicate committed attempt records.

    Repeated uncommitted executions are visible, not an exactly-once claim.
    Explicit tail repair preserves bytes before truncating an uncommitted suffix.
    """
    def __init__(self, directory, contract, *, recover_tail=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.contract = contract
        if contract.get("identity") != canonical_hash({k:v for k,v in contract.items() if k != "identity"}):
            raise ValueError("Changed frozen teacher contract")
        path = self.directory/"contract.json"
        if path.exists():
            if json.loads(path.read_text()) != contract:
                raise ValueError("Existing teacher journal has a different frozen contract")
        elif any(self.directory.iterdir()):
            raise ValueError("Nonempty journal lacks frozen contract")
        else:
            with path.open("x") as handle:
                handle.write(json.dumps(contract, indent=2, allow_nan=False)+"\n"); handle.flush(); os.fsync(handle.fileno())
        self.lock = (self.directory/"writer.lock").open("a+b")
        try: fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.lock.close(); raise ValueError("Teacher journal already has a writer") from error
        try:
            self.events = self._read("events.jsonl", recover_tail)
            self.records = self._read("attempts.jsonl", recover_tail)
            self._validate_records()
        except BaseException:
            self.close(); raise

    def _validate_records(self):
        contract = self.contract
        if any(e.get("contract_id") != contract["identity"] for e in self.events):
            raise ValueError("Foreign execution event contract")
        seen = set()
        item_map = {i["id"]:i for i in contract["protocol"]["items"]}
        for row in self.records:
            validate_digest(row)
            item = item_map.get(row.get("item_id"))
            sample, retry = row.get("sample_index"), row.get("retry_index")
            if item is None or item["split"] != "train" or type(sample) is not int or not 0 <= sample < 8 or type(retry) is not int or not 0 <= retry <= 1:
                raise ValueError("Journal attempt coordinates are outside contract")
            if row.get("contract_id") != contract["identity"] or row.get("item_sha256") != canonical_hash(item) or row.get("attempt_id") != attempt_identity(contract,item["id"],sample,retry):
                raise ValueError("Journal attempt identity does not bind actual item/settings")
            if row["attempt_id"] in seen:
                raise ValueError("Duplicate committed attempt identity")
            seen.add(row["attempt_id"])
        self.by_id = {r["attempt_id"]:r for r in self.records}
        for row in self.records:
            if row["retry_index"]:
                parent = self.by_id.get(attempt_identity(contract,row["item_id"],row["sample_index"],0))
                if parent is None or parent["error"] is None:
                    raise ValueError("Retry requires a persisted failed parent attempt")

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock.fileno(),fcntl.LOCK_UN); self.lock.close()

    def _read(self, name, recover_tail):
        path = self.directory/name
        if not path.exists():
            return []
        data = path.read_bytes()
        if len(data) > 8*1024*1024:
            raise ValueError("Bounded journal exceeds8MiB")
        if data and not data.endswith(b"\n"):
            if not recover_tail:
                raise ValueError("Uncommitted partial tail; explicit preservation/recovery required")
            boundary = data.rfind(b"\n")+1
            index = 0
            while (self.directory/f"{name}.partial-tail-{index:03}.bin").exists(): index += 1
            backup = self.directory/f"{name}.partial-tail-{index:03}.bin"
            with backup.open("xb") as handle:
                handle.write(data[boundary:]); handle.flush(); os.fsync(handle.fileno())
            recovery = {"event":"partial_tail_preserved", "file":name, "backup":backup.name,
                        "bytes":len(data)-boundary, "sha256":file_digest(backup),
                        "reason":"Uncommitted non-newline suffix; complete corrupt records are not recoverable"}
            # Preserve recovery metadata before changing the original journal.
            with (self.directory/(backup.name+".json")).open("x") as handle:
                handle.write(json.dumps(recovery,sort_keys=True)+"\n"); handle.flush(); os.fsync(handle.fileno())
            with path.open("r+b") as handle:
                handle.truncate(boundary); handle.flush(); os.fsync(handle.fileno())
            data = data[:boundary]
        rows = []
        for line in data.splitlines():
            try: row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error: raise ValueError("Complete corrupt JSONL record; preserve and refuse") from error
            validate_digest(row)
            rows.append(row)
        return rows

    def _append(self, name, row):
        encoded = (json.dumps(row,sort_keys=True,allow_nan=False)+"\n").encode()
        with (self.directory/name).open("ab") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())

    def event(self, kind, **fields):
        row = attach_digest({"event":kind,"contract_id":self.contract["identity"], **fields})
        self._append("events.jsonl",row); self.events.append(row)

    def commit(self, row):
        validate_digest(row)
        if row["attempt_id"] in self.by_id:
            raise ValueError("Attempt already committed")
        self.records.append(row)
        try: self._validate_records()
        except BaseException:
            self.records.pop(); self._validate_records(); raise
        self._append("attempts.jsonl",row)


def programmatic_teacher(item, sample, retry):
    """Declared faults are actual executed branches, not prerecorded model scores."""
    mode = MODES[sample]
    words = task_answer(item["prompt"]).split()
    trace = item["prompt"]+" -> "+" ".join(words)
    final, stop = " ".join(words), "END"
    if mode == "wrong_same_length":
        final = " ".join(COLORS[(COLORS.index(w)+1)%4] for w in words)
    elif mode == "wrong_verbose": final = " ".join(words+["please"])
    elif mode == "empty": final = ""
    elif mode == "overlength": final = " ".join(words*4)
    elif mode == "unsupported": final = "☃"
    elif mode == "no_end": stop = "cap"
    elif mode == "error_then_duplicate" and retry == 0:
        raise RuntimeError("Declared local teacher fault before a response")
    return trace, final, stop


def execute_attempt(contract, item, sample, retry, teacher=programmatic_teacher):
    began = time.perf_counter()
    error = None
    try:
        trace, final, stop = teacher(item,sample,retry)
        if not isinstance(trace,str) or not isinstance(final,str) or stop not in ("END","cap"):
            raise ValueError("Malformed programmatic teacher response")
    except Exception as caught:
        trace, final, stop = "", "", "error"
        error = {"type":type(caught).__name__,"message":str(caught)}
    words = final.split()
    ids = [VOCAB[w] for w in words]+([SPECIAL["end"]] if stop == "END" else []) if all(w in VOCAB for w in words) else None
    return attach_digest({"schema":"dongxi-teacher-attempt-v1", "attempt_id":attempt_identity(contract,item["id"],sample,retry),
        "contract_id":contract["identity"], "item_id":item["id"], "item_sha256":canonical_hash(item),
        "source_group":item["source_group"], "split":item["split"], "difficulty":item["difficulty"],
        "prompt":item["prompt"], "sample_index":sample, "retry_index":retry,
        "attempt_seed":derived_seed(contract["protocol"]["teacher"]["seed"],item["id"],sample,retry),
        "teacher":contract["teacher_identity"], "sampler":contract["sampler_identity"],
        "actual_executor":teacher.__module__+":"+teacher.__qualname__,
        "verifier":contract["verifier_identity"], "content_terms":contract["content_terms"],
        "trace":trace,"final_answer":final,"raw_response":trace+"\nFinal: "+final if error is None else None,
        "final_token_ids":ids,"stop":stop,"error":error,
        "cost":{"serialized_trace_words":len(trace.split()),"serialized_final_words":len(words),
                "valid_symbolic_final_tokens":len(ids) if ids is not None else 0,
                "model_forward_calls":0,"api_calls":0,"elapsed_seconds":time.perf_counter()-began,
                "unit":"Executed programmatic text serialization; not LLM inference or billed units"}})


def collect(contract, directory, *, max_new_records=None, recover_tail=False, teacher=programmatic_teacher):
    journal = TeacherJournal(directory,contract,recover_tail=recover_tail)
    try:
        added = 0
        for item in contract["protocol"]["items"]:
            if item["split"] != "train": continue
            for sample in range(8):
                for retry in range(2):
                    if retry and journal.by_id[attempt_identity(contract,item["id"],sample,0)]["error"] is None: break
                    attempt_id = attempt_identity(contract,item["id"],sample,retry)
                    if attempt_id in journal.by_id: continue
                    if max_new_records is not None and added >= max_new_records:
                        journal.event("collection_paused", committed=len(journal.records)); return journal
                    execution = sum(e.get("attempt_id") == attempt_id and e["event"] == "execution_started" for e in journal.events)
                    journal.event("execution_started",attempt_id=attempt_id,execution_index=execution,
                                  actual_executor=teacher.__module__+":"+teacher.__qualname__)
                    try: row = execute_attempt(contract,item,sample,retry,teacher)
                    except BaseException as error:
                        journal.event("execution_interrupted",attempt_id=attempt_id,execution_index=execution,
                                      error_type=type(error).__name__,cost_boundary="Uncommitted elapsed/output cost unknown")
                        raise
                    journal.commit(row); added += 1
        journal.event("collection_complete",committed=len(journal.records))
        return journal
    finally: journal.close()


def verifier_score(candidate, item):
    """Prompt-only executable training verifier; never reads references or mode."""
    return int(" ".join(candidate["final_answer"].split()) == task_answer(item["prompt"]))


def filter_pool(records, protocol):
    items = {i["id"]:i for i in protocol["items"]}
    accepted, audit, seen = [], [], set()
    for row in sorted(records,key=lambda r:(r["item_id"],r["sample_index"],r["retry_index"])):
        validate_digest(row)
        item = items.get(row["item_id"])
        reasons = []
        if item is None or item["split"] != "train" or row["source_group"] != item["source_group"] or row["prompt"] != item["prompt"]:
            reasons.append("source_group_or_split_leakage")
        if row["error"] is not None: reasons.append("teacher_error")
        final = row["final_answer"]
        if not isinstance(final,str): reasons.append("malformed_answer"); words = []
        else: words = final.split()
        if not words: reasons.append("empty_answer")
        if len(words) > protocol["filter"]["max_final_words"]: reasons.append("overlength_answer")
        if row["final_token_ids"] is None or any(w not in VOCAB for w in words): reasons.append("unsupported_answer")
        elif row["final_token_ids"] != [VOCAB[w] for w in words]+([SPECIAL["end"]] if row["stop"] == "END" else []):
            reasons.append("malformed_answer_ids")
        if row["stop"] != "END": reasons.append("missing_END")
        key = canonical_hash([row["item_id"],words])
        if not reasons and key in seen: reasons.append("duplicate_candidate")
        score = None
        if not reasons:
            seen.add(key)
            score = verifier_score(row,item)
            accepted.append({"attempt_id":row["attempt_id"],"attempt_sha256":row["payload_sha256"],
                "sample_index":row["sample_index"],"retry_index":row["retry_index"],
                "item_id":row["item_id"],"source_group":row["source_group"],"difficulty":item["difficulty"],
                "family":item["family"],
                "prompt":item["prompt"],"final_answer":final,"supervised_tokens":len(words)+1,
                "score":score})
        audit.append({"attempt_id":row["attempt_id"],"item_id":row["item_id"],"source_group":row["source_group"],
                      "difficulty":row["difficulty"],"accepted":not reasons,"reasons":reasons,
                      "verifier_score":score,"final_words":len(words),"retry_index":row["retry_index"]})
    pool = {"schema":"dongxi-frozen-candidate-pool-v1","candidates":accepted,"audit":audit,
            "reason_counts":dict(Counter(reason for r in audit for reason in r["reasons"])),
            "candidate_correctness":dict(Counter(str(r["score"]) for r in accepted)),
            "difficulty_audit":{d:{"attempts":sum(r["difficulty"] == d for r in audit),
                "accepted":sum(r["difficulty"] == d and r["accepted"] for r in audit),
                "rejections":dict(Counter(reason for r in audit if r["difficulty"] == d for reason in r["reasons"]))}
                for d in sorted({r["difficulty"] for r in audit})},
            "cost":{"attempts":len(records),"retry_attempts":sum(r["retry_index"] > 0 for r in records),
                    "serialized_words":sum(r["cost"]["serialized_trace_words"]+r["cost"]["serialized_final_words"] for r in records),
                    "actual_elapsed_seconds":sum(r["cost"]["elapsed_seconds"] for r in records),
                    "model_forward_calls":0,"api_calls":0}}
    pool["identity"] = canonical_hash(pool)
    return pool


def selection_summary(rows, train_items):
    prompt_ids = {r["item_id"] for r in rows}
    return {"samples":len(rows),"prompts":len(prompt_ids),"train_prompts":len(train_items),
        "source_groups":len({r["source_group"] for r in rows}),"families":dict(Counter(r["family"] for r in rows)),
        "prompt_coverage":len(prompt_ids)/len(train_items),"supervised_tokens":sum(r["supervised_tokens"] for r in rows),
        "correct":sum(r["score"] for r in rows),"difficulty":dict(Counter(r["difficulty"] for r in rows)),
        "length_counts":dict(Counter(str(r["supervised_tokens"]) for r in rows)),
        "excluded_prompt_ids":sorted({i["id"] for i in train_items}-prompt_ids)}


def select_datasets(pool, protocol):
    if pool["identity"] != canonical_hash({k:v for k,v in pool.items() if k != "identity"}):
        raise ValueError("Frozen candidate pool changed")
    item_map = {i["id"]:i for i in protocol["items"]}
    grouped = defaultdict(list)
    for row in pool["candidates"]:
        item = item_map[row["item_id"]]
        if item["split"] != "train" or row["score"] != verifier_score(row,item):
            raise ValueError("Selection scorer or train-only source contract changed")
        grouped[row["item_id"]].append(row)
    train = [i for i in protocol["items"] if i["split"] == "train"]
    arms = {"top":[],"random":[],"length_random":[]}
    unavailable, strata = [], {}
    for item in sorted(train,key=lambda i:i["id"]):
        candidates = sorted(grouped[item["id"]],key=lambda r:(r["sample_index"],r["retry_index"]))
        if not candidates: raise ValueError("No accepted candidate for a training prompt")
        top = min(candidates,key=lambda r:(-r["score"],r["supervised_tokens"],r["attempt_id"]))
        arms["top"].append(top)
        rng = random.Random(derived_seed(protocol["selection"]["seed"],item["id"],"random"))
        arms["random"].append(rng.choice(candidates))
        eligible = [r for r in candidates if r["supervised_tokens"] == top["supervised_tokens"]]
        strata[item["id"]] = {"eligible":len(eligible),"alternatives_to_top":len(eligible)-1,
                              "score_values":sorted({r["score"] for r in eligible})}
        if not eligible: unavailable.append(item["id"])
        else:
            rng = random.Random(derived_seed(protocol["selection"]["seed"],item["id"],"length_random"))
            arms["length_random"].append(rng.choice(eligible))
    global_rows = sorted(pool["candidates"],key=lambda r:(-r["score"],r["supervised_tokens"],r["attempt_id"]))
    result = {"pool_id":pool["identity"],"seed":protocol["selection"]["seed"],"arms":arms,
        "summaries":{a:selection_summary(r,train) for a,r in arms.items()},
        "length_stratum_unavailable":unavailable,
        "length_strata":strata,
        "global":{str(k):{"attempt_ids":[r["attempt_id"] for r in global_rows[:k]],
                         **selection_summary(global_rows[:k],train)} for k in protocol["selection"]["global_counts"]}}
    result["identity"] = canonical_hash(result)
    return result


def make_student(seed, protocol):
    s = protocol["student"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TinyDecoder(DecoderConfig(vocab=max(ID_TO_WORD)+1,width=s["width"],heads=s["heads"],
            kv_heads=s["heads"],head_dim=s["width"]//s["heads"],layers=s["layers"],hidden=s["hidden"],
            max_length=s["max_length"])).double()


def state_digest(model):
    return canonical_hash({k:{"shape":list(v.shape),"dtype":str(v.dtype),"values":v.detach().cpu().tolist()}
                           for k,v in model.state_dict().items()})


def training_batch(rows):
    return collate([encode_messages([{"role":"system","content":"short answer"},
        {"role":"user","content":r["prompt"]},{"role":"assistant","content":r["final_answer"]}]) for r in rows])


def fit_student(rows, protocol, seed, *, steps=None):
    settings = protocol["student"]
    steps = settings["updates"] if steps is None else steps
    if not rows or not 1 <= steps <= settings["updates"]: raise ValueError("Bounded nonempty training required")
    model = make_student(seed,protocol)
    initial_state = state_digest(model)
    batch = training_batch(rows)
    optimizer = torch.optim.AdamW(model.parameters(),lr=settings["learning_rate"],weight_decay=settings["weight_decay"])
    history, first = [], None
    began = time.perf_counter()
    for update in range(steps):
        optimizer.zero_grad(set_to_none=True)
        total, count = token_loss_sum(model(batch["input_ids"]),batch["labels"])
        loss = total/count
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(),settings["gradient_clip"])
        if not torch.isfinite(loss) or not torch.isfinite(gradient): raise RuntimeError("Nonfinite SFT state")
        if first is None:
            first = {n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        history.append({"update":update+1,"loss":float(loss.detach()),"gradient_norm":float(gradient),
                        "valid_supervised_tokens":count})
    with torch.no_grad():
        final, count = token_loss_sum(model(batch["input_ids"]),batch["labels"])
    return model, {"seed":seed,"updates":steps,"history":history,"first_backward":first,
        "initial_state_sha256":initial_state,"final_state_sha256":state_digest(model),
        "parameters":sum(p.numel() for p in model.parameters()),"final_train_nll":float(final/count),
        "valid_supervised_tokens_per_update":count,"valid_supervised_token_presentations":count*steps,
        "padded_forward_positions":batch["input_ids"].numel()*steps,"actual_elapsed_seconds":time.perf_counter()-began,
        "state_boundary":"Actual in-memory state digest; no checkpoint export or resume claim"}


@torch.no_grad()
def evaluate_student(model, protocol, seed, arm):
    rows = []
    s = protocol["student"]
    for item in protocol["items"]:
        for mode in ("greedy","sampled"):
            for sample in range(1 if mode == "greedy" else s["sampled_responses"]):
                stream_seed = derived_seed(seed,item["id"],mode,sample)
                rng = torch.Generator().manual_seed(stream_seed)
                prompt = prefix_ids(item)
                ids = torch.tensor([prompt])
                tokens, error, positions, forwards = [], None, 0, 0
                began = time.perf_counter()
                try:
                    for _ in range(s["max_new_tokens"]):
                        positions += ids.numel(); forwards += 1
                        logits = model(ids)[0,-1]
                        token = int(logits.argmax()) if mode == "greedy" else int(torch.multinomial((logits/s["temperature"]).softmax(0),1,generator=rng))
                        tokens.append(token)
                        if token == SPECIAL["end"]: break
                        ids = torch.cat((ids,torch.tensor([[token]])),dim=1)
                except Exception as caught: error = {"type":type(caught).__name__,"message":str(caught)}
                natural = bool(tokens and tokens[-1] == SPECIAL["end"] and error is None)
                body = tokens[:-1] if natural else tokens
                invalid_special = any(t in SPECIAL.values() for t in body)
                expected = [VOCAB[w] for w in item["reference"].split()]+[SPECIAL["end"]]
                rows.append({"arm":arm,"seed":seed,"item_id":item["id"],"source_group":item["source_group"],
                    "split":item["split"],"difficulty":item["difficulty"],"family":item["family"],
                    "mode":mode,"sample":sample,"stream_seed":stream_seed,"input_ids":prompt,"generated_ids":tokens,
                    "raw_text":" ".join(ID_TO_WORD[t] for t in tokens),"reference":item["reference"],
                    "correct":error is None and tokens == expected,"valid_format":natural and bool(body) and not invalid_special,
                    "natural_END":natural,"truncated":error is None and not natural,
                    "stop":"error" if error else "END" if natural else "cap","error":error,
                    "cost":{"valid_generated_tokens":len(tokens),"attempted_forward_calls":forwards,
                            "full_prefix_forward_positions":positions,"elapsed_seconds":time.perf_counter()-began,
                            "unit":"Actual individual CPU full-prefix forwards; not API billing"}})
    summaries = {}
    for split in ("train","dev","test","control"):
        for mode in ("greedy","sampled"):
            selected = [r for r in rows if r["split"] == split and r["mode"] == mode]
            if selected:
                summaries[split+"/"+mode] = {"n":len(selected),**{k:sum(r[k] for r in selected)/len(selected)
                    for k in ("correct","valid_format","natural_END","truncated")}}
    return {"rows":rows,"summaries":summaries,"identity_without_measured_cost":canonical_hash([
        {k:v for k,v in r.items() if k != "cost"} for r in rows])}


def run_reference(root, journal_path):
    protocol = load_protocol(root)
    contract = freeze_contract(root,protocol)
    # The actual reference explicitly exercises ordinary partial collection.
    first = collect(contract,journal_path,max_new_records=9)
    first_ids = [r["attempt_id"] for r in first.records]
    resumed = collect(contract,journal_path)
    again = collect(contract,journal_path)
    if [r["attempt_id"] for r in resumed.records] != [r["attempt_id"] for r in again.records]:
        raise RuntimeError("Completed resumption duplicated attempts")
    pool = filter_pool(again.records,protocol)
    selection = select_datasets(pool,protocol)
    journal = Path(journal_path)
    for name, value in (("pool.json",pool),("selection.json",selection)):
        path = journal/name
        if path.exists():
            if json.loads(path.read_text()) != value: raise ValueError("Frozen candidate/selection identity changed")
        else:
            with path.open("x") as handle:
                handle.write(json.dumps(value,indent=2,allow_nan=False)+"\n"); handle.flush(); os.fsync(handle.fileno())
    runs = []
    failures = []
    for seed in protocol["student"]["seeds"]:
        initial = make_student(seed,protocol)
        baseline = evaluate_student(initial,protocol,seed,"baseline")
        for arm in ("top","random","length_random"):
            try:
                if arm == "length_random" and selection["length_stratum_unavailable"]:
                    raise ValueError("Length-stratified comparison unavailable; no fallback")
                model, fit = fit_student(selection["arms"][arm],protocol,seed)
                if fit["initial_state_sha256"] != state_digest(initial): raise RuntimeError("Unpaired student initialization")
                after = evaluate_student(model,protocol,seed,arm)
                runs.append({"arm":arm,"seed":seed,"dataset_id":canonical_hash(selection["arms"][arm]),
                             "fit":fit,"baseline":baseline,"after":after})
            except Exception as error:
                failures.append({"arm":arm,"seed":seed,"type":type(error).__name__,"message":str(error)})
    return {"contract_id":contract["identity"],"protocol":protocol,"pool":pool,"selection":selection,
        "journal":{"path":str(journal),"first_committed_ids":first_ids,"attempts":len(again.records),
                   "unique_attempts":len(again.by_id),"events":again.events,
                   "attempts_sha256":file_digest(journal/"attempts.jsonl"),
                   "repeat_complete_resume_added_records":len(again.records)-len(resumed.records),
                   "execution_starts":sum(e["event"] == "execution_started" for e in again.events),
                   "resumption_boundary":"No duplicate committed records; uncommitted physical execution can repeat with visible starts"},
        "runs":runs,"failures":failures,
        "limits":["Programmatic teacher, not pretrained/API/human judgments or LLM generation costs",
                  "Shared vocabulary and task forms; held-out combinations/polite prefix only",
                  "Trace retained but excluded from supervision; no rationale-faithfulness claim",
                  "Matched examples/prompts; primary random token exposure may differ",
                  "Tiny random sequence student; no larger-model transfer or universal selection-method ranking"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--journal",type=Path,required=True)
    args = parser.parse_args()
    if args.report.exists(): parser.error("Unused report required")
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(1)
    before = freeze_contract(root,load_protocol(root))
    began = time.perf_counter()
    results = run_reference(root,args.journal)
    after = freeze_contract(root,load_protocol(root))
    if before != after: raise RuntimeError("Teacher/SFT inputs or source changed during measurement")
    report = {"schema_version":1,"package":"DXI-09","date_utc":datetime.now(timezone.utc).isoformat(),
        "command":list(sys.orig_argv),"source_sha256":before["source_sha256"],
        "environment":{"python":platform.python_version(),"torch":torch.__version__,"platform":platform.platform(),
                       "prefix":sys.prefix,"executable":sys.executable,"device":"cpu","dtype":"float64","threads":torch.get_num_threads()},
        "results":results,"actual_elapsed_seconds":time.perf_counter()-began}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open("x") as handle: handle.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"report":str(args.report),"teacher_attempts":results["journal"]["attempts"],
                      "student_runs":len(results["runs"]),"failures":results["failures"],"seconds":report["actual_elapsed_seconds"]}))
    if results["failures"]: raise SystemExit(1)


if __name__ == "__main__": main()
