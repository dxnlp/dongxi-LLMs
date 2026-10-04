"""Chapter 15 distribution distillation, inference selection and release audits."""
from collections import Counter
import hashlib

import torch
from torch.nn import functional as F


def distillation_loss(student_logits, teacher_logits, temperature=1., scale=True):
    """Mean forward KL(teacher || student); teacher detached; T² optionally retained."""
    if temperature <= 0 or student_logits.shape != teacher_logits.shape:
        raise ValueError("Positive temperature and matching vocabulary required")
    logp = (student_logits/temperature).log_softmax(-1)
    logq = (teacher_logits.detach()/temperature).log_softmax(-1)
    kl = (logq.exp()*(logq-logp)).sum(-1).mean()
    return kl*temperature**2 if scale else kl


def distill_distribution(updates=80, temperature=2., seed=2628):
    """Actual SGD on three student logits; intentionally isolates target geometry."""
    teacher = torch.tensor([[2., .5, -1.]], dtype=torch.float64)
    student = torch.nn.Parameter(torch.zeros_like(teacher))
    optimizer = torch.optim.SGD([student], lr=.5)
    history = []
    for step in range(updates+1):
        loss = distillation_loss(student, teacher, temperature)
        history.append({"update": step, "scaled_loss": float(loss.detach()),
                        "student_p": student.detach().softmax(-1)[0].tolist(),
                        "teacher_p": teacher.softmax(-1)[0].tolist()})
        if step < updates:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {"seed": seed, "temperature": temperature, "updates": updates,
            "objective": "T² forward KL(teacher_T || student_T)", "history": history}


def majority_vote(answers):
    """Stable tie rule: earliest candidate among equally frequent parsed answers."""
    if not answers:
        raise ValueError("At least one parsed answer required")
    counts = Counter(answers)
    largest = max(counts.values())
    return next(answer for answer in answers if counts[answer] == largest)


def inference_comparison(seed=2628, trials=1200, max_n=16):
    """Original finite simulator; oracle best-of-N is labeled an upper-bound diagnostic.

    Three answer classes: correct (0), repeated wrong (1), other wrong (2).
    No inference conclusion about an LLM follows from these categorical draws.
    """
    generator = torch.Generator().manual_seed(seed)
    distribution = torch.tensor([.45, .40, .15], dtype=torch.float64)
    draws = torch.multinomial(distribution, trials*max_n, replacement=True,
                             generator=generator).reshape(trials, max_n)
    rows = []
    for n in (1, 2, 4, 8, 16):
        if n > max_n:
            continue
        candidates = draws[:, :n]
        votes = [majority_vote(row.tolist()) for row in candidates]
        oracle = (candidates == 0).any(-1)
        # Scores reward the frequent wrong answer; a fixed imperfect ranker.
        scores = torch.tensor([.7, .9, .2])[candidates]
        selected = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        rows.append({"n": n, "candidate_token_budget": n*8,
                     "majority_accuracy": sum(v == 0 for v in votes)/trials,
                     "proxy_best_n_accuracy": float((selected == 0).double().mean()),
                     "oracle_any_correct": float(oracle.double().mean()),
                     "oracle_iid_formula": 1-(1-.45)**n})
    return {"mode": "categorical Monte Carlo simulation", "seed": seed,
            "trials": trials, "single_sample_correct_probability": .45, "rows": rows}


def validate_genealogy(cards):
    """Validate IDs, parent existence, cycles, required identities and frozen eval hash.

    These structural checks do not verify claimed scores or quality of the panel.
    """
    required = ("id", "parent", "weights_sha256", "tokenizer_sha256", "data_sha256",
                "evaluation_sha256", "objective", "evidence")
    mapping = {}
    for card in cards:
        missing = [key for key in required if key not in card]
        if missing:
            raise ValueError(f"Missing card fields: {missing}")
        if card["id"] in mapping:
            raise ValueError("Duplicate checkpoint ID")
        mapping[card["id"]] = card
    for key in mapping:
        seen, cursor = set(), key
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Checkpoint genealogy contains a cycle")
            if cursor not in mapping:
                raise ValueError("Unknown parent checkpoint")
            seen.add(cursor)
            cursor = mapping[cursor]["parent"]
    hashes = {card["evaluation_sha256"] for card in cards}
    if len(hashes) != 1:
        raise ValueError("Comparison does not use one frozen evaluation identity")
    return True


def release_gate(evidence):
    """A preparation gate; passing this function never publishes anything."""
    checks = ("sources_checked", "math_checked", "cpu_tests_pass",
              "notebooks_execute", "licenses_recorded", "claim_sources_linked",
              "limitations_written", "genealogy_valid", "heldout_frozen")
    return {"ready": all(evidence.get(key) is True for key in checks),
            "missing": [key for key in checks if evidence.get(key) is not True],
            "external_publication": "requires a separate user instruction"}


def original_card_fixture():
    """Synthetic documentation fixture, deliberately NOT an actual trained lineage."""
    digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
    cards = []
    for identifier, parent, objective in (("toy-base", None, "next-token"),
                                          ("toy-sft", "toy-base", "assistant NLL"),
                                          ("toy-rl", "toy-sft", "GRPO")):
        cards.append({"id": identifier, "parent": parent,
                      "weights_sha256": digest(identifier), "tokenizer_sha256": digest("toy-tokens"),
                      "data_sha256": digest(identifier+"-data"),
                      "evaluation_sha256": digest("frozen-original-toy-panel-v1"),
                      "objective": objective, "evidence": "synthetic structural fixture"})
    return cards
