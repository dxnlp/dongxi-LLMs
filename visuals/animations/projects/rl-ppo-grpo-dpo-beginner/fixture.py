"""Toy numbers behind every quantity the beginner film shows, with self-checks.

Run:  python fixture.py   ->  writes fixture.json next to this file.

All values are illustrative, chosen so each idea can be shown with simple arithmetic.
None is measured from a trained model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- the wheel of chances
WORDS = ["Sydney", "Canberra", "Melbourne", "other"]
WHEEL = [0.40, 0.35, 0.15, 0.10]      # next-word chances after "The capital of Australia is"
EPS = 0.20                            # PPO clip range: one round moves a chance by <= 20 %


def nudge(shares, i, factor):
    """Scale word i's chance by `factor`; rescale the rest so the total stays 1."""
    new = shares[i] * factor
    rest = (1 - new) / (1 - shares[i])
    return [new if j == i else s * rest for j, s in enumerate(shares)]


# Toy practice rounds: the sampled word is "Sydney" (judged bad) or "Canberra" (judged good);
# each round's step is large enough to reach the clip fence, so every step is exactly x0.8 or x1.2.
ROUNDS = ["Sydney", "Canberra"] * 3

# ---------------------------------------------------------------- PPO: judge + coach
ANSWER = ["The", "capital", "of", "Australia", "is", "Sydney"]
COACH_GUESSES = [6.0, 6.0, 6.5, 6.5, 7.0, 7.0]   # coach's guess of the final score before each word
JUDGE_SCORE = 2.0                               # the judge's score for the finished answer (/10)

RAMBLE = ["Canberra", "is", "a", "city,", "a", "big", "city,", "a", "very,", "very", "big",
          "city…"]
RAMBLE_SCORES = [(1, 7), (4, 8), (8, 9), (12, 10)]   # (words shown, judge score): a judge that loves length

# ---------------------------------------------------------------- GRPO: a group of tries
QUESTION = "What is 17 × 3?"
TRIES = [["10 × 3 = 30", "7 × 3 = 21", "answer: 51"],
         ["10 × 3 = 30", "7 × 3 = 11", "answer: 41"],
         ["17 + 17 = 34", "34 + 17 = 51", "answer: 51"],
         ["20 × 3 = 60", "60 − 3 = 57", "answer: 57"]]
KEY = 51

# ---------------------------------------------------------------- DPO: one saved comparison
REF_LIKING = {"A": 0.20, "B": 0.30}   # how likely the original model is to write A / B (toy)
DPO_STEPS = 8
DPO_TARGET_MARGIN = math.log(9)       # A ends 3x more likely than the original, B 3x less


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def dpo_path(eta, steps=DPO_STEPS):
    """Gradient descent on -log sigmoid(m), where m = log(A's ratio) - log(B's ratio).

    The toy moves both answers symmetrically: A's ratio = e^(m/2), B's ratio = e^(-m/2),
    and each step's size is proportional to sigmoid(-m), the push strength.
    """
    m, rows = 0.0, []
    for _ in range(steps + 1):
        rows.append(dict(margin=m, ratio_a=math.exp(m / 2), ratio_b=math.exp(-m / 2),
                         push=sigmoid(-m), loss=math.log1p(math.exp(-m))))
        m += eta * sigmoid(-m)
    return rows


def solve_eta(target=DPO_TARGET_MARGIN):
    lo, hi = 0.01, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dpo_path(mid)[-1]["margin"] < target else (lo, mid)
    return (lo + hi) / 2


def build():
    checks = {}
    checks["wheel_sums_to_one"] = math.isclose(sum(WHEEL), 1.0)

    i_s, i_c = WORDS.index("Sydney"), WORDS.index("Canberra")
    after = nudge(WHEEL, i_s, 1 - EPS)
    checks["fence_step_is_40_to_32"] = math.isclose(after[i_s], 0.32)
    checks["fence_step_keeps_total"] = math.isclose(sum(after), 1.0)
    rounds = [WHEEL]
    for w in ROUNDS:
        i = WORDS.index(w)
        rounds.append(nudge(rounds[-1], i, 1 - EPS if w == "Sydney" else 1 + EPS))
    # The clip bounds only the picked word's ratio; the others move so the total stays 1.
    checks["picked_word_moves_at_most_20_percent"] = all(
        abs(b[WORDS.index(w)] / a[WORDS.index(w)] - 1) <= EPS + 1e-12
        for w, a, b in zip(ROUNDS, rounds, rounds[1:]))
    checks["rounds_keep_total"] = all(math.isclose(sum(r), 1.0) for r in rounds)
    checks["canberra_ends_on_top"] = max(rounds[-1]) == rounds[-1][i_c]
    assert rounds[1] == after

    credits = [b - a for a, b in zip(COACH_GUESSES, COACH_GUESSES[1:])] + [
        JUDGE_SCORE - COACH_GUESSES[-1]]
    checks["credits_add_up_to_surprise"] = math.isclose(sum(credits),
                                                        JUDGE_SCORE - COACH_GUESSES[0])
    checks["sydney_gets_the_blame"] = min(credits) == credits[ANSWER.index("Sydney")] < 0
    checks["ramble_scores_rise_with_length"] = all(
        b[1] > a[1] and b[0] > a[0] for a, b in zip(RAMBLE_SCORES, RAMBLE_SCORES[1:]))

    def result(t):
        return int(t[-1].split(":")[1])
    scores = [1 if result(t) == KEY else 0 for t in TRIES]
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    grpo_credit = [s - mean for s in scores]
    checks["tries_arithmetic_as_labelled"] = [result(t) for t in TRIES] == [51, 41, 51, 57]
    checks["group_credits_sum_to_zero"] = math.isclose(sum(grpo_credit), 0.0, abs_tol=1e-12)

    eta = solve_eta()
    path = dpo_path(eta)
    last = path[-1]
    checks["dpo_ends_3x_and_one_third"] = (math.isclose(last["ratio_a"], 3.0, rel_tol=1e-9)
                                           and math.isclose(last["ratio_b"], 1 / 3, rel_tol=1e-9))
    checks["dpo_push_starts_half_ends_tenth"] = (math.isclose(path[0]["push"], 0.5)
                                                 and math.isclose(last["push"], 0.1, rel_tol=1e-9))
    checks["dpo_push_fades_each_step"] = all(b["push"] < a["push"] for a, b in zip(path, path[1:]))
    checks["dpo_loss_falls_from_log2"] = math.isclose(path[0]["loss"], math.log(2)) and all(
        b["loss"] < a["loss"] for a, b in zip(path, path[1:]))
    h = 1e-6
    fd = [(math.log1p(math.exp(-(r["margin"] + h))) - math.log1p(math.exp(-(r["margin"] - h))))
          / (2 * h) for r in path]
    checks["dpo_push_is_minus_loss_slope"] = all(math.isclose(g, -r["push"], abs_tol=1e-6)
                                                 for g, r in zip(fd, path))

    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    return dict(
        note="Illustrative toy values for the animation; not measurements of a trained model.",
        wheel=dict(words=WORDS, shares=WHEEL, eps=EPS, after_one_round=after,
                   rounds=ROUNDS, round_shares=rounds),
        ppo=dict(answer=ANSWER, coach_guesses=COACH_GUESSES, judge_score=JUDGE_SCORE,
                 credits=credits, ramble=RAMBLE, ramble_scores=RAMBLE_SCORES),
        grpo=dict(question=QUESTION, tries=TRIES, key=KEY, scores=scores, mean=mean,
                  std=std, credits=grpo_credit,
                  normalized_credits=[c / std for c in grpo_credit]),
        dpo=dict(ref_liking=REF_LIKING, eta=eta, steps=DPO_STEPS, path=path),
        models=dict(ppo=["student", "coach", "judge", "original"],
                    grpo=["student", "judge", "original"], dpo=["student", "original"]),
        checks=checks,
    )


if __name__ == "__main__":
    data = build()
    (HERE / "fixture.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["checks"], indent=2))
    print("after one round", [round(x, 4) for x in data["wheel"]["after_one_round"]])
    print("final round    ", [round(x, 4) for x in data["wheel"]["round_shares"][-1]])
    print("PPO credits    ", data["ppo"]["credits"])
    print("GRPO           ", data["grpo"]["scores"], data["grpo"]["mean"], data["grpo"]["credits"])
    print("DPO eta        ", round(data["dpo"]["eta"], 6))
    for r in data["dpo"]["path"]:
        print("   ", {k: round(v, 4) for k, v in r.items()})
