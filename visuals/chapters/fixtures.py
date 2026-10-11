"""Small, deterministic mechanisms for chapter GIFs; no language-model run.

The finite policy in ``hacking_history`` independently replays the float64
SGD recurrence in optimization_diagnostics_lab.hacking_experiment. All other
examples are constructed teaching fixtures, not observed model activations.
"""

from __future__ import annotations

import math
import random


def softmax(z):
    e = [math.exp(x - max(z)) for x in z]
    return [x / sum(e) for x in e]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [c * v[0] - s * v[1], s * v[0] + c * v[1]]


def ce(z, target=1):
    return -math.log(softmax(z)[target])


def ce_step(z, rate=.6, target=1):
    p = softmax(z)
    g = [x - float(i == target) for i, x in enumerate(p)]
    return [x - rate * d for x, d in zip(z, g)], g


CE_LOGITS = [1.1, -.3, .2]
EMBEDDINGS = [[.2, .4], [.8, -.1], [-.2, .7], [.1, -.3]]
HEAD = [[.8, -.2], [-.4, .6], [.3, .5]]
CONTEXT_WEIGHTS = [.4, .6]


def embedding_fixture():
    # Rows 1 and 2 are context. Only the next answer is a direct loss target.
    h = [sum(w * EMBEDDINGS[row][j] for w, row in zip(CONTEXT_WEIGHTS, [1, 2]))
         for j in range(2)]
    z = [dot(row, h) for row in HEAD]
    p = softmax(z)
    dz = [x - float(i == 1) for i, x in enumerate(p)]
    dh = [sum(d * row[j] for d, row in zip(dz, HEAD)) for j in range(2)]
    gradients = [[0., 0.], [.4 * x for x in dh], [.6 * x for x in dh], [0., 0.]]
    return {"hidden": h, "logits": z, "probabilities": p, "gradient": gradients}


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def preference_loss(scores):
    return -math.log(sigmoid(scores[0] - scores[1]))


def dpo_margin(p, ref=(.2, .1, .7), beta=.5):
    return beta * (math.log(p[0] / ref[0]) - math.log(p[1] / ref[1]))


def dpo_loss(z):
    return -math.log(sigmoid(dpo_margin(softmax(z))))


def ppo(ratio, advantage, epsilon=.2):
    clipped = max(1 - epsilon, min(1 + epsilon, ratio))
    return min(ratio * advantage, clipped * advantage)


POLICY_P = [.5, .3, .2]
REWARDS = [0., 1., 2.]


def policy_estimates(baseline):
    gs = [[(r - baseline) * (float(i == a) - POLICY_P[i]) for i in range(3)]
          for a, r in enumerate(REWARDS)]
    mean = [sum(p * g[i] for p, g in zip(POLICY_P, gs)) for i in range(3)]
    var = sum(p * sum((x - m) ** 2 for x, m in zip(g, mean))
              for p, g in zip(POLICY_P, gs))
    return gs, mean, var


def hacking_history(repair=False):
    rewards = [1., 0., 0.] if repair else [1., 2., 0.]
    z, history = [0.] * 3, []
    for step in range(61):
        p = softmax(z)
        expected = dot(p, rewards)
        history.append({"update": step, "probabilities": p,
                        "proxy_reward": expected, "strict_accuracy": p[0]})
        if step < 60:
            z = [x + .4 * a * (r - expected) for x, a, r in zip(z, p, rewards)]
    return history


TEACHER = [2., .5, -1.]
STUDENT = [.2, .1, -.3]


def distillation(z, temperature):
    q = softmax([x / temperature for x in TEACHER])
    p = softmax([x / temperature for x in z])
    loss = temperature ** 2 * sum(a * math.log(a / b) for a, b in zip(q, p))
    return q, p, loss, [temperature * (b - a) for a, b in zip(q, p)]


def finite_difference(fn, values, eps=1e-5):
    grads = []
    for i in range(len(values)):
        lo, hi = list(values), list(values)
        lo[i] -= eps
        hi[i] += eps
        grads.append((fn(hi) - fn(lo)) / (2 * eps))
    return grads


def close(a, b, tol=1e-8):
    assert len(a) == len(b)
    assert all(abs(x - y) < tol for x, y in zip(a, b)), (a, b)


def verify():
    checks = []
    new, gradient = ce_step(CE_LOGITS)
    close(gradient, finite_difference(ce, CE_LOGITS))
    assert ce(new) < ce(CE_LOGITS)
    checks.append("cross-entropy p-minus-q finite difference and descent")

    emb = embedding_fixture()
    def context_loss(row):
        h = [.4 * row[j] + .6 * EMBEDDINGS[2][j] for j in range(2)]
        return ce([dot(w, h) for w in HEAD])
    close(emb["gradient"][1], finite_difference(context_loss, EMBEDDINGS[1]))
    assert any(abs(x) > 0 for x in emb["gradient"][1])
    assert emb["gradient"][0] == emb["gradient"][3] == [0., 0.]
    checks.append("ignored prompt target still permits contextual embedding gradient")

    # h_out=h+F(h), F(h)=[.3*h0,-.2*h1]. Separate branch and identity gradients.
    h, upstream = [.6, -.4], [.5, -.7]
    residual_loss = lambda v: dot([1.3 * v[0], .8 * v[1]], upstream)
    close(finite_difference(residual_loss, h), [1.3 * upstream[0], .8 * upstream[1]])
    checks.append("residual identity plus branch derivative")

    q, k, theta = [.9, .3], [.2, .8], .45
    for m, n in [(0, 0), (1, 3), (4, 7)]:
        close([dot(rotate(q, m * theta), rotate(k, n * theta))],
              [dot(q, rotate(k, (n - m) * theta))])
        close([dot(rotate(q, m * theta), rotate(q, m * theta))], [dot(q, q)])
    checks.append("RoPE norms and relative-angle dot products")

    # Unequal valid-target counts: target sums, rather than means, aggregate.
    sums, counts = [2., 9.], [2, 6]
    assert sum(sums) / sum(counts) == 1.375
    assert sum(s / n for s, n in zip(sums, counts)) / 2 != 1.375
    close([(2 * .2 + 6 * .8) / 8], [.65])
    checks.append("valid-target-weighted loss and gradient accumulation")

    rng = random.Random(17)
    saved = rng.getstate()
    expected = [rng.random() for _ in range(4)]
    rng.setstate(saved)
    assert [rng.random() for _ in range(4)] == expected
    checks.append("RNG continuation after checkpoint restore")

    # Labels belong to next token; assistant header predicts first answer.
    tokens = ["USER", "copy", "red", "ASSISTANT", "red", "END"]
    ownership = [0, 0, 0, 0, 1, 1]
    assert [tokens[t + 1] for t in range(5) if ownership[t + 1]] == ["red", "END"]
    assert ownership[3] == 0 and ownership[4] == 1
    segments = [0, 0, 0, 1, 1, 1]
    mask = [[j <= i and segments[i] == segments[j] for j in range(6)] for i in range(6)]
    assert mask[4][3] and not mask[4][2] and not mask[3][4]
    checks.append("single causal shift, answer END ownership, block-diagonal causal packing")

    # Rank-one LoRA: B=0 gives zero adapter output and dA, but dB can be nonzero.
    a, b, x = [.6, -.2], [0., 0.], [.5, .8]
    assert dot(a, x) != 0
    assert [v * dot(a, x) for v in b] == [0., 0.]
    db = [dot(a, x), 2 * dot(a, x)]
    close(db, finite_difference(lambda v: dot([v[0] * dot(a, x), v[1] * dot(a, x)], [1., 2.]), b))
    close([0., 0.], finite_difference(lambda v: dot([b[0] * dot(v, x), b[1] * dot(v, x)], [1., 2.]), a))
    checks.append("LoRA zero-output initialization and asymmetric first gradients")

    s = [.4, -.6]
    close(finite_difference(preference_loss, s), [sigmoid(1.) - 1, 1 - sigmoid(1.)])
    checks.append("Bradley-Terry score derivatives")
    z = [math.log(x) for x in [.1, .01, .89]]
    d = dpo_margin(softmax(z))
    close(finite_difference(dpo_loss, z), [.5 * (sigmoid(d) - 1), .5 * (1 - sigmoid(d)), 0.])
    assert softmax(z)[0] < .2 and dpo_loss(z) < math.log(2)
    checks.append("DPO relative-margin gradient and absolute-probability counterexample")

    for baseline, variance in [(0., .8198), (.7, .2472), (5., 10.0598)]:
        _, mean, var = policy_estimates(baseline)
        close(mean, [-.35, .09, .26])
        close([var], [variance])
    checks.append("exact REINFORCE mean and baseline covariance")
    assert ppo(1.5, 1.) == 1.2 and ppo(.5, -1.) == -.8
    assert ppo(.5, 1.) == .5 and ppo(1.5, -1.) == -1.5
    checks.append("PPO asymmetric clipping and harmful-direction slope")
    close([2 / 8, 6 / 8], [.25, .75])
    checks.append("response weighting: equal responses versus equal tokens")

    bad, fixed = hacking_history(), hacking_history(True)
    assert abs(bad[-1]["proxy_reward"] - 1.966561) < 5e-7
    assert abs(bad[-1]["strict_accuracy"] - .016816) < 5e-7
    assert abs(fixed[-1]["strict_accuracy"] - .964428) < 5e-7
    checks.append("60-update finite-policy replay agrees with recorded chapter endpoints")

    for temperature in [1., 2., 4.]:
        q, p, loss, g = distillation(STUDENT, temperature)
        close(g, finite_difference(lambda z: distillation(z, temperature)[2], STUDENT))
        close([sum(q), sum(p)], [1., 1.])
        z_new = [z - .4 * d for z, d in zip(STUDENT, g)]
        assert distillation(z_new, temperature)[2] < loss
    checks.append("temperature-squared KL gradient and student descent")
    return checks


if __name__ == "__main__":
    for check in verify():
        print("PASS", check)
