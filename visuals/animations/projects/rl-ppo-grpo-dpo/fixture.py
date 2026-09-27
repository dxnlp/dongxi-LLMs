"""Toy numbers behind every quantity the video draws, with self-checks.

Run:  python fixture.py   ->  writes fixture.json next to this file.

All values are illustrative. They are chosen so the mechanisms are visible,
not measured from any trained model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# PPO: one sampled response, a learned critic, GAE, clipped ratio updates.
# ---------------------------------------------------------------------------
# V[t] is the critic's estimate of the final score *before* token t is chosen
# (state s_t = prompt + tokens < t). The reward model scores only the finished
# response, so the only nonzero environment reward arrives after the last token.
PPO_VALUES = np.array([0.50, 0.55, 0.80, 0.40, 0.45, 0.50])
PPO_REWARD = 0.62
GAMMA, LAM, EPS = 1.0, 0.95, 0.2


def gae(values, final_reward, gamma, lam):
    nxt = np.append(values[1:], 0.0)           # V(s_{t+1}); terminal value 0
    rewards = np.zeros_like(values)
    rewards[-1] = final_reward
    deltas = rewards + gamma * nxt - values     # TD errors
    adv = np.zeros_like(values)
    running = 0.0
    for t in reversed(range(len(values))):
        running = deltas[t] + gamma * lam * running
        adv[t] = running
    return deltas, adv


def ppo_clip_objective(ratio, adv, eps=EPS):
    return min(ratio * adv, float(np.clip(ratio, 1 - eps, 1 + eps)) * adv)


def ppo_ratio_trajectory(adv, steps=6, lr=0.65, eps=EPS):
    """Schematic ratio motion: move in the advantage direction until clipped.

    d/dr of min(rA, clip(r)A) is A inside the trust band and 0 once r has
    left the band in the direction the advantage favours, so motion stops.
    """
    ratio = np.ones_like(adv)
    path = [ratio.copy()]
    for _ in range(steps):
        grad = np.array([a if (1 - eps) < r < (1 + eps) else 0.0
                         for r, a in zip(ratio, adv)])
        ratio = np.clip(ratio * np.exp(lr * grad), 1 - eps, 1 + eps)
        path.append(ratio.copy())
    return np.array(path)


# ---------------------------------------------------------------------------
# GRPO: a group of answers to the same prompt, compared with each other.
# ---------------------------------------------------------------------------
GRPO_REWARDS = np.array([0.9, 0.3, 0.6, 0.2])
GRPO_LENGTHS = [6, 5, 7, 4]


# ---------------------------------------------------------------------------
# DPO: one fixed preference pair, implicit rewards relative to the reference.
# ---------------------------------------------------------------------------
# rhat = beta * log(pi_theta(y|x) / pi_ref(y|x)). At initialisation the policy
# equals the reference, so both implicit rewards start at exactly zero.
DPO_STEPS, DPO_ALPHA = 8, 0.6


def dpo_loss(margin):
    return float(np.log1p(np.exp(-margin)))      # -log sigmoid(margin)


def dpo_trajectory(steps=DPO_STEPS, alpha=DPO_ALPHA):
    """Gradient descent on -log sigmoid(r_w - r_l) in implicit-reward space."""
    r_w, r_l, rows = 0.0, 0.0, []
    for _ in range(steps + 1):
        m = r_w - r_l
        weight = float(sigmoid(-m))               # |dL/dm|: large when wrong
        rows.append(dict(r_w=r_w, r_l=r_l, margin=m, loss=dpo_loss(m), weight=weight))
        r_w += alpha * weight
        r_l -= alpha * weight
    return rows


def finite_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def build():
    checks = {}

    deltas, adv = gae(PPO_VALUES, PPO_REWARD, GAMMA, LAM)
    explicit = np.array([sum((GAMMA * LAM) ** k * deltas[t + k]
                             for k in range(len(deltas) - t)) for t in range(len(deltas))])
    checks['gae_recursion_matches_sum'] = bool(np.allclose(adv, explicit))
    _, adv_mc = gae(PPO_VALUES, PPO_REWARD, GAMMA, 1.0)
    checks['gae_lambda1_is_return_minus_value'] = bool(np.allclose(adv_mc, PPO_REWARD - PPO_VALUES))
    checks['ppo_advantage_has_both_signs'] = bool((adv > 0).any() and (adv < 0).any())
    checks['ppo_bad_token_is_negative'] = bool(adv[2] < 0 and deltas[2] < 0)

    fd_ok = True
    for a in (1.0, -1.0):
        for r in (0.7, 0.9, 1.1, 1.3):
            g = finite_diff(lambda z: ppo_clip_objective(z, a), r)
            inside = (1 - EPS) < r < (1 + EPS)
            favourable_out = (a > 0 and r > 1 + EPS) or (a < 0 and r < 1 - EPS)
            expected = a if inside or not favourable_out else 0.0
            fd_ok &= math.isclose(g, expected, abs_tol=1e-6)
    checks['ppo_clip_gradient_zero_outside_band'] = bool(fd_ok)
    ratios = ppo_ratio_trajectory(adv)
    checks['ppo_ratios_stay_in_band'] = bool(((ratios >= 1 - EPS - 1e-12) & (ratios <= 1 + EPS + 1e-12)).all())
    checks['ppo_ratio_direction_matches_advantage'] = bool(np.all(np.sign(ratios[-1] - 1) == np.sign(adv)))

    mean, std_pop, std_unb = GRPO_REWARDS.mean(), GRPO_REWARDS.std(), GRPO_REWARDS.std(ddof=1)
    grpo_adv = (GRPO_REWARDS - mean) / std_pop
    checks['grpo_advantages_zero_mean'] = bool(abs(grpo_adv.mean()) < 1e-12)
    checks['grpo_advantages_unit_std'] = bool(abs(grpo_adv.std() - 1) < 1e-12)
    checks['grpo_std_choice_only_rescales'] = bool(np.allclose(
        grpo_adv * std_pop / std_unb, (GRPO_REWARDS - mean) / std_unb))

    dpo = dpo_trajectory()
    checks['dpo_starts_at_log2'] = math.isclose(dpo[0]['loss'], math.log(2))
    checks['dpo_loss_decreases'] = all(b['loss'] < a['loss'] for a, b in zip(dpo, dpo[1:]))
    checks['dpo_weight_decreases'] = all(b['weight'] < a['weight'] for a, b in zip(dpo, dpo[1:]))
    grads_ok = all(math.isclose(finite_diff(dpo_loss, row['margin']), -row['weight'], abs_tol=1e-6)
                   for row in dpo)
    checks['dpo_gradient_is_minus_sigmoid_neg_margin'] = grads_ok

    assert all(checks.values()), {k: v for k, v in checks.items() if not v}

    return dict(
        note='Illustrative toy values for the animation; not measurements of a trained model.',
        ppo=dict(values=PPO_VALUES.tolist(), reward=PPO_REWARD, gamma=GAMMA, lam=LAM, eps=EPS,
                 deltas=deltas.round(6).tolist(), advantages=adv.round(6).tolist(),
                 ratio_path=ratios.round(6).tolist()),
        grpo=dict(rewards=GRPO_REWARDS.tolist(), lengths=GRPO_LENGTHS, mean=float(mean),
                  std_population=float(std_pop), std_unbiased=float(std_unb),
                  advantages=grpo_adv.round(6).tolist()),
        dpo=dict(alpha=DPO_ALPHA, trajectory=[{k: round(v, 6) for k, v in r.items()} for r in dpo]),
        checks=checks,
    )


if __name__ == '__main__':
    data = build()
    (HERE / 'fixture.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data['checks'], indent=2))
    print('PPO deltas     ', data['ppo']['deltas'])
    print('PPO advantages ', data['ppo']['advantages'])
    print('PPO final ratio', data['ppo']['ratio_path'][-1])
    print('GRPO advantages', data['grpo']['advantages'])
    print('DPO margins    ', [r['margin'] for r in data['dpo']['trajectory']])
