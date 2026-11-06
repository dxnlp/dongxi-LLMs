"""Exact arithmetic behind the DPO article: every printed number in the drafts comes from here.

python3 publications/x-articles/x-post-training-kl-dpo/check_numbers.py

The ideal test reuses the notebook's fit (src/dongxi_llms/study_path/s2_kl_optimum_and_dpo.py).
"""
from math import exp, log
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from dongxi_llms.study_path import s2_kl_optimum_and_dpo as s2  # noqa: E402

np.set_printoptions(precision=3, suppress=True)
REF = np.array([.40, .25, .15, .12, .08])
R = np.array([0, 1, .5, 2, -1.])
SMALL = np.array([.30, .25, .15, .22, .08])
ALWAYS = np.array([0, 0, 0, 1., 0])
REF6 = np.array([.40, .25, .15, .12, .075, .005])
R6 = np.array([0, 1, .5, 2, -1, 3.])
GOLD6 = np.array([0, 1, .5, 2, -1, -2.])
BEST, TEN, VAGUE = 3, 1, 0
sig = lambda x: 1 / (1 + exp(-x))
bt = lambda m: -log(sig(m))


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def J(p, ref=REF, r=R, beta=1.):
    return float(p @ r) - beta * kl(p, ref)


def optimum(ref, r, beta):
    w = ref * np.exp(r / beta)
    return w / w.sum(), w, w.sum()


print('1 OPENING')
star, w, Z = optimum(REF, R, 1)
print('small: log-ratio', np.log(SMALL / REF), 'chosen', round(log(.22 / .12), 3), 'gap', round(log(.22 / .12) - 0, 3), 'loss', round(bt(log(.22 / .12)), 3))
print('start gap 0 loss', round(bt(0), 3), ' star: log-ratio', np.log(star / REF), 'gap', round(log(star[BEST] / REF[BEST]) - log(star[TEN] / REF[TEN]), 3), 'loss', round(bt(1), 3), 'sigma(1)', round(sig(1), 3))
print('sigma(.606)', round(sig(log(.22 / .12)), 3))

print('\n2 SCORES')
print('small KL', round(kl(SMALL, REF), 3))
for n, p in [('reference', REF), ('small', SMALL), ('always', ALWAYS), ('star', star)]:
    print(f'{n:10} E[R] {p @ R:.3f}  KL {kl(p, REF):.3f}  J {J(p):.3f}')

print('\n3 REWARD ALONE vs WITH KL (six answers)')
print('reference: reward', round(REF6 @ R6, 3), 'gold', round(REF6 @ GOLD6, 3))
always_trick = np.array([0, 0, 0, 0, 0, 1.])
print('reward alone: all on the trick, reward', R6[5], 'gold', GOLD6[5], 'KL', round(log(1 / .005), 3))
for b in [2, 1, .5]:
    p = optimum(REF6, R6, b)[0]
    print(f'beta {b}: shares {p}  reward {p @ R6:.3f} gold {p @ GOLD6:.3f} KL {kl(p, REF6):.3f}')

print('\n4 FROZEN vs MOVING REFERENCE (each step re-weights by e^(0.4 R), six answers)')
p = REF6.copy()
for t in range(1, 21):
    q = p * np.exp(.4 * R6); q /= q.sum()
    if t in (1, 2, 5, 10, 20):
        print(f'step {t}: KL to previous {kl(q, p):.3f}  KL to start {kl(q, REF6):.3f}  trick {q[5]:.3f} gold {q @ GOLD6:.3f}')
    p = q
steps = []
p = REF6.copy()
for t in range(1, 11):
    q = p * np.exp(.4 * R6); q /= q.sum(); steps.append(kl(q, p)); p = q
print('ten steps: largest step KL', round(max(steps), 3), ' average', round(float(np.mean(steps)), 3), ' total to start', round(kl(p, REF6), 3))

print('\n5 NO MOVE HELPS')
net = lambda p, beta=1.: R - beta * np.log(p / REF)
print('net small', net(SMALL), ' net star', net(star))
def chip(p, frm=VAGUE, to=BEST, d=.01):
    q = p.copy(); q[frm] -= d; q[to] += d
    return J(q) - J(p), J(q)
print('chip small', chip(SMALL), ' star forward', chip(star), ' star back', chip(star, BEST, VAGUE))
print('multiply e^R', np.exp(R), 'weights', w, 'Z', round(Z, 3), 'star', star, 'log Z', round(log(Z), 3), 'check', round(.12 * exp(2) / Z, 3))
for b in [.5, 1, 2, 5]:
    print('beta', b, optimum(REF, R, b)[0], 'KL', round(kl(optimum(REF, R, b)[0], REF), 3))

print('\n6 READ THE REWARD BACK')
print('ratios', star / REF, 'logs', np.log(star / REF), 'plus shift', np.log(star / REF) + log(Z))

print('\n7 LOSS')
for m in [-3, -1, 0, log(.22 / .12), 1, 3, 6]:
    print(f'gap {m:.3f}: sigma {sig(m):.3f}  loss {bt(m):.3f}  push {sig(-m):.3f}')

print('\n8 PUSH')
print('nudge from 0:', round(bt(0), 4), round(bt(.01), 4), 'slope', round((bt(.01) - bt(0)) / .01, 3))
print('nudge from 1:', round(bt(1), 4), round(bt(1.01), 4), 'slope', round((bt(1.01) - bt(1)) / .01, 3))
print('nudge from 3:', round(bt(3), 4), round(bt(3.01), 4), 'slope', round((bt(3.01) - bt(3)) / .01, 3))
print('nudge from -3:', round(bt(-3), 4), round(bt(-2.99), 4), 'slope', round((bt(-2.99) - bt(-3)) / .01, 3))
m = 0.
for t in range(5):
    print(f'step {t}: gap {m:.3f} loss {bt(m):.3f} push {sig(-m):.3f}')
    m += sig(-m)
for beta, lr in [(.5, 2), (1, 1), (2, .5)]:
    d, trace = 0., {}
    for t in range(1, 101):
        d += lr * beta * sig(-2 * beta * d)
        if t in (1, 10, 100): trace[t] = round(d, 3)
    print('beta', beta, 'lr', lr, 'distance after', trace)

print('\n9 WHERE THE PROBABILITY COMES FROM')
theta = np.log(REF).copy(); theta[BEST] += .5; theta[TEN] -= .5
one = np.exp(theta) / np.exp(theta).sum()
print('one step', one, 'change', one - REF, 'log-ratio', np.log(one / REF), 'gap', round(log(one[BEST] / REF[BEST]) - log(one[TEN] / REF[TEN]), 3))
print('star', star)

print('\n10 IDEAL TEST')
REFt, Rt = s2.TOY_REF, s2.TOY_REWARD
STAR = torch.softmax(REFt.log() + Rt, -1)
print('reference to star TV', round(float(s2.tv(REFt, STAR)), 3))
W, L = s2.sample_pairs(1000, Rt, seed=0)
pi, _ = s2.fit_tabular_dpo(s2.pair_counts(W, L), REFt, 1.0)
print('1000 pairs seed 0:', pi.numpy().round(3), 'TV', round(float(s2.tv(pi, STAR)), 3), 'first eight', [f'{int(a)}>{int(b)}' for a, b in zip(W[:8], L[:8])])
Ns = [10, 100, 1000, 10000]
counts = torch.stack([s2.pair_counts(*s2.sample_pairs(n, Rt, seed=100 * n + k)) for n in Ns for k in range(8)])
pis, _ = s2.fit_tabular_dpo(counts, REFt, 1.0)
d = s2.tv(pis.view(len(Ns), 8, 5), STAR)
print('mean TV over 8 datasets', dict(zip(Ns, d.mean(1).numpy().round(3))))
one_sided = ((counts > 0) & (counts.transpose(1, 2) == 0)).flatten(1).any(1).view(len(Ns), -1).float().mean(1)
print('datasets with a one-sided pair type', dict(zip(Ns, one_sided.numpy().round(2))))
print('N=10 example policy', pis.view(len(Ns), 8, 5)[0, 0].numpy().round(3))
print('log-ratio gap best vs 10 min by beta (= 1/beta), KL of the optimum:')
for b in [4, 2, 1, .5, .1]:
    p = optimum(REF, R, b)[0]
    print(f'  beta {b}: gap {log(p[BEST] / REF[BEST]) - log(p[TEN] / REF[TEN]):.3f}  score gap {b * (log(p[BEST] / REF[BEST]) - log(p[TEN] / REF[TEN])):.3f}  KL {kl(p, REF):.3f}  best {p[BEST]:.3f}')
one_pair = pi

print('\n11 TOKENS')
print('forty tokens at probability e^-1 each: total', -40, ' sixty tokens', -60)
print('reference vs trained, illustrative: -40 -> -38, gain', 2, ' ; -60 -> -59, gain', 1)
print('three-token reply from the KL article: log(.24/.16)', round(log(.24 / .16), 3), 'token log-ratios', [round(log(a / b), 3) for a, b in zip([.5, .5, .96], [.4, .5, .8])])
print('chapter 11 constructed case: chosen .20->.10, rejected .10->.01, third .70->.89, beta .5, margin', round(.5 * log(10 / 2), 3), 'loss', round(bt(.5 * log(10 / 2)), 3))

print('\n12 THE GAP CAN WIDEN WHILE THE CHOSEN ANSWER FALLS (constructed)')
shifted = np.array([0, .025, 0, .06, 0])
rest = np.array([.40, 0, .15, 0, .08]); rest = rest / rest.sum() * (1 - shifted.sum())
p = shifted + rest
gap = log(p[BEST] / REF[BEST]) - log(p[TEN] / REF[TEN])
print('shares', p, 'sum', round(p.sum(), 6), 'gap', round(gap, 3), 'chance', round(sig(gap), 3), 'loss', round(bt(gap), 3))

print('\nGIF 09 TRAINING PATH (seed 0, 1,000 pairs, Adam lr 0.1, full batch)')
theta = REFt.log().clone().requires_grad_()
opt = torch.optim.Adam([theta], lr=.1)
marks = [0, 2, 4, 6, 8, 10, 13, 16, 20, 25, 30, 40, 60, 100, 200, 1500]
path = []
for step in range(1501):
    if step in marks:
        p = theta.detach().softmax(-1)
        path.append([round(float(x), 3) for x in p])
        print(step, path[-1], 'distance', round(float(s2.tv(p, STAR)), 3))
    loss = s2.ref_tabular_dpo_loss(theta, W, L, REFt.log(), 1.0)
    opt.zero_grad(); loss.backward(); opt.step()
