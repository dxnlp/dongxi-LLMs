"""Exact arithmetic behind the KL article: every printed number in the drafts comes from here.

python3 publications/x-articles/x-post-training-kl/check_numbers.py
"""
from math import exp, log

import numpy as np

REF = np.array([.40, .25, .15, .12, .08])
R = np.array([0, 1, .5, 2, -1.])
SMALL = np.array([.30, .25, .15, .22, .08])
AWAY = np.array([.50, .25, .15, .02, .08])
EXTREME = np.array([.01, .25, .15, .51, .08])
SPREAD = np.array([.10, .35, .25, .22, .08])
ALWAYS = np.array([0, 0, 0, 1., 0])
DROP = np.array([.40, .25, .15, .20, 0])
# The sixth answer: rare under the reference, overrated by the reward model.
REF6 = np.array([.40, .25, .15, .12, .075, .005])
R6 = np.array([0, 1, .5, 2, -1, 3.])
GOLD6 = np.array([0, 1, .5, 2, -1, -2.])


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def show(name, p):
    print(f'{name:9} KL {kl(p, REF):.3f}  ratio {np.round(p / REF, 3)}  log {np.round(np.log(np.where(p > 0, p, 1) / REF), 3)}'
          f'  term {np.round(p * np.log(np.where(p > 0, p, 1) / REF), 3)}')


def optimum(ref, r, beta):
    w = ref * np.exp(r / beta)
    return w / w.sum(), w.sum()


for name, p in [('small', SMALL), ('away', AWAY), ('extreme', EXTREME), ('spread', SPREAD), ('always', ALWAYS), ('drop', DROP)]:
    show(name, p)
print('reverse small', round(kl(REF, SMALL), 3))
print('other measures small/away: unweighted', round(np.log(SMALL / REF).sum(), 3), round(np.log(AWAY / REF).sum(), 3),
      ' ref-weighted', round((REF * np.log(SMALL / REF)).sum(), 3), round((REF * np.log(AWAY / REF)).sum(), 3))
for name, p in [('reference', REF), ('small', SMALL), ('away', AWAY), ('always', ALWAYS)]:
    print(f'J {name}: E[R] {(p * R).sum():.3f}  J {(p * R).sum() - kl(p, REF):.3f}')
print('best share at the optimum:', {b: round(float(optimum(REF, R, b)[0][3]), 2) for b in [.01, .5, 1, 2, 5, 1000]})
print('replies to 20:1 at KL 0.047:', round(log(20) / kl(SMALL, REF), 1), ' one best reply under always-best:', round(1 / .12, 2))

for name, i in [('about 10 minutes', 1), ('the trick', 5)]:
    p = REF6.copy(); p[0] -= .1; p[i] += .1
    print(f'0.10 to {name}: moved {.5 * abs(p - REF6).sum():.2f}  ratio {p[i] / REF6[i]:.2f}  term {p[i] * log(p[i] / REF6[i]):.3f}  KL {kl(p, REF6):.3f}')

rng = np.random.default_rng(0)
draws = rng.choice(len(SMALL), size=10_000, p=SMALL)
lr = np.log(SMALL / REF)[draws]
print('samples:', {n: round(float(lr[:n].mean()), 3) for n in [10, 100, 1000, 10000]}, 'exact', round(kl(SMALL, REF), 3))

print('tokens:', [round(log(a / b), 3) for a, b in zip([.50, .50, .96], [.40, .50, .80])], 'sum', round(log(.24 / .16), 3))
d = np.log(REF / SMALL)
print('GRPO form per answer (x - log x - 1):', np.round(np.exp(d) - d - 1, 3), 'mean over new', round(float((SMALL * (np.exp(d) - d - 1)).sum()), 3))

betas = np.exp(np.linspace(log(.2), log(1e4), 6000))
rows = []
for b in betas:
    p = optimum(REF6, R6, b)[0]
    rows.append((kl(p, REF6), (p * R6).sum(), (p * GOLD6).sum(), p[3], p[5]))
rows = np.array(rows)
print('reference: reward', round((REF6 * R6).sum(), 3), 'gold', round((REF6 * GOLD6).sum(), 3))
print('gold peak: KL %.2f reward %.2f gold %.2f best %.2f trick %.3f' % tuple(rows[rows[:, 2].argmax()]))
end = optimum(REF6, R6, .25)[0]
print('end (beta 0.25): KL %.2f reward %.2f gold %.2f best %.2f trick %.3f' % (kl(end, REF6), end @ R6, end @ GOLD6, end[3], end[5]))

ps, z = optimum(REF, R, 1)
print('optimum', ps.round(3), 'Z', round(z, 3), 'log ratio', np.round(np.log(ps / REF), 3), 'shift', round(log(z), 3))
sig = lambda x: 1 / (1 + exp(-x))
print('sigma(1)', round(sig(1), 3), 'loss at start', round(-log(sig(0)), 3), 'loss at optimum', round(-log(sig(1)), 3))
