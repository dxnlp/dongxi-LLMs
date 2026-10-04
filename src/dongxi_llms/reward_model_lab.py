"""Transparent Chapter 10 reward microscopies; original synthetic data, CPU only."""
import torch
from torch.nn import functional as F


def bt_loss(chosen, rejected, preference=None, reduction="mean"):
    """BCE on the score difference; preference is P(first wins), not a score."""
    if chosen.shape != rejected.shape:
        raise ValueError("Score shapes must match")
    q = torch.ones_like(chosen) if preference is None else preference
    if q.shape != chosen.shape or not torch.isfinite(q).all() or ((q < 0) | (q > 1)).any():
        raise ValueError("Preference probabilities must be finite, same shape, in [0,1]")
    return F.binary_cross_entropy_with_logits(chosen - rejected, q, reduction=reduction)


def preference_fixture(n=256, seed=1516, confounded=False):
    """Three observed differences: substantive quality, length, polished format.

    Soft labels come from quality alone. Confounded training makes nuisance
    features nearly redundant with quality. No human judgments are claimed.
    """
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=generator, dtype=torch.float64)
    if confounded:
        x[:, 1] = x[:, 0] + .04 * x[:, 1]
        x[:, 2] = x[:, 0] + .04 * x[:, 2]
    q = torch.sigmoid(2 * x[:, 0])
    return x, q


def fit_reward(x, q, steps=240, lr=.2, penalty=.002):
    """Linear difference model; intercept omitted because pairwise loss cancels it."""
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    history = []
    for _ in range(steps):
        loss = bt_loss(x @ w, torch.zeros_like(q), q) + penalty * w.square().sum()
        (gradient,) = torch.autograd.grad(loss, w)
        with torch.no_grad():
            w -= lr * gradient
        history.append(float(loss.detach()))
    return w.detach(), history


def calibration(probability, q, bins=8):
    """Expected Brier score for Bernoulli(q), plus weighted binned calibration.

    Using known soft q exposes the simulator's population behavior. Human-data
    calibration would use independent observed labels with uncertainty instead.
    """
    if probability.shape != q.shape or bins < 1:
        raise ValueError("Invalid calibration inputs")
    rows, ece = [], 0.
    for i in range(bins):
        mask = (probability >= i / bins) & (probability <= 1 if i == bins - 1 else probability < (i + 1) / bins)
        count = int(mask.sum())
        if count:
            predicted, observed = float(probability[mask].mean()), float(q[mask].mean())
            ece += count / len(q) * abs(predicted - observed)
            rows.append((count, predicted, observed))
    brier = (q * (1 - probability).square() + (1 - q) * probability.square()).mean()
    return {"expected_brier": float(brier), "ece": ece, "bins": rows}


def reward_comparison():
    validation, labels = preference_fixture(seed=1616)
    result = {}
    for name, confounded in (("confounded", True), ("balanced", False)):
        x, q = preference_fixture(confounded=confounded)
        w, history = fit_reward(x, q)
        # Worse substantive quality, longer and more polished presentation.
        adversary = torch.tensor([-1., 3., 3.], dtype=torch.float64)
        result[name] = {"weights": w.tolist(), "history": history,
                        "train_nll": float(bt_loss(x @ w, torch.zeros_like(q), q)),
                        "validation_nll": float(bt_loss(validation @ w, torch.zeros_like(labels), labels)),
                        "adversarial_first_win_probability": float(torch.sigmoid(adversary @ w)),
                        **calibration(torch.sigmoid(validation @ w), labels)}
    return result
