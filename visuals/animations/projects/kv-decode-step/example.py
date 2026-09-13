"""Preserved article example: one head, one position, no positional operations."""
import numpy as np


def attend_new(h, Wq, Wk, Wv, K, V):
    q = h @ Wq
    K = np.vstack([K, h @ Wk])
    V = np.vstack([V, h @ Wv])
    scores = q @ K.T / np.sqrt(q.size)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return weights @ V, K, V


def fixture():
    """Explicit didactic inputs; not measured activations from a trained model."""
    h = np.array([1., 1.])
    Wq = np.array([[1., 0.], [0., 0.]])
    Wk = Wv = np.eye(2)
    past = np.eye(2)
    output, K, V = attend_new(h, Wq, Wk, Wv, past, past)
    q = h @ Wq
    scores = q @ K.T / np.sqrt(2)
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    np.testing.assert_allclose(output, weights @ V)
    return dict(h=h.tolist(), q=q.tolist(), K=K.tolist(), V=V.tolist(),
                scores=scores.tolist(), weights=weights.tolist(), output=output.tolist())
