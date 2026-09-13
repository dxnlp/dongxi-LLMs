"""Check the preserved executable companion after removing code from the article."""
from pathlib import Path
import importlib.util

import numpy as np


def main():
    root = Path(__file__).resolve().parent
    example = root.parents[2] / 'visuals/animations/projects/kv-decode-step/example.py'
    spec = importlib.util.spec_from_file_location('decode_example', example)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for folder in (root, root / 'zh'):
        source = (folder / 'x-editor-draft-body-with-image-placeholders.md').read_text()
        assert '```' not in source and 'def attend_new' not in source
        assert '[IMAGE 04]' in source
    attend_new = module.attend_new
    module.fixture()
    rng = np.random.default_rng(42)
    h = rng.normal(size=(6, 4))
    Wq, Wk, Wv = rng.normal(size=(3, 4, 3))
    K = np.empty((0, 3))
    V = np.empty((0, 3))
    incremental = []
    for t in range(len(h)):
        output, K, V = attend_new(h[t], Wq, Wk, Wv, K, V)
        assert K.shape == V.shape == (t + 1, 3)
        incremental.append(output)

    # Independent full-sequence, causally masked attention.
    scores = (h @ Wq) @ (h @ Wk).T / np.sqrt(3)
    scores[np.triu_indices(len(h), k=1)] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    reference = weights @ (h @ Wv)
    np.testing.assert_allclose(incremental, reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(K, h @ Wk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(V, h @ Wv, rtol=1e-12, atol=1e-12)
    print('PASS: preserved companion matches full causal attention over six steps; toy animation fixture verified; both articles are code-free.')


if __name__ == '__main__':
    main()
