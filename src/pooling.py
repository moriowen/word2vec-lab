"""Sentence vectors. Mean pooling discards word order, which is why negation breaks later."""
import numpy as np


def mean_pool(rows, kv):
    """Returns (X, y, oov_rate). A sentence with no in-vocabulary word gets a zero vector."""
    dim = kv.vector_size
    X = np.zeros((len(rows), dim), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int32)
    seen = missed = 0
    for i, (toks, lab) in enumerate(rows):
        vecs = []
        for w in toks:
            if w in kv:
                vecs.append(kv[w]); seen += 1
            else:
                missed += 1
        if vecs:
            X[i] = np.mean(vecs, axis=0)
        y[i] = lab
    return X, y, missed / max(seen + missed, 1)
