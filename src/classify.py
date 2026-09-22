"""One classifier for every model. Any per-model branching here invalidates the comparison."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .pooling import mean_pool


def evaluate(kv, data: dict) -> dict:
    Xtr, ytr, oov_tr = mean_pool(data["train"], kv)
    Xte, yte, oov_te = mean_pool(data["test"], kv)

    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "sst2_test_acc": round(float(accuracy_score(yte, pred)), 4),
        "sst2_test_f1": round(float(f1_score(yte, pred)), 4),
        "oov_rate_train": round(float(oov_tr), 4),
        "oov_rate_test": round(float(oov_te), 4),
    }, clf, prob, pred, yte
