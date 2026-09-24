"""Deliverable 3: top-K ranked results for five fixed test examples, across all five models.

Two ranked outputs per (model, sentence):

  the classifier's ranked labels with probabilities, which is the sentiment decision itself
  the top-K nearest TRAINING sentences by cosine in the embedding space, which is what
  "ranked results" means for a retrieval view of the same vectors

The second is the more informative of the two. A binary classifier's ranking is only two
rows deep, whereas the retrieved neighbours show which training evidence each embedding
actually places near the test sentence, and whether the labels of that evidence agree.
"""
import json

import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from . import data, evaluate, pretrained, schema
from .classify import evaluate as classify
from .main import CORPUS
from .pooling import mean_pool
from .run_eval import MODELS, load_all_kv

K = 5


def pick(rows, n=5) -> list[int]:
    """Five diagnostic test sentences, chosen by rule so the choice is reproducible."""
    neg = evaluate.NEGATION
    con = evaluate.CONTRAST
    lens = np.array([len(t) for t, _ in rows])
    ok = (lens >= 8) & (lens <= 22)          # long enough to be a sentence, short enough to quote

    def first(pred):
        for i in np.where(ok)[0]:
            if pred(rows[i][0], rows[i][1]) and i not in chosen:
                return int(i)
        raise LookupError

    rules = [
        ("plain positive", lambda t, y: y == 1 and not (set(t) & (neg | con))),
        ("plain negative", lambda t, y: y == 0 and not (set(t) & (neg | con))),
        ("positive despite negation", lambda t, y: y == 1 and set(t) & neg),
        ("contrastive clause", lambda t, y: set(t) & con),
        ("negative with stacked negation", lambda t, y: y == 0 and set(t) & neg),
    ]
    chosen, reasons = [], []
    for why, pred in rules:
        chosen.append(first(pred)); reasons.append(why)
    return chosen, reasons


def main():
    d = data.load_sst2("phrases")
    idx, reasons = pick(d["test"])
    out = {"indices": idx,
           "sentences": [{"text": " ".join(d["test"][i][0]), "gold": int(d["test"][i][1]),
                          "category": evaluate.categorise(d["test"][i][0]),
                          "selected_for": why} for i, why in zip(idx, reasons)],
           "models": {}}

    for mid, kv, _ in load_all_kv():
        Xtr, ytr, _ = mean_pool(d["train"], kv)
        ext, clf, prob, pred, y = classify(kv, d)

        # Normalise once so cosine is a dot product.
        norms = np.linalg.norm(Xtr, axis=1, keepdims=True)
        Xn = Xtr / np.where(norms == 0, 1, norms)

        per = []
        for i in idx:
            v, _, _ = mean_pool([d["test"][i]], kv)
            vn = v[0] / (np.linalg.norm(v[0]) or 1)
            sims = Xn @ vn
            top = np.argsort(-sims)[:K]
            p1 = float(prob[i])
            ranked = sorted([("positive", p1), ("negative", 1 - p1)], key=lambda t: -t[1])
            per.append({
                "gold": int(y[i]), "pred": int(pred[i]),
                "ranked_labels": [[lab, round(p, 4)] for lab, p in ranked],
                "correct": bool(pred[i] == y[i]),
                "neighbours": [{"text": " ".join(d["train"][j][0])[:80],
                                "label": int(ytr[j]), "cos": round(float(sims[j]), 3)}
                               for j in top],
                "neighbour_label_agreement": round(float(np.mean(ytr[top] == y[i])), 2),
            })
        out["models"][mid] = {"name": kv_name(mid), "cases": per,
                              "test_acc": ext["sst2_test_acc"]}
        print(f"{mid}: {sum(c['correct'] for c in per)}/{len(per)} of the five correct")

    (schema.RESULTS / "testcases.json").write_text(json.dumps(out, indent=2))
    print(f"wrote results/testcases.json")


def kv_name(mid):
    return {"A": "gensim skip-gram + NS", "B": "gensim CBOW + HS",
            "G": "GoogleNews-300 (pretrained)", "G-ft": "GoogleNews fine-tuned",
            "C": "from scratch (PyTorch)"}[mid]


if __name__ == "__main__":
    main()
