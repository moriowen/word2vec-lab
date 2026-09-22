"""Phases 6 to 8: intrinsic similarity, systems measurement, and error mining.

Augments the existing per-model records in results/ rather than rewriting them, so the
training numbers captured at training time are never recomputed from a different run.
"""
import json, time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval"
BENCHMARKS = {"wordsim353": "EN-WS-353-ALL.txt",
              "simlex999": "EN-SIMLEX-999.txt",
              "men3k": "EN-MEN-TR-3k.txt"}

NEGATION = {"not", "n't", "no", "never", "without", "nothing", "none", "nor", "cannot"}
CONTRAST = {"but", "although", "though", "yet", "however", "despite", "while"}


# ---------------------------------------------------------------- phase 6

def similarity(kv, name: str) -> dict:
    """Spearman between human ratings and cosine. Skipped pairs are part of the result."""
    gold, pred, skipped = [], [], 0
    for line in (EVAL / BENCHMARKS[name]).read_text().splitlines():
        if not line.strip():
            continue
        a, b, score = line.split()
        if a in kv and b in kv:
            gold.append(float(score))
            pred.append(float(kv.similarity(a, b)))
        else:
            skipped += 1
    rho = spearmanr(gold, pred).statistic if len(gold) > 2 else float("nan")
    return {"spearman": round(float(rho), 4), "pairs_scored": len(gold),
            "pairs_skipped": skipped,
            "coverage": round(len(gold) / (len(gold) + skipped), 3)}


# ---------------------------------------------------------------- phase 7

def latency(kv, queries, repeats=200) -> dict:
    """Brute-force most_similar is a dense matvec over the whole vocabulary."""
    kv.fill_norms()
    times = []
    for i in range(repeats):
        q = queries[i % len(queries)]
        t0 = time.perf_counter()
        kv.most_similar(q, topn=10)
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {"p50_ms": round(float(np.percentile(times, 50)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "repeats": repeats, "vocab_size": len(kv)}


def memory_model(vocab_size, dim, measured_mb) -> dict:
    """Two float32 matrices of vocab_size x dim. Any gap against measured is the interesting part."""
    predicted = 2 * vocab_size * dim * 4 / 1e6
    return {"predicted_matrices_mb": round(predicted, 1),
            "measured_peak_rss_mb": measured_mb,
            "overhead_mb": round(measured_mb - predicted, 1) if measured_mb else None}


# ---------------------------------------------------------------- phase 8

def categorise(tokens) -> str:
    s = set(tokens)
    if s & NEGATION:
        return "negation"
    if s & CONTRAST:
        return "contrast"
    return "other"


def mine_errors(rows, prob, pred, y, n=6) -> dict:
    """Most confident mistakes. Confident errors carry an explanation; random ones do not."""
    conf = np.where(pred == 1, prob, 1 - prob)
    wrong = np.where(pred != y)[0]
    right = np.where(pred == y)[0]
    order = wrong[np.argsort(-conf[wrong])]

    def row(i):
        toks = rows[i][0]
        return {"sentence": " ".join(toks), "gold": int(y[i]), "pred": int(pred[i]),
                "confidence": round(float(conf[i]), 3), "category": categorise(toks)}

    cats = {}
    for i in wrong:
        cats[categorise(rows[i][0])] = cats.get(categorise(rows[i][0]), 0) + 1
    return {"n_errors": int(len(wrong)),
            "error_categories": cats,
            "confident_errors": [row(i) for i in order[:n]],
            "confident_correct": [row(i) for i in right[np.argsort(-conf[right])][:n]]}
