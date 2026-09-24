"""Deliverable 5 hyperparameter sweep on the SST phrase corpus.

One factor at a time around A's and B's settings, plus the 2x2 of architecture and loss that
A-versus-B leaves confounded. Three seeds per setting, because the A-B gap is small enough
that a single run could sit inside seed noise. Models are not saved; results/sweep.json holds
every run.
"""
import json
import statistics as st

from gensim.models import Word2Vec

from . import data, evaluate, schema
from .classify import evaluate as classify
from .train_gensim import CONFIGS

SEEDS = (42, 43, 44)
ARCH = {"A": dict(sg=1, hs=0, negative=10), "B": dict(sg=0, hs=1, negative=0)}


def settings():
    """(factor, label, base model, overrides) for every point in the sweep."""
    yield from [("arch x loss", "skip-gram + NS", "A", {}),
                ("arch x loss", "skip-gram + HS", "A", dict(hs=1, negative=0)),
                ("arch x loss", "CBOW + NS", "B", dict(hs=0, negative=10)),
                ("arch x loss", "CBOW + HS", "B", {})]
    for m in ("A", "B"):
        for w in (2, 5, 10):
            yield "window", f"{m}, window {w}", m, dict(window=w)
        for dim in (50, 100, 300):
            yield "dimensions", f"{m}, {dim} dims", m, dict(vector_size=dim)
    for k in (5, 10, 20):
        yield "negatives", f"A, {k} negatives", "A", dict(negative=k)


def config(base, over):
    cfg = {k: v for k, v in CONFIGS[base].items() if k != "name"}
    cfg.update(ARCH[base])
    cfg.update(over)
    return cfg


def run_one(sents, d, cfg, seed):
    cfg = {**cfg, "seed": seed}
    with schema.timed() as t:
        m = Word2Vec(sentences=sents, **cfg)
    ext, *_ = classify(m.wv, d)
    return {"wall_s": round(t.wall_s, 2), "acc": ext["sst2_test_acc"], "f1": ext["sst2_test_f1"],
            "wordsim353": evaluate.similarity(m.wv, "wordsim353")["spearman"],
            "men3k": evaluate.similarity(m.wv, "men3k")["spearman"]}


def summarise(runs):
    out = {}
    for k in runs[0]:
        vals = [r[k] for r in runs]
        out[k] = {"mean": round(st.mean(vals), 4), "sd": round(st.stdev(vals), 4)}
    return out


def main():
    d = data.load_sst2("phrases")
    sents = data.sentences(d["train"])
    rows, cache = [], {}
    for factor, label, base, over in settings():
        cfg = config(base, over)
        key = tuple(sorted((k, v) for k, v in cfg.items() if k != "seed"))
        if key not in cache:
            cache[key] = [run_one(sents, d, cfg, s) for s in SEEDS]
        s = summarise(cache[key])
        rows.append({"factor": factor, "label": label, "base": base, "overrides": over,
                     "seeds": list(SEEDS), "runs": cache[key], **s})
        print(f"{factor:12s} {label:22s} acc {s['acc']['mean']:.4f} ±{s['acc']['sd']:.4f}  "
              f"ws353 {s['wordsim353']['mean']:+.3f}  {s['wall_s']['mean']:.1f}s")
    (schema.RESULTS / "sweep.json").write_text(json.dumps(
        {"corpus": "sst2-phrases", "machine": schema.machine(), "rows": rows}, indent=2))
    print("wrote results/sweep.json")


if __name__ == "__main__":
    main()
