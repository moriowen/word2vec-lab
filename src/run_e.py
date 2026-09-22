"""Train Model E and score it through the same harness as every other model."""
import numpy as np

from . import data, schema
from .classify import evaluate
from .main import CORPUS, QUERIES, neighbours, report
from .sgns import train, to_keyedvectors

CFG = dict(dim=300, window=5, negatives=10, min_count=2, sample=1e-3,
           epochs=10, alpha=0.001, min_alpha=1e-5, batch_size=8192, seed=42,
           sparse=False, device="cpu")


def main():
    d = data.load_sst2("phrases")
    sents = data.sentences(d["train"])
    tokens = sum(len(s) for s in sents)

    with schema.timed() as t:
        model, vocab, history, device = train(sents, **CFG)
    kv = to_keyedvectors(model, vocab)

    rec = {
        "model_id": "E",
        "name": "skip-gram + negative sampling, from scratch (PyTorch)",
        "corpus": CORPUS,
        "machine": f"{schema.machine()}/{device}",
        "hyperparams": {"sg": 1, "hs": 0, "negative": CFG["negatives"],
                        "vector_size": CFG["dim"], "window": CFG["window"],
                        "min_count": CFG["min_count"], "epochs": CFG["epochs"],
                        "alpha": CFG["alpha"], "sample": CFG["sample"],
                        "batch_size": CFG["batch_size"]},
        "train": {"wall_s": round(t.wall_s, 2),
                  "words_per_sec": round(tokens * CFG["epochs"] / t.wall_s),
                  "peak_rss_mb": round(schema.peak_rss_mb(), 1),
                  "vocab_size": len(vocab),
                  "corpus_tokens": tokens,
                  "loss_curve": history},
    }
    ext, *_ = evaluate(kv, d)
    rec["extrinsic"], rec["intrinsic"] = ext, {"neighbours": neighbours(kv)}
    schema.write(rec)
    report(rec, ext)
    print(f"  device {device}  loss {history[0]} -> {history[-1]}")
    kv.save(str(__import__("pathlib").Path("models") / f"E_{CORPUS}.kv"))


if __name__ == "__main__":
    main()
