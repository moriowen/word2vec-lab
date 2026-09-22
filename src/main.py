"""Phase 0-3 runner: data -> models A and B -> pooling -> logistic regression -> results JSON."""
import json, sys

from . import data, schema
from .classify import evaluate
from .train_gensim import train

QUERIES = ["plot", "terrible", "star", "cinematography", "the"]


def main():
    d = data.load_sst2()
    print(json.dumps(data.stats(d), indent=2))
    sents = data.sentences(d["train"])

    for model_id in ("A", "B"):
        model, rec = train(model_id, sents)
        ext, *_ = evaluate(model.wv, d)
        rec["extrinsic"] = ext
        rec["intrinsic"] = {"neighbours": {
            q: [[w, round(float(s), 3)] for w, s in model.wv.most_similar(q, topn=10)]
            for q in QUERIES if q in model.wv}}
        path = schema.write(rec)
        print(f"\n{model_id}  {rec['name']}")
        print(f"  vocab {rec['train']['vocab_size']}  {rec['train']['wall_s']}s  "
              f"{rec['train']['words_per_sec']} w/s  {rec['train']['peak_rss_mb']}MB")
        print(f"  acc {ext['sst2_test_acc']}  f1 {ext['sst2_test_f1']}  oov(test) {ext['oov_rate_test']}")
        for q, ns in rec["intrinsic"]["neighbours"].items():
            print(f"  {q:16s} {', '.join(w for w, _ in ns[:5])}")
        print(f"  -> {path.name}")


if __name__ == "__main__":
    sys.exit(main())
