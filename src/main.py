"""Runner: data -> models A, B, C -> pooling -> logistic regression -> results JSON."""
import json, sys

from . import data, pretrained, schema
from .classify import evaluate
from .train_gensim import train

QUERIES = ["plot", "terrible", "star", "cinematography", "the"]
CORPUS = "sst2-phrases"


def report(rec, ext):
    tr = rec["train"]
    print(f"\n{rec['model_id']}  {rec['name']}")
    wall = tr.get("wall_s") or tr.get("load_wall_s")
    print(f"  vocab {tr['vocab_size']:,}  {wall}s  peak {tr['peak_rss_mb']}MB")
    print(f"  acc {ext['sst2_test_acc']}  f1 {ext['sst2_test_f1']}  oov(test) {ext['oov_rate_test']}")
    for q, ns in rec["intrinsic"]["neighbours"].items():
        print(f"  {q:16s} {', '.join(w for w, _ in ns[:5])}")


def neighbours(kv):
    return {q: [[w, round(float(s), 3)] for w, s in kv.most_similar(q, topn=10)]
            for q in QUERIES if q in kv}


def main():
    d = data.load_sst2(train_on="phrases")
    print(json.dumps(data.stats(d), indent=2))
    sents = data.sentences(d["train"])

    for model_id in ("A", "B"):
        model, rec = train(model_id, sents, corpus_name=CORPUS)
        ext, *_ = evaluate(model.wv, d)
        rec["extrinsic"], rec["intrinsic"] = ext, {"neighbours": neighbours(model.wv)}
        schema.write(rec); report(rec, ext)

    kv, rec = pretrained.load()
    ext, *_ = evaluate(kv, d)
    rec["extrinsic"], rec["intrinsic"] = ext, {"neighbours": neighbours(kv)}
    schema.write(rec); report(rec, ext)


if __name__ == "__main__":
    sys.exit(main())
