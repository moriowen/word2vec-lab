"""Models A and B retrained on text8, with SST-2 still the sentiment test.

Every hyperparameter matches the SST runs, so the corpus is the only thing that changes.
Records go to results/{A,B}_text8.json and carry their own analogy results, so rerunning
src.run_analogies cannot drop them.
"""
import json
import urllib.request
import zipfile
from pathlib import Path

from gensim.models.word2vec import Text8Corpus

from . import analogies, data, evaluate, schema
from .classify import evaluate as classify
from .main import QUERIES, neighbours
from .train_gensim import MODELS, train

TEXT8 = Path(__file__).resolve().parent.parent / "data" / "text8"
URL = "http://mattmahoney.net/dc/text8.zip"
CORPUS = "text8"


def fetch() -> Path:
    path = TEXT8 / "text8"
    if not path.exists():
        TEXT8.mkdir(parents=True, exist_ok=True)
        zpath = TEXT8 / "text8.zip"
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(TEXT8)
        zpath.unlink()
    return path


def oov_profile():
    """What the text8 vocabulary misses in the SST-2 test split, by kind of token."""
    from collections import Counter
    from gensim.models import Word2Vec
    kv = Word2Vec.load(str(MODELS / f"A_{CORPUS}.model")).wv
    kv_sst = Word2Vec.load(str(MODELS / "A_sst2-phrases.model")).wv
    test = data.load_sst2("phrases")["test"]
    toks = [w for t, _ in test for w in t]
    oov = [w for w in toks if w not in kv]

    def kind(w):
        if "'" in w:
            return "apostrophe"
        if not any(ch.isalpha() for ch in w):
            return "punctuation or number"
        return "hyphenated" if "-" in w else "word"

    kinds = Counter(kind(w) for w in oov)
    review = Counter(w for w in toks if w.isalpha() and w in kv_sst and w not in kv)
    out = {"test_tokens": len(toks), "oov_tokens": len(oov),
           "kinds": {k: {"tokens": n, "share": round(n / len(oov), 3)} for k, n in kinds.items()},
           "top_oov": Counter(oov).most_common(20),
           "sentences_with_nt": sum(1 for t, _ in test if "n't" in t), "test_sentences": len(test),
           "sst_only_words": review.most_common(20)}
    (schema.RESULTS / "text8_oov.json").write_text(json.dumps(out, indent=2))
    print("wrote results/text8_oov.json")


def score(kv, rec, d):
    """Every evaluation the SST models get, written into rec; returns the classifier metrics."""
    ext, clf, prob, pred, y = classify(kv, d)
    rec["extrinsic"] = {**ext, "query_latency": evaluate.latency(kv, QUERIES),
                        "memory": evaluate.memory_model(len(kv), kv.vector_size,
                                                        rec["train"]["peak_rss_mb"])}
    rec["intrinsic"] = {"neighbours": neighbours(kv),
                        **{b: evaluate.similarity(kv, b) for b in evaluate.BENCHMARKS}}
    rec["errors"] = evaluate.mine_errors(d["test"], prob, pred, y)
    rec["analogies"] = {"arithmetic": analogies.arithmetic(kv),
                        "benchmark": analogies.benchmark(kv)}
    return ext


def main():
    corpus = Text8Corpus(str(fetch()))
    d = data.load_sst2("phrases")
    for mid in ("A", "B"):
        model, rec = train(mid, corpus, corpus_name=CORPUS)
        ext = score(model.wv, rec, d)
        schema.write(rec)
        t, i = rec["train"], rec["intrinsic"]
        print(f"{mid}_text8: vocab {t['vocab_size']:,}  tokens {t['corpus_tokens']:,}  "
              f"{t['wall_s']}s  acc {ext['sst2_test_acc']}  oov(test) {ext['oov_rate_test']}  "
              f"ws353 {i['wordsim353']['spearman']}  "
              f"analogy {rec['analogies']['benchmark']['overall_accuracy']}")
        print("   ", json.dumps({q: [w for w, _ in n[:5]] for q, n in i["neighbours"].items()}))
    oov_profile()


if __name__ == "__main__":
    import sys
    oov_profile() if "--oov" in sys.argv else main()
