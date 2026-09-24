"""Run phases 6 to 8 across every trained model and augment results/*.json."""
import json
from pathlib import Path

from gensim.models import KeyedVectors, Word2Vec

from . import data, evaluate, pretrained, schema
from .classify import evaluate as classify
from .main import CORPUS, QUERIES

MODELS = Path("models")


def load_all_kv():
    for mid in ("A", "B", "G-ft"):
        yield mid, Word2Vec.load(str(MODELS / f"{mid}_{CORPUS}.model")).wv, f"{mid}_{CORPUS}"
    yield "C", KeyedVectors.load(str(MODELS / f"C_{CORPUS}.kv")), f"C_{CORPUS}"
    kv, _ = pretrained.load()
    yield "G", kv, "G_googlenews-100b"


def main():
    d = data.load_sst2("phrases")
    table = []
    for mid, kv, stem in load_all_kv():
        path = schema.RESULTS / f"{stem}.json"
        rec = json.loads(path.read_text())

        rec["intrinsic"] = rec.get("intrinsic", {})
        for bench in evaluate.BENCHMARKS:
            rec["intrinsic"][bench] = evaluate.similarity(kv, bench)

        ext, clf, prob, pred, y = classify(kv, d)
        rec["extrinsic"] = {**rec.get("extrinsic", {}), **ext,
                            "query_latency": evaluate.latency(kv, QUERIES),
                            "memory": evaluate.memory_model(
                                len(kv), kv.vector_size, rec["train"].get("peak_rss_mb"))}
        rec["errors"] = evaluate.mine_errors(d["test"], prob, pred, y)
        path.write_text(json.dumps(schema.validate(rec), indent=2))

        i, e = rec["intrinsic"], rec["extrinsic"]
        table.append((mid, i["wordsim353"], i["simlex999"], i["men3k"],
                      e["query_latency"], e["memory"], rec["errors"]))

    print(f"\n{'':3s} {'WordSim-353':>22s} {'SimLex-999':>22s} {'MEN-3k':>22s}")
    print(f"{'':3s} {'rho  cov':>22s} {'rho  cov':>22s} {'rho  cov':>22s}")
    for mid, ws, sl, men, *_ in table:
        f = lambda b: f"{b['spearman']:6.3f}  {b['coverage']:5.1%}"
        print(f"{mid:3s} {f(ws):>22s} {f(sl):>22s} {f(men):>22s}")

    print(f"\n{'':3s} {'vocab':>8s} {'p50 ms':>8s} {'p95 ms':>8s} {'pred MB':>9s} {'peak MB':>9s}")
    for mid, _, _, _, lat, mem, _ in table:
        print(f"{mid:3s} {lat['vocab_size']:>8,} {lat['p50_ms']:>8.2f} {lat['p95_ms']:>8.2f} "
              f"{mem['predicted_matrices_mb']:>9.1f} {str(mem['measured_peak_rss_mb']):>9s}")

    print(f"\n{'':3s} {'errors':>7s}  categories")
    for mid, *_, err in table:
        print(f"{mid:3s} {err['n_errors']:>7,}  {err['error_categories']}")


if __name__ == "__main__":
    main()
