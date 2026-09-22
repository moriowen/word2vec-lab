"""Isolated per-model memory measurement.

Peak RSS of a whole process includes the interpreter, gensim, numpy, torch and whatever
corpus happens to be resident, so it is not a measurement of the model. This runs one model
per subprocess and reports the RSS delta across the load, against the size of the matrices
actually retained.

A trained Word2Vec keeps two matrices while training (W_in and W_out) and a KeyedVectors
keeps one, so the prediction differs by which object is being measured.
"""
import json, subprocess, sys

CHILD = '''
import json, resource, sys
def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3
import numpy, gensim
from gensim.models import KeyedVectors, Word2Vec
base = rss_mb()
which = sys.argv[1]
if which == "C":
    from src.pretrained import GZ, LIMIT
    kv = KeyedVectors.load_word2vec_format(str(GZ), binary=True, limit=LIMIT); n = 1
elif which == "E":
    kv = KeyedVectors.load("models/E_sst2-phrases.kv"); n = 1
else:
    m = Word2Vec.load(f"models/{which}_sst2-phrases.model"); kv = m.wv; n = 2
after = rss_mb()
V, d = len(kv), kv.vector_size
print(json.dumps({"model_id": which, "vocab": V, "dim": d, "matrices_retained": n,
                  "predicted_mb": round(n * V * d * 4 / 1e6, 1),
                  "baseline_rss_mb": round(base, 1),
                  "loaded_rss_mb": round(after, 1),
                  "delta_mb": round(after - base, 1)}))
'''


def main():
    rows = []
    for mid in ("A", "B", "D", "E", "C"):
        out = subprocess.run([sys.executable, "-c", CHILD, mid],
                             capture_output=True, text=True)
        rows.append(json.loads(out.stdout.strip().splitlines()[-1]))

    print(f"{'':3s} {'vocab':>8s} {'keeps':>6s} {'predicted':>10s} {'baseline':>9s} "
          f"{'loaded':>8s} {'delta':>8s} {'ratio':>6s}")
    for r in rows:
        ratio = r["delta_mb"] / r["predicted_mb"] if r["predicted_mb"] else 0
        print(f"{r['model_id']:3s} {r['vocab']:>8,} {r['matrices_retained']:>6d} "
              f"{r['predicted_mb']:>10.1f} {r['baseline_rss_mb']:>9.1f} "
              f"{r['loaded_rss_mb']:>8.1f} {r['delta_mb']:>8.1f} {ratio:>6.2f}")

    from . import schema
    for r in rows:
        stem = "C_googlenews-100b" if r["model_id"] == "C" else f"{r['model_id']}_sst2-phrases"
        p = schema.RESULTS / f"{stem}.json"
        rec = json.loads(p.read_text())
        rec["extrinsic"]["memory"] = r
        p.write_text(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
