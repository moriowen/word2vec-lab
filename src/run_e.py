"""Train Model C, the from-scratch skip-gram, and score it through the same harness.

    python -m src.run_e            SST phrase corpus, dense Adam on CPU (the recorded run)
    python -m src.run_e --text8    text8, sparse Adam, pairs generated per epoch;
                                   checkpoints each epoch and resumes if interrupted
    --device cuda                  run on a GPU (the PACE ICE job in slurm/c_text8.sbatch)
"""
import sys
from pathlib import Path

from gensim.models.word2vec import Text8Corpus

from . import data, schema
from .classify import evaluate
from .main import CORPUS, neighbours, report
from .sgns import train, to_keyedvectors

MODEL_ID = "C"
CFG = dict(dim=300, window=5, negatives=10, min_count=2, sample=1e-3,
           epochs=10, alpha=0.001, min_alpha=1e-5, batch_size=8192, seed=42,
           sparse=False, device="cpu")
# Same hyperparameters; only the machinery changes. See sgns.train for the measurements.
TEXT8_CFG = {**CFG, "sparse": True, "precompute_pairs": False, "log_every": 2000}


def record(corpus_name, cfg, device, tokens, wall_s, vocab, history):
    return {
        "model_id": MODEL_ID,
        "name": "skip-gram + negative sampling, from scratch (PyTorch)",
        "corpus": corpus_name,
        "machine": f"{schema.machine()}/{device}",
        "hyperparams": {"sg": 1, "hs": 0, "negative": cfg["negatives"],
                        "vector_size": cfg["dim"], "window": cfg["window"],
                        "min_count": cfg["min_count"], "epochs": cfg["epochs"],
                        "alpha": cfg["alpha"], "sample": cfg["sample"],
                        "batch_size": cfg["batch_size"], "sparse": cfg["sparse"]},
        "train": {"wall_s": round(wall_s, 2),
                  "words_per_sec": round(tokens * cfg["epochs"] / wall_s),
                  "peak_rss_mb": round(schema.peak_rss_mb(), 1),
                  "vocab_size": len(vocab),
                  "corpus_tokens": tokens,
                  "loss_curve": history},
    }


def main_sst():
    d = data.load_sst2("phrases")
    sents = data.sentences(d["train"])
    tokens = sum(len(s) for s in sents)
    model, vocab, history, device, wall_s = train(sents, **CFG)
    kv = to_keyedvectors(model, vocab)
    rec = record(CORPUS, CFG, device, tokens, wall_s, vocab, history)
    ext, *_ = evaluate(kv, d)
    rec["extrinsic"], rec["intrinsic"] = ext, {"neighbours": neighbours(kv)}
    schema.write(rec)
    report(rec, ext)
    print(f"  device {device}  loss {history[0]} -> {history[-1]}")
    kv.save(str(Path("models") / f"{MODEL_ID}_{CORPUS}.kv"))


def main_text8():
    from .run_text8 import CORPUS as T8, fetch, score
    corpus = Text8Corpus(str(fetch()))
    tokens = sum(len(s) for s in corpus)
    ckpt = Path("models") / f"{MODEL_ID}_{T8}.ckpt"
    cfg = {**TEXT8_CFG, "device": device_arg() or TEXT8_CFG["device"]}
    model, vocab, history, device, wall_s = train(corpus, **cfg, checkpoint=ckpt)
    kv = to_keyedvectors(model, vocab)
    kv.save(str(Path("models") / f"{MODEL_ID}_{T8}.kv"))
    rec = record(T8, TEXT8_CFG, device, tokens, wall_s, vocab, history)
    ext = score(kv, rec, data.load_sst2("phrases"))
    schema.write(rec)
    ckpt.unlink()
    print(f"{MODEL_ID}_{T8}: vocab {len(vocab):,}  {wall_s:.0f}s  acc {ext['sst2_test_acc']}  "
          f"ws353 {rec['intrinsic']['wordsim353']['spearman']}  "
          f"analogy {rec['analogies']['benchmark']['overall_accuracy']}  loss {history}")


def device_arg():
    return sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else None


if __name__ == "__main__":
    main_text8() if "--text8" in sys.argv else main_sst()
