"""Model G-ft. GoogleNews-300 warm-started, then trained on SST.

    python -m src.finetune   # trains G-ft and its A2 control; run_eval adds the rest

This is what the handout calls fine-tuning a pretrained Word2Vec model. Initialise W_in from
the published vectors, then keep running the ordinary word2vec loop on the movie-review
corpus at a low learning rate.

Two properties of the recipe are worth stating because they shape the result:

  1. `intersect_word2vec_format` only overwrites rows for words already in the vocabulary
     built from our corpus. It does not import Google's 3 million word vocabulary.
  2. It loads W_in only. Google never published their output matrix, so W_out stays randomly
     initialised and the first updates are partly spent re-learning a decoder. That pushes
     the imported vectors around more than a true fine-tune would, which is why alpha is low
     and the epoch count small, and why drift is measured rather than assumed.
"""
import numpy as np
from gensim.models import Word2Vec

from . import data, schema
from .classify import evaluate
from .main import CORPUS, neighbours, report
from .pretrained import GZ
from .train_gensim import MODELS, train

CFG = dict(vector_size=300, window=5, min_count=2, sg=1, hs=0, negative=10,
           alpha=0.005, min_alpha=0.0001, epochs=8, sample=1e-3, seed=42, workers=4)


def _cos(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    out = np.zeros(len(a))
    out[ok] = np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok])
    return out


def finetune(corpus_sents, corpus_name="sst2-phrases"):
    model = Word2Vec(**CFG)
    model.build_vocab(corpus_sents)
    # gensim 4.x does not create vectors_lockf by default; without it the intersect either
    # fails or silently freezes every row. 1.0 means trainable.
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)

    before_all = model.wv.vectors.copy()
    model.wv.intersect_word2vec_format(str(GZ), binary=True, lockf=1.0)
    warm = model.wv.vectors.copy()
    # A row the intersect touched is one that changed from its random init.
    seeded = ~np.all(np.isclose(warm, before_all), axis=1)

    tokens = sum(len(s) for s in corpus_sents)
    with schema.timed() as t:
        model.train(corpus_sents, total_examples=model.corpus_count, epochs=CFG["epochs"])

    drift = _cos(warm[seeded], model.wv.vectors[seeded])
    words = [model.wv.index_to_key[i] for i in np.where(seeded)[0]]
    order = np.argsort(drift)

    MODELS.mkdir(exist_ok=True)
    model.save(str(MODELS / f"G-ft_{corpus_name}.model"))

    rec = {
        "model_id": "G-ft",
        "name": "GoogleNews-300 warm start, fine-tuned on SST",
        "corpus": corpus_name,
        "machine": schema.machine(),
        "hyperparams": {k: v for k, v in CFG.items() if k != "workers"},
        "train": {"wall_s": round(t.wall_s, 2),
                  "words_per_sec": round(tokens * CFG["epochs"] / t.wall_s),
                  "peak_rss_mb": round(schema.peak_rss_mb(), 1),
                  "vocab_size": len(model.wv),
                  "corpus_tokens": tokens},
        "finetune": {
            "seeded_from_pretrained": int(seeded.sum()),
            "randomly_initialised": int((~seeded).sum()),
            # Cosine between warm-start and final vector: 1.0 means the word did not move.
            # Named for what it holds (a similarity) rather than "drift", which reads inverted.
            "mean_cosine_before_after": round(float(drift.mean()), 4),
            "median_cosine_before_after": round(float(np.median(drift)), 4),
            "moved_most": [[words[i], round(float(drift[i]), 3)] for i in order[:10]],
            "moved_least": [[words[i], round(float(drift[i]), 3)] for i in order[-10:]],
        },
    }
    return model, rec


def main():
    d = data.load_sst2(train_on="phrases")
    sents = data.sentences(d["train"])
    # A2 is the same schedule with random init, so it is trained here, next to what it controls.
    for run in (lambda: finetune(sents, CORPUS), lambda: train("A2", sents, corpus_name=CORPUS)):
        model, rec = run()
        ext, *_ = evaluate(model.wv, d)
        rec["extrinsic"], rec["intrinsic"] = ext, {"neighbours": neighbours(model.wv)}
        schema.write(rec); report(rec, ext)


if __name__ == "__main__":
    main()
