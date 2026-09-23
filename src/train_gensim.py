"""Models A and B. Everything but architecture and loss is held constant between them."""
from pathlib import Path
from gensim.models import Word2Vec

from . import schema

MODELS = Path(__file__).resolve().parent.parent / "models"

BASE = dict(vector_size=300, window=5, min_count=2, epochs=10, alpha=0.025,
            sample=1e-3, seed=42, workers=4)

CONFIGS = {
    "A": {"name": "gensim skip-gram + negative sampling", **BASE, "sg": 1, "hs": 0, "negative": 10},
    "B": {"name": "gensim CBOW + hierarchical softmax", **BASE, "sg": 0, "hs": 1, "negative": 0},
    # Control for the A-versus-D comparison. Identical to D in every hyperparameter, including
    # the lower learning rate and shorter schedule, and differing only in that W_in starts
    # random instead of warm-started from GoogleNews. A against D confounds initialisation
    # with the optimiser settings; A2 against D does not.
    "A2": {"name": "gensim skip-gram + NS, D's schedule, random init", **BASE,
           "sg": 1, "hs": 0, "negative": 10, "epochs": 8, "alpha": 0.005, "min_alpha": 0.0001},
}


def train(model_id: str, corpus_sents: list[list[str]], corpus_name: str = "sst2"):
    cfg = dict(CONFIGS[model_id])
    name = cfg.pop("name")
    tokens = sum(len(s) for s in corpus_sents)

    with schema.timed() as t:
        model = Word2Vec(sentences=corpus_sents, **cfg)

    MODELS.mkdir(exist_ok=True)
    model.save(str(MODELS / f"{model_id}_{corpus_name}.model"))

    rec = {
        "model_id": model_id,
        "name": name,
        "corpus": corpus_name,
        "machine": schema.machine(),
        # Read back off the model object, never retyped, so the report cannot disagree with the run.
        "hyperparams": {k: getattr(model, k, cfg.get(k)) for k in
                        ("sg", "hs", "negative", "vector_size", "window", "min_count", "epochs", "alpha", "sample")},
        "train": {"wall_s": round(t.wall_s, 2),
                  "words_per_sec": round(tokens * cfg["epochs"] / t.wall_s),
                  "peak_rss_mb": round(schema.peak_rss_mb(), 1),
                  "vocab_size": len(model.wv),
                  "corpus_tokens": tokens},
    }
    return model, rec
