"""Model C. GoogleNews-300, the required pretrained reference.

GoogleNews rather than GloVe deliberately: GloVe factorises a co-occurrence matrix and is
not a Word2Vec model, so using it here would quietly answer a different question than the
handout asks. GoogleNews-300 is genuine skip-gram with negative sampling, trained by Mikolov
et al. on roughly 100 billion tokens of news text.

Source: https://code.google.com/archive/p/word2vec/ , fetched via gensim-data as
`word2vec-google-news-300`.
"""
from pathlib import Path
from gensim.models import KeyedVectors

from . import schema

GZ = Path.home() / "gensim-data" / "word2vec-google-news-300" / "word2vec-google-news-300.gz"

# The full file is 3 million words and wants roughly 4 GB resident, which is most of an 8 GB
# laptop. The top 500k covers essentially all of SST. The limit is reported, because it
# changes the brute-force query latency numbers.
LIMIT = 500_000


def load(limit: int = LIMIT):
    with schema.timed() as t:
        kv = KeyedVectors.load_word2vec_format(str(GZ), binary=True, limit=limit)
    rec = {
        "model_id": "C",
        "name": "GoogleNews-300 (pretrained, frozen)",
        "corpus": "googlenews-100b",
        "machine": schema.machine(),
        "hyperparams": {"sg": 1, "hs": 0, "negative": 5, "vector_size": 300,
                        "window": 5, "min_count": 5, "epochs": None, "alpha": 0.025,
                        "sample": 1e-5, "note": "as published by Mikolov et al.; not retrained here"},
        "train": {"wall_s": None, "words_per_sec": None,
                  "load_wall_s": round(t.wall_s, 2),
                  "peak_rss_mb": round(schema.peak_rss_mb(), 1),
                  "vocab_size": len(kv),
                  "vocab_limit": limit,
                  "corpus_tokens": 100_000_000_000},
    }
    return kv, rec
