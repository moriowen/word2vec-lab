"""PCA and t-SNE scatter plots of the vector space, emitted as inline SVG.

The handout asks to reduce the vectors to two dimensions and plot them, to see whether
semantically similar words cluster. Rendered as SVG rather than PNG so the figures inherit
the page's theme tokens and stay sharp in the PDF.

Words are drawn from fixed semantic groups so clustering is visible rather than asserted:
if the method works, group members land near each other.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

GROUPS = {
    "positive sentiment": ["good", "great", "wonderful", "brilliant", "charming",
                           "beautiful", "funny", "delightful"],
    "negative sentiment": ["bad", "terrible", "awful", "dull", "boring",
                           "stupid", "tedious", "bland"],
    "film craft": ["cinematography", "screenplay", "direction", "editing",
                   "performances", "script", "acting", "soundtrack"],
    "genre": ["comedy", "drama", "thriller", "documentary", "horror",
              "romance", "western", "musical"],
}
COLORS = ["var(--accent)", "var(--warm)", "var(--good)", "var(--bad)"]


def _project(kv, method, seed=42):
    words, labels = [], []
    for gi, (g, ws) in enumerate(GROUPS.items()):
        for w in ws:
            if w in kv:
                words.append(w); labels.append(gi)
    if len(words) < 6:
        return None
    X = np.array([kv[w] for w in words], dtype=np.float64)
    if method == "pca":
        red = PCA(n_components=2, random_state=seed)
        xy = red.fit_transform(X)
        extra = {"explained_variance": [round(float(v), 4)
                                        for v in red.explained_variance_ratio_]}
    else:
        # Perplexity must stay below the sample count; these groups give ~32 points.
        per = max(5, min(15, (len(words) - 1) // 3))
        xy = TSNE(n_components=2, random_state=seed, perplexity=per,
                  init="pca", max_iter=1000).fit_transform(X)
        extra = {"perplexity": per}
    return words, labels, xy, extra


def svg(kv, method="pca", w=720, h=380, pad=44):
    got = _project(kv, method)
    if got is None:
        return '<p class="caption">Too few of the plotted words are in this vocabulary.</p>', {}
    words, labels, xy, extra = got

    lo, hi = xy.min(0), xy.max(0)
    span = np.where(hi - lo == 0, 1, hi - lo)
    pts = (xy - lo) / span * np.array([w - 2 * pad, h - 2 * pad]) + pad
    pts[:, 1] = h - pts[:, 1]          # SVG y grows downward

    body = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{method.upper()} projection '
            f'of word vectors, coloured by semantic group.">',
            f'<rect x="0" y="0" width="{w}" height="{h}" fill="var(--raised)" rx="4"/>',
            '<g font-family="ui-monospace, Menlo, monospace" font-size="10">']
    for (x, y), lab, word in zip(pts, labels, words):
        c = COLORS[lab % len(COLORS)]
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{c}" opacity="0.85"/>')
        body.append(f'<text x="{x + 6:.1f}" y="{y + 3.4:.1f}" fill="var(--muted)">{word}</text>')
    for i, g in enumerate(GROUPS):
        body.append(f'<circle cx="{16}" cy="{18 + i * 17}" r="3.4" fill="{COLORS[i]}"/>')
        body.append(f'<text x="{26}" y="{21 + i * 17}" fill="var(--faint)">{g}</text>')
    body.append("</g></svg>")

    # How well the 2-D picture separates the groups, so the figure is not judged by eye alone.
    sil = None
    if len(set(labels)) > 1:
        from sklearn.metrics import silhouette_score
        sil = round(float(silhouette_score(xy, labels)), 3)
    return "".join(body), {"words_plotted": len(words), "silhouette_2d": sil, **extra}


def separation(kv):
    """Cosine silhouette in the full space, for all four groups and for the pairs that matter.

    The 2-D scores above measure the picture; these measure the vectors themselves, and the
    positive-against-negative pair shows whether a model tells praise from criticism.
    """
    from sklearn.metrics import silhouette_score
    g = list(GROUPS)
    pairs = {"all": g, "positive_vs_negative": g[:2], "craft_vs_genre": g[2:]}
    out = {}
    for name, groups in pairs.items():
        words = [(w, i) for i, k in enumerate(groups) for w in GROUPS[k] if w in kv]
        X = np.array([kv[w] for w, _ in words])
        out[name] = round(float(silhouette_score(X, [i for _, i in words], metric="cosine")), 3)
    return out


def build_all(models):
    out = {}
    for mid, kv in models:
        out[mid] = {"separation": separation(kv)}
        for method in ("pca", "tsne"):
            markup, meta = svg(kv, method)
            out[mid][method] = {"svg": markup, **meta}
    return out


def load_models():
    from gensim.models import KeyedVectors, Word2Vec
    from .pretrained import load
    m = Path("models")
    yield "A", Word2Vec.load(str(m / "A_sst2-phrases.model")).wv
    yield "B", Word2Vec.load(str(m / "B_sst2-phrases.model")).wv
    yield "C", KeyedVectors.load(str(m / "C_sst2-phrases.kv"))
    yield "G", load()[0]
    yield "G-ft", Word2Vec.load(str(m / "G-ft_sst2-phrases.model")).wv
    for mid in ("A", "B"):
        yield f"{mid}-text8", Word2Vec.load(str(m / f"{mid}_text8.model")).wv
    yield "C-text8", KeyedVectors.load(str(m / "C_text8.kv"))


if __name__ == "__main__":
    out = Path("results") / "visualization.json"
    out.write_text(json.dumps(build_all(load_models()), indent=2))
    print(f"wrote {out}")
