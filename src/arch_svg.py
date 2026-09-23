"""Architecture figures for deliverable 1: skip-gram and CBOW, input through code to output."""

_DEFS = """
  <defs>
    <marker id="a{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
            orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--line-2)"/></marker>
  </defs>"""

_STYLE = 'font-family="ui-monospace, Menlo, monospace" font-size="11"'


def _box(x, y, w, h, title, sub, accent=False):
    fill = "var(--accent-soft)" if accent else "var(--raised)"
    stroke = "var(--accent)" if accent else "var(--line)"
    out = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{stroke}"/>'
    out += f'<text x="{x + w/2}" y="{y + 22}" fill="var(--ink)" font-size="12" text-anchor="middle">{title}</text>'
    if sub:
        out += f'<text x="{x + w/2}" y="{y + 38}" fill="var(--faint)" text-anchor="middle">{sub}</text>'
    return out


def _arrow(x1, y1, x2, y2, k, label=None):
    out = (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--line-2)" '
           f'stroke-width="1.4" marker-end="url(#a{k})"/>')
    if label:
        out += (f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2 - 7}" fill="var(--muted)" '
                f'font-size="10" text-anchor="middle">{label}</text>')
    return out


def skipgram(V, d, window, k):
    """One centre word in, C context slots out."""
    s = [f'<svg viewBox="0 0 740 320" role="img" aria-label="Skip-gram architecture: a one-hot '
         f'input of size V multiplies through W_in to a {d}-dimensional code, then through W_out '
         f'to {2*window} context predictions.">', _DEFS.format(k="sg"), f'<g {_STYLE}>']

    s.append(_box(10, 128, 132, 56, "one-hot", f"V = {V:,}"))
    s.append(_arrow(146, 156, 204, 156, "sg", "W_in"))
    s.append(_box(208, 128, 132, 56, "code", f"d = {d}", accent=True))
    s.append(_arrow(344, 156, 402, 156, "sg", "W_out"))

    ys = [22, 84, 146, 208, 270]
    labels = ["context -{w}".format(w=window), "...", "context -1",
              "context +1", "... context +{w}".format(w=window)]
    for y, lab in zip(ys, labels):
        s.append(_box(406, y, 150, 44, lab, None))
        s.append(_arrow(392, 156, 402, y + 22, "sg"))

    s.append(_box(586, 128, 140, 56, f"{k} negatives", "per positive"))
    s.append(f'<text x="656" y="206" fill="var(--faint)" text-anchor="middle">sampled from</text>')
    s.append(f'<text x="656" y="220" fill="var(--faint)" text-anchor="middle">unigram^0.75</text>')
    s.append('<text x="276" y="204" fill="var(--muted)" text-anchor="middle" font-size="10">'
             'encoder output, kept</text>')
    s.append('<text x="276" y="112" fill="var(--muted)" text-anchor="middle" font-size="10">'
             'no nonlinearity</text>')
    s.append("</g></svg>")
    return "".join(s)


def cbow(V, d, window):
    """C context words averaged in, one centre word out."""
    s = [f'<svg viewBox="0 0 740 320" role="img" aria-label="CBOW architecture: {2*window} '
         f'one-hot context inputs multiply through W_in and are averaged into a {d}-dimensional '
         f'code, then a hierarchical softmax tree predicts the centre word.">',
         _DEFS.format(k="cb"), f'<g {_STYLE}>']

    ys = [22, 84, 146, 208, 270]
    labels = [f"context -{window}", "...", "context -1", "context +1", f"... context +{window}"]
    for y, lab in zip(ys, labels):
        s.append(_box(10, y, 150, 44, lab, None))
        s.append(_arrow(164, y + 22, 214, 156, "cb"))

    s.append(_box(218, 128, 122, 56, "mean", "of C rows"))
    s.append('<text x="279" y="112" fill="var(--muted)" text-anchor="middle" font-size="10">'
             'W_in, shared</text>')
    s.append(_arrow(344, 156, 396, 156, "cb"))
    s.append(_box(400, 128, 122, 56, "code", f"d = {d}", accent=True))
    s.append(_arrow(526, 156, 578, 156, "cb", "W_out"))
    s.append(_box(582, 118, 148, 76, "Huffman tree", f"log2(V) ~ {round(V.bit_length())} nodes"))
    s.append('<text x="656" y="214" fill="var(--faint)" text-anchor="middle">hierarchical softmax</text>')
    s.append('<text x="461" y="204" fill="var(--muted)" text-anchor="middle" font-size="10">'
             'encoder output, kept</text>')
    s.append("</g></svg>")
    return "".join(s)
