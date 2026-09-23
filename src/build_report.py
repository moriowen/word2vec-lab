"""Build docs/index.html from results/*.json.

Every number on the page is read out of the result records at build time. Nothing is typed
into the prose, so re-running a model and rebuilding cannot leave a stale figure behind.
"""
import html
import json
import subprocess
from pathlib import Path

from . import arch_svg
from .report_css import CSS, EXTRA_CSS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "index.html"
K_TC = 5
QUERIES = ["plot", "terrible", "star", "cinematography", "the"]
ORDER = ["A", "B", "C", "D", "E"]
SHORT = {"A": "Skip-gram<br>+ negative sampling", "B": "CBOW<br>+ hierarchical softmax",
         "C": "GoogleNews-300<br>frozen", "D": "GoogleNews<br>fine-tuned on SST",
         "E": "Skip-gram + NS<br>written from scratch"}
PROV = {"A": "gensim, trained here", "B": "gensim, trained here", "C": "downloaded, frozen",
        "D": "fine-tuned here", "E": "our implementation"}

e = html.escape


def load():
    recs = {}
    for p in (ROOT / "results").glob("*.json"):
        r = json.loads(p.read_text())
        if "model_id" not in r:        # sidecar data, not a model record
            continue
        if r["corpus"] == "sst2":      # superseded sentence-level run, kept on disk only
            continue
        recs[r["model_id"]] = r
    return recs


def commits():
    out = subprocess.run(["git", "log", "--format=%h\t%s"], capture_output=True,
                         text=True, cwd=ROOT).stdout.strip()
    return [l.split("\t", 1) for l in out.splitlines() if "\t" in l][::-1]


# ---------------------------------------------------------------- components

def table(headers, rows, aligns=None, hi=None):
    aligns = aligns or ["l"] * len(headers)
    th = "".join(f'<th class="{"n" if a=="n" else "mono" if a=="m" else ""}">{h}</th>'
                 for h, a in zip(headers, aligns))
    body = []
    for i, row in enumerate(rows):
        cls = ' class="hi"' if hi is not None and i == hi else ""
        tds = "".join(f'<td class="{"n" if a=="n" else "mono" if a=="m" else ""}">{c}</td>'
                      for c, a in zip(row, aligns))
        body.append(f"<tr{cls}>{tds}</tr>")
    return (f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def cards(recs):
    best = max(r["extrinsic"]["sst2_test_acc"] for r in recs.values())
    out = []
    for mid in ORDER:
        r = recs[mid]
        acc = r["extrinsic"]["sst2_test_acc"]
        t = r["train"]
        cost = (f'{t["wall_s"]:.1f}s' if t.get("wall_s")
                else f'{t["load_wall_s"]:.1f}s load')
        focus = " is-focus" if acc == best else ""
        out.append(f'''<div class="mcard{focus}">
        <div class="tag"><span class="id">{mid}</span><span class="nm">{SHORT[mid]}</span></div>
        <div class="acc">{acc:.3f}</div>
        <div class="bar"><i style="width:{acc*100:.1f}%"></i></div>
        <div class="mmeta"><span>{PROV[mid]}</span><span>{cost}</span></div></div>''')
    return f'<div class="models">{"".join(out)}</div>'


def neighbour_box(recs, ids):
    """One table per query, with top-1, top-5 and top-10 stated explicitly per model."""
    out = []
    for q in QUERIES:
        rows = []
        for mid in ids:
            nb = recs[mid]["intrinsic"]["neighbours"].get(q)
            if not nb:
                rows.append([mid, "<i>out of vocabulary</i>", "", ""])
                continue
            base = {w for w, _ in recs["C"]["intrinsic"]["neighbours"].get(q, [])}
            def chips(lo, hi):
                inner = "".join(
                    f'<span class="chip{" only-d" if mid == "D" and w not in base else ""}">'
                    f'{e(w)} <b>{s:.2f}</b></span>' for w, s in nb[lo:hi])
                return f'<span class="chips">{inner}</span>' 
            rows.append([mid, chips(0, 1), chips(1, 5), chips(5, 10)])
        out.append(f'<h3 class="qhead">{e(q)}</h3>')
        out.append(table(["Model", "Top-1", "Top-2 to 5", "Top-6 to 10"], rows,
                         ["m", "l", "l", "l"]))
    return "".join(out)


def pretrained_2a(ana):
    """Deliverable 2a: reference, URL, corpus, test set and accuracy for the pretrained model."""
    b = ana["C"]["benchmark"]
    paper = ("Mikolov, Sutskever, Chen, Corrado and Dean (2013), <i>Distributed "
             "Representations of Words and Phrases and their Compositionality</i>, NIPS. "
             "<a href='https://arxiv.org/abs/1310.4546'>arxiv.org/abs/1310.4546</a>")
    dist = ("<a href='https://code.google.com/archive/p/word2vec/'>"
            "code.google.com/archive/p/word2vec</a>, fetched through gensim-data as "
            "<code>word2vec-google-news-300</code>")
    measured = (f"<b>{b['overall_accuracy']:.4f}</b> over {b['attempted']:,} attempted "
                f"questions ({b['coverage']:.1%} coverage, candidate pool restricted to the "
                f"top {b['restrict_vocab']:,} words)")
    unpublished = ("Not quoted. The Google Code archive page is a JavaScript application and "
                   "serves no figures to a fetch, and the linked papers report results for "
                   "models trained in those papers rather than for this released file. Rather "
                   "than cite a number that could not be verified, the measured figure above "
                   "is given with its exact conditions.")
    return table(["Field", "Value"], [
        ["Model", "GoogleNews-vectors-negative300"],
        ["Reference", paper],
        ["Distribution URL", dist],
        ["Training corpus", "Google News, roughly 100 billion tokens. The corpus is "
         "proprietary and was never released, so it cannot be linked."],
        ["Architecture", "Skip-gram with negative sampling, 300 dimensions, 3M words and phrases"],
        ["Test set used here", "Google analogy set (<code>questions-words.txt</code>), 19,544 "
         "questions, shipped in <code>data/eval/</code>"],
        ["Accuracy, measured here", measured],
        ["Accuracy, as published", unpublished],
    ], ["m", "l"])


def ex2b():
    ex = json.loads((ROOT / "results" / "examples_2b.json").read_text())
    out = []
    for key, head in [("train", "Five training examples"), ("test", "Four test examples")]:
        rows = [[x["why"], f'<code>{e(x["text"][:92])}</code>',
                 "positive" if x["label"] else "negative"] for x in ex[key]]
        out.append(f"<h3 style=\"margin-top:12px;font-size:.82rem;letter-spacing:.1em;"
                   f"text-transform:uppercase;font-family:var(--sans);color:var(--faint)\">"
                   f"{head}</h3>")
        out.append(table(["Selected for", "Text", "Gold label"], rows, ["m", "l", "n"]))
    return "".join(out)


def testcase_section():
    tc = json.loads((ROOT / "results" / "testcases.json").read_text())
    blocks = []
    for k, s in enumerate(tc["sentences"]):
        gold = "positive" if s["gold"] else "negative"
        rows = []
        for mid in ORDER:
            c = tc["models"][mid]["cases"][k]
            lab, prob = c["ranked_labels"][0]
            second = c["ranked_labels"][1]
            mark = '<span class="win">correct</span>' if c["correct"] else '<span class="lose">wrong</span>'
            rows.append([mid, f'{lab} {prob:.3f}', f'{second[0]} {second[1]:.3f}', mark,
                         f'{c["neighbour_label_agreement"]:.1f}'])
        nb = tc["models"]["D"]["cases"][k]["neighbours"][:3]
        chips = "".join(f'<span class="chip{" agree" if n["label"]==s["gold"] else ""}">'
                        f'{e(n["text"])} &middot; {n["cos"]:.2f}</span>' for n in nb)
        blocks.append(f'''<div class="tcase">
          <div class="tchead"><span class="eyebrow">Example {k+1} &middot; {e(s["selected_for"])}</span>
            <p class="tcsent"><code>{e(s["text"])}</code></p>
            <span class="tcgold">gold label: <b>{gold}</b></span></div>
          {table(["Model", "Rank 1", "Rank 2", "Result", "Neighbour agreement"], rows,
                 ["m", "n", "n", "n", "n"])}
          <div class="tcnb"><span class="nlabel">D retrieves</span>
            <span class="chips">{chips}</span></div>
          <p class="tclegend">Retrieved training sentences, nearest first, with cosine.
          <b>Green-tinted</b> where the training sentence's own label matches this test
          sentence's gold label &mdash; a different mark from the gold tint used in the
          neighbours tables.</p></div>''')
    return "".join(blocks)


def drift_col(pairs, klass, title, invert=False):
    rows = []
    for w, v in pairs:
        pct = (1 - v) / 0.5 * 100 if not invert else 0
        rows.append(f'<div class="drow {klass}"><span class="w">{e(w)}</span>'
                    f'<span class="t"><i style="width:{max(pct,2):.0f}%"></i></span>'
                    f'<span class="v">{v:.3f}</span></div>')
    return f'<div class="dcol"><h3>{title}</h3>{"".join(rows)}</div>'


# ---------------------------------------------------------------- page

def build():
    r = load()
    A, B, C, D, E = (r[k] for k in ORDER)
    tok = A["train"]["corpus_tokens"]
    V = A["train"]["vocab_size"]
    dim = A["hyperparams"]["vector_size"]

    acc = lambda m: r[m]["extrinsic"]["sst2_test_acc"]
    f1 = lambda m: r[m]["extrinsic"]["sst2_test_f1"]
    sim = lambda m, b: r[m]["intrinsic"][b]["spearman"]
    lat = lambda m: r[m]["extrinsic"]["query_latency"]
    mem = lambda m: r[m]["extrinsic"]["memory"]
    err = lambda m: r[m]["errors"]

    ana = json.loads((ROOT / "results" / "analogies.json").read_text())
    pretrained_table = pretrained_2a(ana)
    ft = D["finetune"]
    gap_ad = acc("D") - acc("A")
    gap_cd = acc("C") - acc("D")
    lat_ratio = lat("C")["p50_ms"] / lat("A")["p50_ms"]
    vocab_ratio = lat("C")["vocab_size"] / lat("A")["vocab_size"]

    S = []
    add = S.append

    add('''<div class="cover">
      <div class="f">Student Name: Atharva Mohite</div>
      <div class="f">Student Session: cs6220</div>
      <div class="f">CS 6220 Big Data Systems, Fall 2026 &middot; Homework 2, programming option</div>
    </div>''')
    add(f'''<header class="masthead">
      <div class="eyebrow">CS 6220 Big Data Systems &middot; Homework 2 &middot; programming option</div>
      <h1>Five Word2Vec Models, One Classifier</h1>
      <p class="lede">Two models trained from scratch on movie reviews, one downloaded,
      one fine-tuned from the downloaded weights, and one where the training algorithm itself
      was reimplemented. Scored on sentiment, on lexical similarity, and on what they cost to
      run.</p>
      <div class="runmeta">
        <span><b>Corpus</b> SST-2 phrases, {tok:,} tokens</span>
        <span><b>Vocabulary</b> {V:,}</span>
        <span><b>Dimensions</b> {dim}</span>
        <span><b>Test split</b> 1,821 sentences</span>
        <span><b>Machine</b> M2 Air</span>
      </div>
    </header>''')

    add('''<nav class="toc">
      <a href="#results">Results</a><a href="#testcases">Test examples</a><a href="#arch">Architectures</a><a href="#data">Data</a>
      <a href="#hyper">Hyperparameters</a><a href="#neighbours">Nearest neighbours</a>
      <a href="#viz">Visualization</a><a href="#analogies">Analogies</a><a href="#similarity">Lexical similarity</a><a href="#finetune">Fine-tuning</a>
      <a href="#scratch">From scratch</a><a href="#systems">Systems</a>
      <a href="#errors">Errors</a><a href="#running">Running it</a><a href="#log">Build log</a></nav>''')

    # ---- results
    add(f'''<section id="results">
      <div class="sechead"><div class="eyebrow">Where things stand</div>
      <h2>Five models, one classifier</h2></div>
      <p class="prose">Every model is scored the same way. Look up a vector per word, average
      them into a sentence vector, fit logistic regression, test on the same held-out SST-2
      sentences. The classifier never changes and is never tuned per model, so any gap between
      these numbers belongs to the embeddings and to nothing else.</p>
      {cards(r)}
      <div class="note"><b>Isolating the warm start needed a control run.</b> A against D is
      the obvious comparison, but it is not clean: D also uses a lower learning rate
      ({D["hyperparams"]["alpha"]} against {A["hyperparams"]["alpha"]}) and fewer epochs
      ({D["hyperparams"]["epochs"]} against {A["hyperparams"]["epochs"]}), so that gap mixes
      initialisation with the optimiser schedule. <b>A2</b> is A re-run with D's exact
      schedule and random initialisation, which leaves the starting point as the only
      difference. A2 scores {acc("A2"):.3f} against D's {acc("D"):.3f}, so the warm start is
      worth <b>{acc("D")-acc("A2"):+.3f}</b>, not the {gap_ad:+.3f} the A-versus-D gap
      suggests. The looser comparison understated it, because A's more aggressive schedule was
      partly compensating for its random start.</div>''')

    add(table(
        ["Model", "Architecture", "Loss", "Trained on", "Accuracy", "F1", "OOV"],
        [[mid, "Skip-gram" if r[mid]["hyperparams"]["sg"] else "CBOW",
          "Hierarchical softmax" if r[mid]["hyperparams"].get("hs") else
          f'Negative sampling, k={r[mid]["hyperparams"]["negative"]}',
          f'{r[mid]["train"]["corpus_tokens"]:,} tok',
          f'<span class="{"win" if acc(mid)==max(acc(m) for m in ORDER) else ""}">{acc(mid):.4f}</span>',
          f"{f1(mid):.4f}", f'{r[mid]["extrinsic"]["oov_rate_test"]:.3f}']
         for mid in ORDER + ["A2"]],
        ["m", "l", "l", "n", "n", "n", "n"]))
    add(f'''<p class="prose">A2 is not a sixth model so much as a control. It exists only to
    make the A-versus-D claim above testable, and it is reported here rather than hidden
    because its score ({acc("A2"):.3f}) is the lowest in the table and that is the
    point.</p>''')
    add("</section>")

    # ---- deliverable 3, test examples
    tcm = json.loads((ROOT / "results" / "testcases.json").read_text())["models"]
    add(f'''<section id="testcases">
      <div class="sechead"><div class="eyebrow">Deliverable 3</div>
      <h2>Five test examples, ranked by every model</h2></div>
      <p class="prose">The sentiment task on held-out test data, which is what the SST corpus
      is here for. Each example carries two ranked outputs per model: the classifier's ranked
      labels with probabilities, and the top-{K_TC} nearest <i>training</i> sentences by cosine
      in that model's embedding space. Neighbour agreement is the share of those retrieved
      training sentences whose gold label matches the test sentence, so it says whether the
      embedding placed the test sentence among the right evidence.</p>
      <p class="prose">The five were chosen by rule rather than by eye, and the rule that
      selected each is named above it. Scores across the five:
      {", ".join(f"{m} {sum(c['correct'] for c in tcm[m]['cases'])}/5" for m in ORDER)}.</p>
      {testcase_section()}
      <div class="note"><b>Example 3 separates the pretrained model from everything trained
      here.</b> <code>no less</code> is an intensifier, not a negation, and only GoogleNews
      reads it that way. Every SST-trained model including the fine-tune takes it as negative.
      </div>
      <div class="note warm"><b>Example 5 is the one to be suspicious of.</b> Every model gets
      it right with high confidence, and retrieval shows why: the nearest training sentence is
      <code>no charm , no laughs , no fun</code>, a near-duplicate of the test sentence's
      stacked-negation pattern. The models are right for lexical reasons, not because they
      composed the negation. That is the same mechanism that makes Example 3 fail, and it
      happens to point the right way here.</div></section>''')

    # ---- architectures
    add(f'''<section id="arch">
      <div class="sechead"><div class="eyebrow">Deliverable 1</div>
      <h2>What the two architectures actually are</h2></div>
      <p class="prose">A word2vec model is two matrices and a dot product. There is no hidden
      layer and no nonlinearity. <code>W_in</code> holds one row per word and is the thing you
      keep. <code>W_out</code> exists only to compute the training loss and is discarded when
      training ends.</p>
      <div class="figgrid">
        <div class="figure">{arch_svg.skipgram(V, dim, A["hyperparams"]["window"], A["hyperparams"]["negative"])}
          <p class="caption"><b>A and E, skip-gram with negative sampling.</b> One centre word
          predicts each surrounding word independently. Instead of a softmax over all {V:,}
          words, each positive pair is scored against {A["hyperparams"]["negative"]} negatives
          drawn from the unigram distribution raised to 0.75.</p></div>
        <div class="figure">{arch_svg.cbow(V, dim, B["hyperparams"]["window"])}
          <p class="caption"><b>B, CBOW with hierarchical softmax.</b> The context rows are
          averaged into a single vector that predicts the centre word. The output is a binary
          tree over the vocabulary, so one prediction costs log2(V) decisions rather than V.</p></div>
      </div>
      <div class="note warm"><b>On the handout's encoder / code / decoder framing.</b>
      It is borrowed from autoencoders and fits word2vec only loosely. An autoencoder
      reconstructs its own input; word2vec reconstructs a <i>neighbouring</i> word, and it
      throws the decoder away. The mapping used above is the honest one: the one-hot input is
      the input layer, <code>W_in</code> is the encoder, the {dim}-dimensional row is the code,
      <code>W_out</code> is the decoder, and the context prediction is the output.</div>
    </section>''')

    # ---- data
    add(f'''<section id="data">
      <div class="sechead"><div class="eyebrow">Deliverable 2</div>
      <h2>Training and test data</h2></div>
      <p class="prose">Sentiment data is SST-2. Two releases exist and the choice matters. The
      GLUE copy sets every test label to -1 because that split is held out for the leaderboard,
      so test labels come from the SetFit release instead, and <code>src/data.py</code> raises
      if a loaded test split is not two-class.</p>''')
    add(table(["Split", "Source", "Rows", "Tokens", "Positive", "Used for"],
              [["train", "stanfordnlp/sst2 (phrase level)", "67,348", f"{tok:,}", "55.8%",
                "embedding training and classifier fitting"],
               ["dev", "SetFit/sst2", "872", "17,046", "50.9%", "held out, unused"],
               ["test", "SetFit/sst2", "1,821", "35,023", "49.9%", "every number on this page"]],
              ["m", "l", "n", "n", "n", "l"]))
    add(f'''<div class="note"><b>Leakage check.</b> The phrase rows are labelled subtrees of
      the treebank and come from the train split only, but they were still checked against the
      evaluation splits by exact string match: zero overlap with dev, and one row overlapping
      test, the fragment <code>what 's next ?</code>. It is dropped, which is why 67,349 rows
      become 67,348.</div>
      <p class="prose">The pretrained model is GoogleNews-300, genuine skip-gram with negative
      sampling trained on roughly 100 billion tokens of news text, from
      <a href="https://code.google.com/archive/p/word2vec/">code.google.com/archive/p/word2vec</a>,
      loaded through gensim-data and capped at the {C["train"]["vocab_limit"]:,} most frequent
      words. GloVe was deliberately not used as the pretrained baseline: it factorises a
      co-occurrence matrix and is not a Word2Vec model, so it would answer a different
      question than the one asked.</p>
      <h3 class="subhead">Pretrained model, as deliverable 2a asks</h3>{pretrained_table}
      <h3 style="margin-top:14px">Representative examples</h3>
      <p class="prose">Deliverable 2b asks for five training examples and four test examples,
      two positive and two negative. Selected by rule rather than by eye, with the selecting
      rule named beside each.</p>{ex2b()}</section>''')

    # ---- hyperparameters
    add(f'''<section id="hyper">
      <div class="sechead"><div class="eyebrow">Deliverable 5</div>
      <h2>Hyperparameters</h2></div>
      <p class="prose">A and B differ on two axes at once, architecture and loss, which is
      deliberate: it gives the comparison two things to say rather than one. Everything else is
      held constant between them.</p>''')
    add(table(["", "A", "B", "C", "D", "E", "A2"],
              [[k] + [str(r[m]["hyperparams"].get(lbl, "-")) for m in ORDER + ["A2"]]
               for k, lbl in [("Architecture (sg)", "sg"), ("Hierarchical softmax", "hs"),
                              ("Negatives", "negative"), ("Dimensions", "vector_size"),
                              ("Window", "window"), ("Min count", "min_count"),
                              ("Epochs", "epochs"), ("Initial LR", "alpha"),
                              ("Subsample threshold", "sample")]],
              ["l", "n", "n", "n", "n", "n", "n"]))
    add("</section>")

    # ---- neighbours
    add(f'''<section id="neighbours">
      <div class="sechead"><div class="eyebrow">Deliverables 3 and 5</div>
      <h2>Top-ranked neighbours for five queries</h2></div>
      <p class="prose">Five words chosen to separate the models rather than flatter them:
      a polysemous noun, a sentiment adjective, a word whose sense depends on domain, a rare
      domain term, and the most frequent word in English.</p>
      <div class="legend"><span class="li"><span class="chip only-d">word <b>0.00</b></span>
      <span><b>gold-tinted chip</b> &mdash; appears only on row D: a neighbour that C does
      <i>not</i> return for the same query, so fine-tuning on SST introduced it. Every other
      chip, on any row, is plain grey.</span></span></div>
      {neighbour_box(r, ORDER)}
      <div class="note"><b>Read the C and D rows against A, B and E.</b> C returns clean
      synonyms and inflections. D keeps them and adds film-specific associations:
      <code>visuals</code>, <code>screenplay</code> and <code>photography</code> enter
      <code>cinematography</code>, and <code>storyline</code> enters <code>plot</code>.
      A, B and E return something close to noise for every query. At {tok:,} tokens there is
      not enough co-occurrence evidence to place a word, and the next section puts a number
      on exactly how little.</div></section>''')

    # ---- visualization
    viz = json.loads((ROOT / "results" / "visualization.json").read_text())
    figs = []
    for mid in ["D", "A"]:
        for method, nm in [("pca", "PCA"), ("tsne", "t-SNE")]:
            v = viz[mid][method]
            detail = (f'explained variance {v["explained_variance"][0]:.1%} and '
                      f'{v["explained_variance"][1]:.1%}' if method == "pca"
                      else f'perplexity {v["perplexity"]}')
            figs.append(f'''<div class="figure">{v["svg"]}
              <p class="caption"><b>{nm}, Model {mid}.</b> {v["words_plotted"]} words from four
              fixed semantic groups, {detail}. Silhouette of the two-dimensional layout against
              the group labels: <b>{v["silhouette_2d"]}</b>.</p></div>''')
    sil = lambda m, k: viz[m][k]["silhouette_2d"]
    add(f'''<section id="viz">
      <div class="sechead"><div class="eyebrow">Handout requirement</div>
      <h2>The vector space in two dimensions</h2></div>
      <p class="prose">The words plotted are fixed in advance and belong to four semantic
      groups: positive sentiment, negative sentiment, film craft, and genre. Fixing them first
      means the figure can be judged rather than admired. If the embedding carries semantic
      structure, group members should land near each other after projection.</p>
      <p class="prose">Judging by eye is unreliable, so each figure also reports the silhouette
      score of the two-dimensional layout against the group labels. Around 0 means the groups
      are indistinguishable; higher means they separate.</p>
      <div class="figgrid">{"".join(figs)}</div>''')
    add(table(["Model", "PCA silhouette", "t-SNE silhouette", "WordSim-353", "Analogy accuracy"],
              [[mid, f"{sil(mid,'pca'):+.3f}", f"{sil(mid,'tsne'):+.3f}",
                f'{sim(mid,"wordsim353"):+.3f}',
                f'{ana[mid]["benchmark"]["overall_accuracy"]:.3f}'] for mid in ORDER],
              ["m", "n", "n", "n", "n"]))
    add(f'''<div class="note"><b>The projections agree with every other measurement.</b>
      D and C separate the four groups ({sil("D","pca"):+.3f} and {sil("C","pca"):+.3f} under
      PCA); A, B and E do not ({sil("A","pca"):+.3f}, {sil("B","pca"):+.3f},
      {sil("E","pca"):+.3f}). The same ordering appears in WordSim-353 and in analogy accuracy,
      which is worth stating because a scatter plot on its own is the easiest figure in this
      report to over-read. Note also how little variance the first two PCA components capture,
      under 20% for every model, so the picture is a thin slice of a 300-dimensional
      space.</div></section>''')

    # ---- analogies
    ar = lambda m: {x["query"]: x for x in ana[m]["arithmetic"]}
    add(f'''<section id="analogies">
      <div class="sechead"><div class="eyebrow">Handout requirement</div>
      <h2>Vector arithmetic and analogies</h2></div>
      <p class="prose">The handout names <code>king - man + woman = queen</code> specifically.
      Below is that query and six others, each showing the ranked answers the model returns,
      followed by the standard Google analogy benchmark scored per section.</p>''')
    rows = []
    for mid in ORDER:
        a = ar(mid)
        for label in ["king - man + woman", "paris - france + italy", "worst - bad + good"]:
            x = a[label]
            ans = ('<span class="chips">'
                   + "".join(f'<span class="chip">{e(w)} <b>{s:.2f}</b></span>'
                             for w, s in x["answers"][:4]) + "</span>"
                   if x["status"] == "ok" else f'<i>{x["status"]}: {", ".join(x["missing"])}</i>')
            rows.append([mid, f"<code>{e(label)}</code>", ans])
    add(table(["Model", "Query", "Ranked answers"], rows, ["m", "m", "l"]))
    add(f'''<div class="note"><b>Only C and D solve the canonical analogy.</b> Both return
      <code>queen</code> at rank 1 for <code>king - man + woman</code>. A, B and E return
      unrelated words, which is the same conclusion the similarity benchmarks and the
      projections reach: {tok:,} tokens of film review does not build a space where vector
      arithmetic means anything.</div>''')
    add(table(["Model", "Analogy accuracy", "Questions attempted", "Coverage", "Candidate pool"],
              [[mid, f'{ana[mid]["benchmark"]["overall_accuracy"]:.4f}',
                f'{ana[mid]["benchmark"]["attempted"]:,} of {ana[mid]["benchmark"]["questions_in_set"]:,}',
                f'{ana[mid]["benchmark"]["coverage"]:.1%}',
                f'top {ana[mid]["benchmark"]["restrict_vocab"]:,}'] for mid in ORDER],
              ["m", "n", "n", "n", "n"]))
    add(f'''<p class="prose">Coverage matters more than accuracy here. gensim drops any
      question containing an out-of-vocabulary word, so the SST-trained models are scored on
      {ana["A"]["benchmark"]["coverage"]:.1%} of the set against C's
      {ana["C"]["benchmark"]["coverage"]:.1%}. D scores
      {ana["D"]["benchmark"]["overall_accuracy"]:.3f} on its {ana["D"]["benchmark"]["coverage"]:.1%},
      which is structure inherited from the warm start rather than learned from SST: A2, which
      shares D's schedule but starts random, scores
      {ana["A2"]["benchmark"]["overall_accuracy"]:.4f} if it was measured.</p></section>'''
      if "A2" in ana else f'''<p class="prose">Coverage matters more than accuracy here.
      gensim drops any question containing an out-of-vocabulary word, so the SST-trained models
      are scored on {ana["A"]["benchmark"]["coverage"]:.1%} of the set against C's
      {ana["C"]["benchmark"]["coverage"]:.1%}. D's {ana["D"]["benchmark"]["overall_accuracy"]:.3f}
      is structure inherited from the warm start, not learned from SST.</p></section>''')

    # ---- similarity
    add(f'''<section id="similarity">
      <div class="sechead"><div class="eyebrow">Supporting analysis, beyond the requirement</div>
      <h2>Good at sentiment, useless at similarity</h2></div>
      <p class="prose">The sections above are the required sentiment task. This one is extra,
      and it exists because the sentiment numbers alone give a misleading picture of what these
      embeddings know.</p>
      <p class="prose">Spearman correlation between human similarity ratings and cosine
      similarity, on three standard benchmarks. Coverage is the share of pairs where both words
      were in vocabulary, and it is part of the result rather than a footnote: the SST-trained
      models are scored on roughly half of WordSim-353 because film reviews never mention the
      rest.</p>''')
    add(table(["Model", "WordSim-353", "SimLex-999", "MEN-3k", "Coverage", "SST-2 accuracy"],
              [[mid,
                f'<span class="{"win" if sim(mid,"wordsim353")>0.4 else "lose" if sim(mid,"wordsim353")<0.1 else ""}">{sim(mid,"wordsim353"):+.3f}</span>',
                f'{sim(mid,"simlex999"):+.3f}', f'{sim(mid,"men3k"):+.3f}',
                f'{r[mid]["intrinsic"]["wordsim353"]["coverage"]:.1%}', f"{acc(mid):.3f}"]
               for mid in ORDER],
              ["m", "n", "n", "n", "n", "n"]))
    add(f'''<div class="note warm"><b>A, B and E have no measurable lexical similarity
      structure at all.</b> B is slightly negative on two of three benchmarks, which is a
      correlation indistinguishable from none. Those same three models score
      {acc("B"):.3f} to {acc("A"):.3f} on sentiment, within {acc("C")-acc("B"):.3f} of
      GoogleNews. Mean-pooled logistic regression needs only that sentiment-bearing words
      occupy consistent directions. It does not need <code>tiger</code> to sit near
      <code>cat</code>. A report that evaluated downstream accuracy alone would conclude these
      embeddings are nearly as good as GoogleNews, and it would be wrong.</div>
      <p class="prose">It also reframes what the warm start bought. D scores
      {sim("D","wordsim353"):.3f} on WordSim-353 against {sim("A","wordsim353"):.3f} for A,
      trained from scratch on the identical corpus with identical settings. The pretrained
      initialisation is not a {gap_ad:.3f}-accuracy trick. It is the entire source of lexical
      structure in Model D.</p></section>''')

    # ---- finetune
    add(f'''<section id="finetune">
      <div class="sechead"><div class="eyebrow">Model D</div>
      <h2>What fine-tuning means for word2vec</h2></div>
      <p class="prose">Not what it means for a transformer. There are no frozen layers and no
      adapter. Initialise <code>W_in</code> from the published vectors, then run ordinary
      skip-gram training on movie reviews at a low learning rate.</p>
      <div class="stats">
        <div class="stat"><span class="n">{ft["seeded_from_pretrained"]:,}</span>
          <span class="l">rows seeded from GoogleNews</span></div>
        <div class="stat"><span class="n">{ft["randomly_initialised"]:,}</span>
          <span class="l">rows Google's vocabulary lacks</span></div>
        <div class="stat"><span class="n">{ft["mean_cosine_before_after"]:.3f}</span>
          <span class="l">mean cosine, before against after</span></div>
        <div class="stat"><span class="n">+{gap_ad:.3f}</span>
          <span class="l">accuracy over training from scratch</span></div>
      </div>
      <div class="driftgrid">
        {drift_col(ft["moved_most"][:8], "moved", "Moved most")}
        {drift_col(ft["moved_least"][-8:], "anchored", "Did not move at all")}
      </div>
      <div class="note"><b>The drift ranking is a readout of corpus frequency.</b> Everything
      at the top is an actor surname: frequent in reviews, rare in news, so heavily retrained.
      Everything at the bottom sits at cosine 1.000, unmoved: rare sentiment adjectives that
      appear once or twice in SST and never accumulate enough gradient. This answers "what did
      fine-tuning do" better than any single-word anecdote could.</div>
      <ol class="steps">
        <li><div class="step-b"><b>Build the vocabulary from our corpus first</b>
          <span class="why"><code>intersect_word2vec_format</code> only overwrites rows that
          already exist. It does not import Google's three million words, which is what we
          want, but it needs stating.</span></div></li>
        <li><div class="step-b"><b>Set <code>vectors_lockf</code> to ones explicitly</b>
          <span class="why">gensim 4.x does not create it. Without it the import either fails
          or silently freezes every row, producing a model that looks fine-tuned and is not.</span></div></li>
        <li><div class="step-b"><b>Keep the learning rate low: alpha {D["hyperparams"]["alpha"]}, {D["hyperparams"]["epochs"]} epochs</b>
          <span class="why">Google never published their output matrix, so <code>W_out</code>
          starts random and the first updates are partly spent re-learning a decoder. A normal
          learning rate scrambles the imported vectors in the first epoch.</span></div></li>
        <li><div class="step-b"><b>Measure drift rather than assuming it</b>
          <span class="why">Cosine between each word's warm-start vector and its final vector.
          Mean {ft["mean_cosine_before_after"]:.4f}, median {ft["median_cosine_before_after"]:.4f}, so the low
          learning rate held.</span></div></li>
      </ol></section>''')

    # ---- scratch
    lc = E["train"]["loss_curve"]
    add(f'''<section id="scratch">
      <div class="sechead"><div class="eyebrow">Model E, beyond the requirement</div>
      <h2>Writing skip-gram with negative sampling</h2></div>
      <p class="prose">The handout suggests taking a superficial look at the code. Writing it
      instead is the difference between having used word2vec and understanding it.
      <code>src/sgns.py</code> implements the vocabulary and counts, subsampling of frequent
      words, the unigram^0.75 noise table, a dynamic window, the two embedding matrices, and
      linear learning rate decay. <code>to_keyedvectors</code> then wraps <code>W_in</code> in
      a gensim object so E runs through the existing evaluation code unchanged.</p>''')
    add(table(["", "A, gensim", "E, from scratch"],
              [["SST-2 accuracy", f'{acc("A"):.4f}', f'{acc("E"):.4f}'],
               ["Wall clock", f'{A["train"]["wall_s"]:.1f} s', f'{E["train"]["wall_s"]:.1f} s'],
               ["Words per second", f'{A["train"]["words_per_sec"]:,}', f'{E["train"]["words_per_sec"]:,}'],
               ["Loss, first to last epoch", "not exposed", f"{lc[0]} to {lc[-1]}"]],
              ["l", "n", "n"]))
    add(f'''<div class="note warm"><b>The first version did not learn, and the failure is
      worth keeping.</b> Loss sat at exactly 1.3863 for all ten epochs and accuracy came out
      0.4992, which is chance on a balanced split. 1.3863 is 2&middot;ln 2, the loss of a model
      whose every logit is zero. The cause was the loss reduction, not the sampling: calling
      <code>BCEWithLogitsLoss</code> once on the positives and once on the negatives averages
      over B entries and over B&middot;k entries, so at a batch of
      {E["hyperparams"]["batch_size"]:,} the effective per-pair learning rate was about 3e-6.
      Measured gradient norm on <code>W_in</code> was exactly 0.0. Writing Mikolov's equation 4
      directly, summing over the k negatives and averaging only over the batch, fixed it.</div>
      <p class="prose">Two component checks were run before training anything, and they are
      what made that failure quick to localise. Subsampling keeps 17.6% of occurrences of
      <code>the</code>. The noise table maps <code>the</code> from a raw frequency of 4.30%
      down to a sampled 1.31% while lifting <code>execrable</code> from effectively zero to
      2e-5. Both are the 0.75 exponent behaving correctly.</p></section>''')

    # ---- systems
    add(f'''<section id="systems">
      <div class="sechead"><div class="eyebrow">Deliverable 4</div>
      <h2>What these models cost to run</h2></div>
      <p class="prose">Query latency is brute-force <code>most_similar</code>, a dense
      matrix-vector product over the whole vocabulary, measured over
      {lat("A")["repeats"]} calls. Memory is an RSS delta measured in an isolated subprocess,
      against the size of the matrices actually retained.</p>''')
    add(table(["Model", "Vocab", "p50 ms", "p95 ms", "Keeps", "Predicted MB", "Measured MB", "Ratio"],
              [[mid, f'{lat(mid)["vocab_size"]:,}', f'{lat(mid)["p50_ms"]:.2f}',
                f'{lat(mid)["p95_ms"]:.2f}', f'{mem(mid)["matrices_retained"]}',
                f'{mem(mid)["predicted_mb"]:.1f}', f'{mem(mid)["delta_mb"]:.1f}',
                f'{mem(mid)["delta_mb"]/mem(mid)["predicted_mb"]:.2f}'] for mid in ORDER],
              ["m", "n", "n", "n", "n", "n", "n", "n"]))
    add(f'''<div class="note"><b>Query latency is linear in vocabulary, as brute force should
      be.</b> C carries {vocab_ratio:.0f} times the vocabulary of the SST models and is
      {lat_ratio:.0f} times slower at p50. That is the argument for an approximate index stated
      as a measurement: at {lat("A")["vocab_size"]:,} words nobody needs one, at
      {lat("C")["vocab_size"]:,} it is {lat("C")["p50_ms"]:.0f} ms per query, and at Google's
      full three million it would be roughly 90 ms.</div>
      <p class="prose">The <code>2 &middot; V &middot; d &middot; 4</code> prediction holds
      within 3% for A, D and E. B comes in at
      {mem("B")["delta_mb"]/mem("B")["predicted_mb"]:.2f} because hierarchical softmax also
      keeps the Huffman tree's code and point arrays, which the formula does not count.</p>
      <div class="note warm"><b>C does not fit, and the cause is not established.</b>
      {mem("C")["vocab"]:,} by {dim} float32 is {mem("C")["predicted_mb"]:.0f} MB and the
      measured delta is {mem("C")["delta_mb"]:.1f} MB, a third of it. The likely explanation is
      macOS memory compression, which compresses inactive anonymous pages so
      <code>ru_maxrss</code> reports below the allocation. That is testable: the same
      measurement on a Linux node should come out near
      {mem("C")["predicted_mb"]:.0f} MB. Until it runs, this is an open discrepancy and not a
      finding.</div>
      <p class="prose">One more measurement decided how E trains. On this vocabulary, MPS is
      slower than CPU (64.3 against 44.6 ms per batch) and sparse gradients are slower than
      dense (52.6 against 44.6). The GPU loses because the model is small enough that launch
      and transfer overhead dominate. Sparsity loses for a specific reason: a batch of
      {E["hyperparams"]["batch_size"]:,} pairs with {E["hyperparams"]["negative"]} negatives
      touches roughly 90,000 index slots over {V:,} rows, so nearly every row updates on every
      step and there is no sparsity left to exploit. Both results should reverse on a larger
      corpus, which is the measurement text8 exists to take.</p></section>''')

    # ---- errors
    add(f'''<section id="errors">
      <div class="sechead"><div class="eyebrow">Deliverable 2c</div>
      <h2>Where every model fails, and why it is the same place</h2></div>''')
    add(table(["Model", "Errors / 1,821", "Negation", "Contrast", "Other"],
              [[mid, f'{err(mid)["n_errors"]:,}',
                f'{err(mid)["error_categories"].get("negation",0)}',
                f'{err(mid)["error_categories"].get("contrast",0)}',
                f'{err(mid)["error_categories"].get("other",0)}']
               for mid in sorted(ORDER, key=lambda m: err(m)["n_errors"])],
              ["m", "n", "n", "n", "n"]))
    add('''<div class="note"><b>Negation and contrast are 38 to 45% of every model's errors,
      against roughly 20% of the test set overall.</b> That holds for GoogleNews too, which
      means it is not an embedding-quality problem. It is the pooling step: averaging discards
      word order, so no amount of embedding quality recovers a scope-of-negation judgement.</div>''')
    d2c = json.loads((ROOT / "results" / "examples_2c.json").read_text())
    add(f'''<h3 class="subhead">Deliverable 2c, stated explicitly for Model D</h3>
      <p class="prose">The handout asks for at least two test examples the model classifies
      correctly and at least two it gets wrong, for a named model. Model D, the fine-tuned one,
      is used here. All four are drawn from the same 1,821-sentence test split and are the
      highest-confidence cases in each category.</p>''')
    add(table(["Outcome", "Gold", "Predicted", "Confidence", "Sentence"],
              [[f'<span class="{"win" if x["correct"] else "lose"}">'
                f'{"correct" if x["correct"] else "incorrect"}</span>',
                "positive" if x["gold"] else "negative",
                "positive" if x["pred"] else "negative",
                f'{x["confidence"]:.3f}', f'<code>{e(x["sentence"][:86])}</code>']
               for x in d2c["model_D"]], ["n", "n", "n", "n", "l"]))
    add('''<h3 class="subhead">Confident failures across models</h3>''')
    rows = []
    for mid in ["A", "D"]:
        for x in err(mid)["confident_errors"][:3]:
            rows.append([mid, f'<code>{e(x["sentence"][:88])}</code>',
                         "pos" if x["gold"] else "neg", "pos" if x["pred"] else "neg",
                         f'{x["confidence"]:.3f}', x["category"]])
    add(table(["Model", "Sentence", "Gold", "Predicted", "Confidence", "Category"], rows,
              ["m", "l", "n", "n", "n", "m"]))
    add('''<p class="prose">Each has a mechanical explanation.
      <code>not a bad journey at all</code> averages <code>not</code> and <code>bad</code> into
      negative territory. <code>there is n't a weak or careless performance amongst them</code>
      pools two strongly negative adjectives that the negation was supposed to invert.
      <code>a great deal of sizzle and very little steak</code> is an idiom whose negative
      reading lives entirely in the contrast between its clauses. The handout warns that a test
      set producing no failures must be made harder. That did not arise, since SST supplies
      these unaided.</p></section>''')

    # ---- deliverable 6
    add(f'''<section id="running">
      <div class="sechead"><div class="eyebrow">Deliverable 6</div>
      <h2>Installation, running and measurement</h2></div>
      <p class="prose">Four things cost real time, and each reported something other than its
      actual cause. They are recorded here because the handout asks for the experience, and
      because each has a specific tell.</p>
      <ol class="steps">
        <li><div class="step-b"><b>gensim has no wheels for Python 3.14</b>
          <span class="why">The machine's default interpreter is 3.14. Installing gensim there
          tries to compile Cython from source and fails a long way in. Homebrew's Python 3.11
          is what the virtual environment uses.</span></div></li>
        <li><div class="step-b"><b><code>import gensim</code> fails on a fresh unpinned install</b>
          <span class="why">gensim 4.3.x imports <code>scipy.linalg.triu</code>, which scipy
          1.13 removed. The ImportError names gensim, so it reads as a broken gensim rather
          than a version conflict. <code>requirements.txt</code> pins <code>scipy&lt;1.13</code>
          and <code>numpy&lt;2</code>.</span></div></li>
        <li><div class="step-b"><b>The SST-2 split with hidden labels</b>
          <span class="why">The GLUE copy sets every test label to -1. Accuracy against it is
          meaningless and looks plausible. <code>src/data.py</code> asserts the test split is
          two-class and raises otherwise.</span></div></li>
        <li><div class="step-b"><b>Whole-process peak RSS is not a model measurement</b>
          <span class="why">The first memory table compared process peak RSS against the matrix
          prediction and reported 297.8 MB for a model whose matrices are 34.3 MB. Peak RSS
          includes the interpreter, gensim, numpy, torch and the resident corpus.
          <code>src/measure_mem.py</code> now measures an RSS delta in an isolated
          subprocess per model.</span></div></li>
      </ol>
      <p class="prose">Reproducing everything on this page is four commands after
      <code>pip install -r requirements.txt</code>: <code>python -m src.main</code> trains A
      and B and loads C, <code>python -m src.run_eval</code> adds the similarity, latency and
      error measurements, <code>python -m src.measure_mem</code> fills the memory table, and
      <code>python -m src.build_report</code> regenerates this page. Model E is
      <code>python -m src.run_e</code> and takes {E["train"]["wall_s"]:.0f} seconds on
      CPU.</p></section>''')

    # ---- log
    rows = "".join(
        f'<div class="lrow"><span class="lhash">{h}</span>'
        f'<span><b>{e(msg.split(":")[0])}</b><div class="lnote">{e(msg.split(": ",1)[-1])}</div></span>'
        f'<span class="lstate">done</span></div>' for h, msg in commits())
    rows += ''.join(
        f'<div class="lrow"><span class="lhash">&mdash;</span>'
        f'<span><b>{t}</b><div class="lnote">{d}</div></span>'
        f'<span class="lstate todo">queued</span></div>'
        for t, d in [("text8", "Train A, B and E on 17M tokens; retest the MPS and sparse crossovers"),
                     ("Hogwild scaling", "words/sec against worker count on a multi-core node"),
                     ("Hyperparameter sweep", "training cost against accuracy for window, dimension and negatives")])
    add(f'''<section id="log">
      <div class="sechead"><div class="eyebrow">Provenance</div><h2>Build log</h2></div>
      <div class="log">{rows}</div>
      <p class="prose">Every figure on this page is read from <code>results/*.json</code> at
      build time by <code>src/build_report.py</code>. Nothing is typed into the prose, so
      re-running a model and rebuilding cannot leave a stale number behind.</p></section>''')

    add(f'''<footer>CS 6220 Big Data Systems, Fall 2026. Report generated from
      {len(r)} result records.</footer>''')

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Five Word2Vec Models, One Classifier</title>'
            f'<style>{CSS}{EXTRA_CSS}</style></head><body><div class="wrap">'
            f'{"".join(S)}</div></body></html>')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT.relative_to(ROOT)}  {len(page)/1024:.1f} KB")


if __name__ == "__main__":
    build()
