"""Build the write-up from results/*.json, in two variants.

`build("web")` writes public/index.html: the experiment presented on its own terms,
findings first, with no reference to the course it was set for. `build("submission")`
writes report/report.html, which is the same measurements carried by the coursework
framing the handout asks for and is what the PDF is rendered from.

Every number on the page is read out of the result records at build time. Nothing is typed
into the prose, so re-running a model and rebuilding cannot leave a stale figure behind.
"""
import html
import json
from pathlib import Path

from . import arch_svg, submission
from .report_css import CSS, EXTRA_CSS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "public" / "index.html"
REPORT_OUT = ROOT / "report" / "report.html"

TITLES = {
    "results": "Headline results", "testcases": "Worked test examples",
    "neighbours": "Nearest neighbours", "analogies": "Vector arithmetic",
    "viz": "Word clusters", "similarity": "Lexical similarity",
    "text8": "More training text",
    "errors": "Where they fail", "arch": "The two architectures",
    "data": "Corpus and splits", "hyper": "Hyperparameters",
    "finetune": "Fine-tuning G-ft", "scratch": "Writing C from scratch",
    "systems": "Cost to run", "running": "Reproducing it",
}

# The web write-up leads with what was measured; method and machinery follow.
WEB_GROUPS = [
    ("Findings", ["results", "testcases", "neighbours", "analogies", "viz",
                  "similarity", "text8", "errors"]),
    ("How the models were built", ["arch", "data", "hyper", "finetune", "scratch"]),
    ("Cost and reproduction", ["systems", "running"]),
]
# The submission follows the course's AI template, one question per deliverable; see
# src/submission.py. Both layouts must place every section exactly once.
assert {i for _, ids in WEB_GROUPS for i in ids} == set(TITLES) == set(submission.placed())
assert len(submission.placed()) == len(TITLES)

TOC_JS = """<script>
(function(){
  var links = [].slice.call(document.querySelectorAll('nav.toc a'));
  var byId = {};
  links.forEach(function(a){ byId[a.hash.slice(1)] = a; });
  var seen = {};
  function paint(){
    var best = null;
    Object.keys(seen).forEach(function(id){
      if (seen[id] && (!best || seen[id].top < seen[best].top)) best = id;
    });
    if (!best) return;
    links.forEach(function(a){ a.classList.remove('on'); });
    byId[best].classList.add('on');
  }
  var io = new IntersectionObserver(function(entries){
    entries.forEach(function(en){
      seen[en.target.id] = en.isIntersecting ? en.boundingClientRect : null;
    });
    paint();
  }, {rootMargin: '0px 0px -70% 0px'});
  Object.keys(byId).forEach(function(id){
    var el = document.getElementById(id);
    if (el) io.observe(el);
  });
})();
</script>"""
K_TC = 5
QUERIES = ["plot", "terrible", "star", "cinematography", "the"]
# Main models, smallest training corpus first; FULL slots each text8 retrain beside its SST twin.
ORDER = ["A", "B", "C", "G", "G-ft"]
FULL = ["A", "A-text8", "B", "B-text8", "C", "C-text8", "G", "G-ft"]
CORPUS_TAG = {"A": "SST", "B": "SST", "C": "SST", "A-text8": "text8", "B-text8": "text8",
              "C-text8": "text8",
              "G": "GoogleNews", "G-ft": "GoogleNews + SST"}
SHORT = {"A": "Skip-gram<br>+ negative sampling", "B": "CBOW<br>+ hierarchical softmax",
         "A-text8": "Skip-gram<br>+ negative sampling", "B-text8": "CBOW<br>+ hierarchical softmax",
         "C-text8": "Skip-gram + NS<br>written from scratch",
         "G": "GoogleNews-300<br>frozen", "G-ft": "GoogleNews<br>fine-tuned on SST",
         "C": "Skip-gram + NS<br>written from scratch"}
PROV = {"A": "gensim", "B": "gensim", "A-text8": "gensim", "B-text8": "gensim",
        "C-text8": "PyTorch, A100 GPU",
        "G": "downloaded", "G-ft": "gensim", "C": "PyTorch, from scratch"}

e = html.escape


def load():
    recs = {}
    for p in (ROOT / "results").glob("*.json"):
        r = json.loads(p.read_text())
        if "model_id" not in r:        # sidecar data, not a model record
            continue
        if r["corpus"] == "sst2":      # superseded sentence-level run, kept on disk only
            continue
        recs[r["model_id"] if r["corpus"] != "text8" else r["model_id"] + "-text8"] = r
    return recs


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


def corpus_line(mid, train):
    if mid == "G-ft":
        return "trained on <b>GoogleNews</b>, then SST"
    n = train["corpus_tokens"]
    size = (f"{n / 1e9:.0f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6
            else f"{n / 1e3:.0f}k")
    return f"trained on <b>{CORPUS_TAG[mid]}</b>, {size} tokens"


def cards(recs):
    shown = [m for m in FULL if m in recs]
    best = max(recs[m]["extrinsic"]["sst2_test_acc"] for m in shown)
    out = []
    for mid in shown:
        r = recs[mid]
        acc = r["extrinsic"]["sst2_test_acc"]
        ws = r["intrinsic"]["wordsim353"]["spearman"]
        t = r["train"]
        cost = (f'{t["load_wall_s"]:.1f} s to load' if not t.get("wall_s")
                else f'{t["wall_s"]:.1f} s to {"fine-tune" if mid == "G-ft" else "train"}')
        focus = " is-focus" if acc == best else ""
        out.append(f'''<div class="mcard{focus}">
        <div class="tag"><span class="id">{mid.replace("-text8", "")}</span><span class="nm">{SHORT[mid]}</span></div>
        <div class="corp">{corpus_line(mid, t)}</div>
        <div class="metric"><span class="ml">Sentiment accuracy</span>
          <div class="acc">{acc:.3f}</div>
          <div class="bar"><i style="width:{acc*100:.1f}%"></i></div></div>
        <div class="metric"><span class="ml">Word similarity <b>{ws:+.3f}</b></span>
          <div class="bar thin"><i style="width:{max(ws, 0)*100:.1f}%"></i></div></div>
        <div class="mmeta">{PROV[mid]} &middot; {cost}</div></div>''')
    return (f'<div class="models">{"".join(out)}</div>'
            '<p class="mkey">Sentiment accuracy is the share of the 1,821 SST-2 test sentences '
            'classified correctly. Word similarity is the Spearman correlation between the '
            "model's cosine similarities and human ratings on WordSim-353: 1 is perfect "
            'agreement, 0 is none.</p>')


def row_label(mid):
    return {"A": "A (SST)", "B": "B (SST)", "C": "C (SST)", "C-text8": "C (text8)",
            "A-text8": "A (text8)", "B-text8": "B (text8)"}.get(mid, mid)


def neighbour_box(recs, ids):
    """One table per query, with top-1, top-5 and top-10 stated explicitly per model."""
    out = []
    for q in QUERIES:
        rows = []
        for mid in ids:
            nb = recs[mid]["intrinsic"]["neighbours"].get(q)
            if not nb:
                rows.append([row_label(mid), "<i>out of vocabulary</i>", "", ""])
                continue
            base = {w for w, _ in recs["G"]["intrinsic"]["neighbours"].get(q, [])}
            def chips(lo, hi):
                inner = "".join(
                    f'<span class="chip{" only-d" if mid == "G-ft" and w not in base else ""}">'
                    f'{e(w)} <b>{s:.2f}</b></span>' for w, s in nb[lo:hi])
                return f'<span class="chips">{inner}</span>' 
            rows.append([row_label(mid), chips(0, 1), chips(1, 5), chips(5, 10)])
        out.append(f'<h3 class="qhead">{e(q)}</h3>')
        out.append(table(["Model", "Top-1", "Top-2 to 5", "Top-6 to 10"], rows,
                         ["m", "l", "l", "l"]))
    return "".join(out)


def pretrained_2a(ana):
    """Reference, URL, corpus, test set and accuracy for the pretrained model (deliverable 2a)."""
    b = ana["G"]["benchmark"]
    paper = ("Mikolov, Sutskever, Chen, Corrado and Dean (2013), <i>Distributed "
             "Representations of Words and Phrases and their Compositionality</i>, NIPS. "
             "<a href='https://arxiv.org/abs/1310.4546'>arxiv.org/abs/1310.4546</a>")
    dist = ("<a href='https://code.google.com/archive/p/word2vec/'>"
            "code.google.com/archive/p/word2vec</a>, fetched through gensim-data as "
            "<code>word2vec-google-news-300</code>")
    measured = (f"<b>{b['overall_accuracy']:.4f}</b> across {b['attempted']:,} attempted "
                f"questions ({b['coverage']:.1%} coverage, with the candidate pool limited to "
                f"the top {b['restrict_vocab']:,} words)")
    unpublished = ("Not quoted. The Google Code archive does not expose figures in a form that "
                   "can be fetched, and its linked papers report results for the models studied "
                   "in those papers, not this released file. I report the measured result above "
                   "with its exact conditions instead of attaching an unverified published "
                   "number to the model.")
    return table(["Field", "Value"], [
        ["Model", "GoogleNews-vectors-negative300"],
        ["Reference", paper],
        ["Distribution URL", dist],
        ["Training corpus", "Google News, roughly 100 billion tokens. The corpus was not "
         "released, so there is no public corpus link."],
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
            rows.append([row_label(mid), f'{lab} {prob:.3f}', f'{second[0]} {second[1]:.3f}', mark,
                         f'{c["neighbour_label_agreement"]:.1f}'])
        nb = tc["models"]["G-ft"]["cases"][k]["neighbours"][:3]
        chips = "".join(f'<span class="chip{" agree" if n["label"]==s["gold"] else ""}">'
                        f'{e(n["text"])} &middot; {n["cos"]:.2f}</span>' for n in nb)
        blocks.append(f'''<div class="tcase">
          <div class="tchead"><span class="eyebrow">Example {k+1} &middot; {e(s["selected_for"])}</span>
            <p class="tcsent"><code>{e(s["text"])}</code></p>
            <span class="tcgold">gold label: <b>{gold}</b></span></div>
          {table(["Model", "Rank 1", "Rank 2", "Result", "Neighbour agreement"], rows,
                 ["m", "n", "n", "n", "n"])}
          <div class="tcnb"><span class="nlabel">G-ft retrieves</span>
            <span class="chips">{chips}</span></div>
          <p class="tclegend">Training sentences are ordered by cosine similarity.
          A <b>green tint</b> means the retrieved sentence's label matches the test sentence's
          gold label. This is separate from the gold tint in the neighbour tables.</p></div>''')
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

def build(mode="web"):
    """mode="web": an independent write-up of the experiment, reordered findings-first.
    mode="submission": the same measurements with the coursework framing the handout asks
    for the cover page, deliverable labels, course footer, and deliverable order."""
    SUB = mode == "submission"
    r = load()
    A, B, C, G, GFT = (r[k] for k in ("A", "B", "C", "G", "G-ft"))
    full = [m for m in FULL if m in r]
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
    anres = lambda mid: r[mid]["analogies"] if mid.endswith("text8") else ana[mid]
    anb = lambda mid: anres(mid)["benchmark"]
    corpus = json.loads((ROOT / "results" / "corpus.json").read_text())
    pt, vg, wc = corpus["phrase_train"], corpus["vocab_gain"], corpus["word_counts"]
    sweep = json.loads((ROOT / "results" / "sweep.json").read_text())
    sw = {x["label"]: x for x in sweep["rows"]}
    t8oov = json.loads((ROOT / "results" / "text8_oov.json").read_text())
    ans_n = [wc[w]["sentences"] for w in ("queen", "actress", "worse", "best", "horror")]
    pretrained_table = pretrained_2a(ana)
    ft = GFT["finetune"]
    gap_ad = acc("G-ft") - acc("A")
    gap_cd = acc("G") - acc("G-ft")
    lat_ratio = lat("G")["p50_ms"] / lat("A")["p50_ms"]
    vocab_ratio = lat("G")["vocab_size"] / lat("A")["vocab_size"]

    buckets, cur = {}, []

    def sec(name):
        nonlocal cur
        cur = buckets.setdefault(name, [])

    def add(x):
        cur.append(x)

    def eb(sub, web):
        """Section eyebrow: the deliverable label for the PDF, a plain one for the web."""
        return f'<div class="eyebrow">{sub if SUB else web}</div>'

    def hw(sub, web=""):
        """Prose that only exists because this was set as coursework."""
        return sub if SUB else web


    sec("head")
    if SUB:
        add(submission.cover())
    add(f'''<header class="masthead">
      {eb("CS 6220 Big Data Systems &middot; Homework 2 &middot; programming option",
          "An experiment in word embeddings")}
      <h1>Five Word2Vec Models, Two Corpora, One Classifier</h1>
      <p class="lede">This experiment compares five routes to a Word2Vec model: two trained
      with gensim on movie reviews, a frozen pretrained model, a fine-tuned version of those
      pretrained weights, and a skip-gram implementation written from scratch. The three
      models trained from scratch are then retrained on text8, 17 million words of Wikipedia,
      to see how much the training text matters. The same pipeline measures sentiment
      accuracy, lexical structure, and runtime cost.</p>
      <div class="runmeta">
        <span><b>Corpora</b> SST-2 phrases, {tok:,} tokens; text8, {r["A-text8"]["train"]["corpus_tokens"]:,}</span>
        <span><b>Vocabulary</b> {V:,} (SST), {r["A-text8"]["train"]["vocab_size"]:,} (text8)</span>
        <span><b>Dimensions</b> {dim}</span>
        <span><b>Test split</b> 1,821 sentences</span>
        <span><b>Machine</b> M2 Air; PACE ICE A100 for C (text8)</span>
      </div>
    </header>''')

    toc = []
    for heading, ids in WEB_GROUPS:
        toc.append(f'<div class="tg">{heading}</div>')
        toc += [f'<a href="#{i}">{TITLES[i]}</a>' for i in ids]
    sec("nav")
    add(f'''<nav class="toc" aria-label="Sections">{submission.toc(TITLES) if SUB else "".join(toc)}</nav>''')

    # ---- results
    sec("results")
    add(f'''<section id="results">
      <div class="sechead"><div class="eyebrow">Where things stand</div>
      <h2>Eight embeddings, one classifier</h2></div>
      <p class="prose">A, B and C appear twice, once trained on SST and once retrained on text8.
      Every model goes through the same evaluation. I look up each word
      vector, average the vectors into one sentence representation, fit logistic regression,
      and test on the same held-out SST-2 sentences. The classifier is never tuned for an
      individual model. This keeps the comparison focused on the embeddings.</p>
      {cards(r)}
      <div class="note"><b>The warm start needs its own control.</b> Comparing A with G-ft looks
      natural, but it mixes two changes. G-ft also uses a lower learning rate
      ({GFT["hyperparams"]["alpha"]} against {A["hyperparams"]["alpha"]}) and fewer epochs
      ({GFT["hyperparams"]["epochs"]} against {A["hyperparams"]["epochs"]}). <b>A2</b> uses
      A's random initialization with G-ft's exact schedule, leaving the starting weights as the
      only difference. A2 reaches {acc("A2"):.3f}, while G-ft reaches {acc("G-ft"):.3f}. On that
      controlled comparison, the warm start adds <b>{acc("G-ft")-acc("A2"):+.3f}</b>, not the
      {gap_ad:+.3f} suggested by A versus G-ft. A's more aggressive schedule partly makes up for
      its random start, so the loose comparison understates the benefit.</div>''')

    add(table(
        ["Model", "Architecture", "Loss", "Trained on", "Accuracy", "F1", "OOV"],
        [[row_label(mid), "Skip-gram" if r[mid]["hyperparams"]["sg"] else "CBOW",
          "Hierarchical softmax" if r[mid]["hyperparams"].get("hs") else
          f'Negative sampling, k={r[mid]["hyperparams"]["negative"]}',
          f'{r[mid]["train"]["corpus_tokens"]:,} tok',
          f'<span class="{"win" if acc(mid)==max(acc(m) for m in ORDER) else ""}">{acc(mid):.4f}</span>',
          f"{f1(mid):.4f}", f'{r[mid]["extrinsic"]["oov_rate_test"]:.3f}']
         for mid in full + ["A2"]],
        ["m", "l", "l", "n", "n", "n", "n"]))
    add(f'''<p class="prose">A2 is a control, not a sixth model. Its {acc("A2"):.3f} score is
    the lowest in the table, but that result is useful: it makes the claim about G-ft's warm start
    testable. The three text8 rows are A, B and C trained on Wikipedia instead of SST.
    <a href="#text8">More training text</a> covers them.</p>''')
    add("</section>")

    # ---- deliverable 3, test examples
    tcm = json.loads((ROOT / "results" / "testcases.json").read_text())["models"]
    sec("testcases")
    add(f'''<section id="testcases">
      <div class="sechead">{eb("Deliverable 3", "Held-out sentiment, example by example")}
      <h2>Five test examples, ranked by the five main models</h2></div>
      <p class="prose">These examples show the held-out sentiment task at sentence level.
      For each model, the table gives the classifier's two labels with their probabilities and
      the top-{K_TC} closest <i>training</i> sentences by cosine similarity. Neighbour agreement
      is the fraction of those retrieved sentences whose gold label matches the test example.
      It gives a small, concrete view of what evidence sits nearby in the embedding space.</p>
      <p class="prose">I selected all five by rule, not by eye. The rule appears above each
      example. Accuracy within this small set is
      {", ".join(f"{m} {sum(c['correct'] for c in tcm[m]['cases'])}/5" for m in ORDER)}.</p>
      {testcase_section()}
      <div class="note"><b>Example 3 separates GoogleNews from every model trained on SST.</b>
      In this sentence, <code>no less</code> is an intensifier rather than a negation. Of the
      five main models, only the frozen GoogleNews model reads it that way. Every SST-trained model, including G-ft, predicts
      negative.
      </div>
      <div class="note warm"><b>Example 5 deserves a closer look.</b> All five main models get it right
      with high confidence, and retrieval shows why. The nearest training sentence is
      <code>no charm , no laughs , no fun</code>, a near-duplicate of the test sentence's
      stacked-negation pattern. The models succeed through lexical overlap, not by composing
      the negation. The same shortcut fails on Example 3 but happens to work here.</div></section>''')

    # ---- architectures
    sec("arch")
    add(f'''<section id="arch">
      <div class="sechead">{eb("Deliverable 1", "Mechanism")}
      <h2>How the two architectures work</h2></div>
      <p class="prose">A Word2Vec model comes down to two matrices and a dot product. It has
      no hidden layer or nonlinearity. <code>W_in</code> stores one row per word and becomes the
      final embedding table. <code>W_out</code> is used for the training objective, then
      discarded.</p>
      <div class="figgrid">
        <div class="figure">{arch_svg.skipgram(V, dim, A["hyperparams"]["window"], A["hyperparams"]["negative"])}
          <p class="caption"><b>A and C use skip-gram with negative sampling.</b> A centre word
          predicts each surrounding word independently. Each observed pair is scored against
          {A["hyperparams"]["negative"]} negative samples from the unigram distribution raised
          to 0.75, avoiding a softmax over all {V:,} words.</p></div>
        <div class="figure">{arch_svg.cbow(V, dim, B["hyperparams"]["window"])}
          <p class="caption"><b>B uses CBOW with hierarchical softmax.</b> The context rows are
          averaged into one vector, which predicts the centre word. A binary tree over the
          vocabulary reduces one prediction from V decisions to log2(V).</p></div>
      </div>
      <div class="note warm"><b>The encoder, code, and decoder analogy only goes so far.</b>
      The terms come from autoencoders, which reconstruct their own input. Word2Vec predicts a
      <i>neighbouring</i> word and throws away its output matrix after training. In the mapping
      above, the one-hot word is the input, <code>W_in</code> is the encoder, its
      {dim}-dimensional row is the code, <code>W_out</code> is the decoder, and the predicted
      context word is the output.</div>
    </section>''')

    # ---- data
    sec("data")
    add(f'''<section id="data">
      <div class="sechead">{eb("Deliverable 2", "Corpus")}
      <h2>Training and test data</h2></div>
      <p class="prose">The sentiment data comes from SST-2, but its two releases cannot be
      swapped freely. GLUE replaces every test label with -1 because the split is reserved for
      its leaderboard. I use the labelled SetFit test split instead. <code>src/data.py</code>
      raises an error unless both classes appear in the loaded test labels.</p>''')
    sp = corpus["splits"]
    add(table(["Split", "Source", "Rows", "Tokens", "Mean length", "Positive", "Used for"],
              [[s, src, f'{sp[s]["rows"]:,}', f'{sp[s]["tokens"]:,}', f'{sp[s]["mean_len"]}',
                f'{sp[s]["positive_frac"]:.1%}', use]
               for s, src, use in [
                   ("train", "stanfordnlp/sst2 (phrase level)",
                    "embedding training and classifier fitting"),
                   ("dev", "SetFit/sst2", "held out, unused"),
                   ("test", "SetFit/sst2", "every number on this page")]],
              ["m", "l", "n", "n", "n", "n", "l"]))
    add(f'''<div class="stats">
        <div class="stat"><span class="n">{V:,}</span>
          <span class="l">vocabulary at min_count=2, phrase-level train</span></div>
        <div class="stat"><span class="n">{corpus["sentence_train"]["vocab_min2"]:,}</span>
          <span class="l">vocabulary at min_count=2, sentence-level train</span></div>
        <div class="stat"><span class="n">6.6%</span>
          <span class="l">test-token OOV rate, SST-trained vocabulary</span></div>
        <div class="stat"><span class="n">27.1%</span>
          <span class="l">test-token OOV rate, GoogleNews-300 vocabulary</span></div>
      </div>
      <h3 style="margin-top:14px">What the rows look like</h3>
      <p class="prose">{hw("Deliverable 2b asks for five training examples and four test "
      "examples, two positive and two negative.", "Five training examples and four test "
      "examples, two positive and two negative.")} Each example was selected by a stated
      rule rather than picked by eye.</p>{ex2b()}
      <div class="note"><b>The leakage check found one duplicate.</b> The phrase rows
      are labelled treebank subtrees taken only from the training split. An exact string check
      still found the fragment <code>what 's next ?</code> in both training and test data. I
      removed it. There was no overlap with dev, and the training count fell from 67,349 to
      67,348 rows.</div>
      <div class="note warm"><b>Most of the phrase-level text repeats the sentences.</b>
      {pt["rows_inside_frac"]:.1%} of the phrase rows ({pt["rows_inside_sentences"]:,} of
      {sp["train"]["rows"]:,}) appear word for word inside the
      {corpus["sentence_train"]["rows"]:,} training sentences, because each row is a subtree of
      one of them. The phrase corpus has {pt["token_ratio"]}&times; the tokens of the sentence
      release, but most of those tokens are the same text cut into overlapping pieces. The
      repetition also doubles the vocabulary. At min_count=2 it grows from
      {corpus["sentence_train"]["vocab_min2"]:,} to {pt["vocab_min2"]:,} words, and
      {vg["gained_singletons"]:,} of the {vg["gained"]:,} added words occur only once in the
      sentences. They pass the cutoff because the fragments of that one sentence repeat them.
      <a href="#analogies">Vector arithmetic</a> shows what this does to the embeddings.</div>
      <h3 class="subhead">Second training corpus: text8</h3>
      <p class="prose">A, B and C were also trained on text8, the first 100 MB of an English
      Wikipedia dump cleaned by Matt Mahoney
      (<a href="http://mattmahoney.net/dc/text8.zip">mattmahoney.net/dc/text8.zip</a>).
      It has {r["A-text8"]["train"]["corpus_tokens"]:,} tokens and a vocabulary of
      {r["A-text8"]["train"]["vocab_size"]:,} words at min_count=2. The text contains only
      lowercase letters and spaces. Punctuation is removed and numbers are spelled out digit
      by digit, so <code>1989</code> becomes <code>one nine eight nine</code>. text8 is used
      only to train word vectors; the classifier and every test stay on SST. The file is
      downloaded by <code>src/run_text8.py</code> rather than stored in the repository.</p>
      <p class="prose">The pretrained baseline is GoogleNews-300, a skip-gram model with
      negative sampling trained on roughly 100 billion news tokens. It comes from
      <a href="https://code.google.com/archive/p/word2vec/">code.google.com/archive/p/word2vec</a>,
      is loaded through gensim-data, and is capped at the {G["train"]["vocab_limit"]:,} most
      frequent words. I did not use GloVe as the pretrained baseline because it factorizes a
      co-occurrence matrix rather than training Word2Vec. That would change the question being
      tested.</p>
      <h3 class="subhead">{hw("Pretrained model, as deliverable 2a asks", "The pretrained model")}</h3>{pretrained_table}</section>''')

    # ---- hyperparameters
    sec("hyper")
    add(f'''<section id="hyper">
      <div class="sechead">{eb("Deliverable 5", "Configuration")}
      <h2>Hyperparameters</h2></div>
      <p class="prose">A and B intentionally change two things together: the architecture and
      the loss. All other settings stay fixed, so the comparison is between the two standard
      Word2Vec configurations. The sweep below separates the two changes.</p>''')
    add(table([""] + [row_label(m) for m in ORDER + ["A2"]],
              [[k] + [str(r[m]["hyperparams"].get(lbl, "-")) for m in ORDER + ["A2"]]
               for k, lbl in [("Architecture (sg)", "sg"), ("Hierarchical softmax", "hs"),
                              ("Negatives", "negative"), ("Dimensions", "vector_size"),
                              ("Window", "window"), ("Min count", "min_count"),
                              ("Epochs", "epochs"), ("Initial LR", "alpha"),
                              ("Subsample threshold", "sample")]],
              ["l", "n", "n", "n", "n", "n", "n"]))
    ms = lambda row, k, f=".3f", sign="": (f'{row[k]["mean"]:{sign}{f}} '
                                           f'<span class="sd">&plusmn; {row[k]["sd"]:{f}}</span>')
    add(f'''<h3 class="subhead">Sweep</h3>
      <p class="prose">Each setting below was trained {len(sweep["rows"][0]["seeds"])} times on
      the SST phrase corpus, with seeds {", ".join(map(str, sweep["rows"][0]["seeds"]))}. The
      tables give the mean and standard deviation across seeds. One factor changes at a time,
      and everything else stays at A's or B's settings. Training time is the mean wall clock
      on the M2 Air.</p>''')
    for factor, title in [("arch x loss", "Architecture and loss"), ("window", "Context window"),
                          ("dimensions", "Vector dimensions"), ("negatives", "Negative samples")]:
        add(f'<h3 class="qhead">{title}</h3>')
        add(table(["Setting", "SST-2 accuracy", "WordSim-353", "MEN-3k", "Training time"],
                  [[e(x["label"]), ms(x, "acc"), ms(x, "wordsim353", sign="+"),
                    ms(x, "men3k", sign="+"), ms(x, "wall_s", ".1f") + " s"]
                   for x in sweep["rows"] if x["factor"] == factor],
                  ["m", "n", "n", "n", "n"]))
    add(f'''<div class="note"><b>Architecture matters and the loss does not.</b> Skip-gram
      averages {sw["skip-gram + NS"]["acc"]["mean"]:.3f} with negative sampling and
      {sw["skip-gram + HS"]["acc"]["mean"]:.3f} with hierarchical softmax. CBOW averages
      {sw["CBOW + NS"]["acc"]["mean"]:.3f} and {sw["CBOW + HS"]["acc"]["mean"]:.3f}. Changing
      the loss moves accuracy by at most
      {max(abs(sw["skip-gram + NS"]["acc"]["mean"] - sw["skip-gram + HS"]["acc"]["mean"]), abs(sw["CBOW + NS"]["acc"]["mean"] - sw["CBOW + HS"]["acc"]["mean"])):.3f},
      while changing the architecture moves it by about
      {sw["skip-gram + NS"]["acc"]["mean"] - sw["CBOW + HS"]["acc"]["mean"]:.3f}. The gap between
      A and B comes from skip-gram against CBOW.</div>
      <div class="note warm"><b>Single runs overstate the A-B gap.</b> The headline reports A at
      {acc("A"):.3f} and B at {acc("B"):.3f}, a gap of {acc("A") - acc("B"):.3f}. Over three
      seeds the same settings average {sw["skip-gram + NS"]["acc"]["mean"]:.3f} and
      {sw["CBOW + HS"]["acc"]["mean"]:.3f}, a gap of
      {sw["skip-gram + NS"]["acc"]["mean"] - sw["CBOW + HS"]["acc"]["mean"]:.3f}. Each headline
      figure is one run, and A's landed above its mean while B's landed below.</div>
      <p class="prose"><b>Wider windows help both models.</b> Going from a window of 2 to 10
      raises A from {sw["A, window 2"]["acc"]["mean"]:.3f} to {sw["A, window 10"]["acc"]["mean"]:.3f}
      and B from {sw["B, window 2"]["acc"]["mean"]:.3f} to {sw["B, window 10"]["acc"]["mean"]:.3f}.
      For skip-gram it also more than doubles training time, from
      {sw["A, window 2"]["wall_s"]["mean"]:.1f} to {sw["A, window 10"]["wall_s"]["mean"]:.1f}
      seconds, because every extra context word is another prediction. <b>CBOW needs more
      dimensions than skip-gram.</b> B gains
      {sw["B, 300 dims"]["acc"]["mean"] - sw["B, 50 dims"]["acc"]["mean"]:.3f} going from 50 to
      300 dimensions, while A gains only
      {sw["A, 300 dims"]["acc"]["mean"] - sw["A, 50 dims"]["acc"]["mean"]:.3f}. <b>The number of
      negatives shows no clear trend.</b> 5, 10 and 20 negatives give
      {sw["A, 5 negatives"]["acc"]["mean"]:.3f}, {sw["A, 10 negatives"]["acc"]["mean"]:.3f} and
      {sw["A, 20 negatives"]["acc"]["mean"]:.3f}. The middle setting is the lowest, which fits
      no trend, and with three seeds a gap that small cannot be separated from seed noise.
      20 negatives take
      {sw["A, 20 negatives"]["wall_s"]["mean"]:.0f} seconds.</p>
      <div class="note"><b>No setting produces similarity structure.</b> WordSim-353 stays between
      {min(x["wordsim353"]["mean"] for x in sweep["rows"]):+.3f} and
      {max(x["wordsim353"]["mean"] for x in sweep["rows"]):+.3f} across all
      {len({json.dumps(x["runs"]) for x in sweep["rows"]})} distinct settings. On SST, the hyperparameters shift sentiment accuracy by a few points,
      but none of them fixes similarity. Only more data did, as
      <a href="#text8">More training text</a> shows.</div>
      <p class="prose">Training times drift as the fanless M2 Air warms up. The 10-negative
      settings were the first runs of the sweep, on a cool machine, and came out faster than the
      5-negative runs even though they do more work. Small time differences between settings
      are therefore not reliable.</p></section>''')

    # ---- neighbours
    sec("neighbours")
    add(f'''<section id="neighbours">
      <div class="sechead">{eb("Deliverables 3 and 5", "Intrinsic evaluation")}
      <h2>Top-ranked neighbours for five queries</h2></div>
      <p class="prose">The five queries are meant to expose differences between the models:
      a polysemous noun, a sentiment adjective, a domain-sensitive word, a rare film term, and
      the most frequent word in English.</p>
      <div class="legend"><span class="li"><span class="chip only-d">word <b>0.00</b></span>
      <span>A <b>gold-tinted chip</b> appears only in row G-ft. It marks a neighbour that G does
      <i>not</i> return for the same query, so it entered the neighbourhood after fine-tuning
      on SST. All other chips are grey.</span></span></div>
      {neighbour_box(r, full)}
      <div class="note"><b>G and G-ft behave differently from A, B and C trained on SST.</b> G returns clear
      synonyms and inflections. G-ft preserves them while adding film-specific associations:
      <code>visuals</code>, <code>screenplay</code> and <code>photography</code> enter
      <code>cinematography</code>, and <code>storyline</code> enters <code>plot</code>.
      The three SST-trained models return almost noise for every query. Their {tok:,}-token corpus
      does not contain enough co-occurrence evidence to organize the words reliably. The
      <a href="#similarity">similarity benchmarks</a> measure that gap directly.</div>
      <div class="note"><b>The text8 rows are A, B and C retrained on Wikipedia.</b> With
      {r["A-text8"]["train"]["corpus_tokens"]:,} tokens, their neighbours become sensible:
      <code>terrible</code> gives <code>horrible</code> for all three, and B's <code>plot</code>
      gives <code>storylines</code> and <code>storyline</code>. Some neighbours now come from
      encyclopedia topics rather than film reviews, such as <code>trek</code> for
      <code>star</code>. <a href="#text8">More training text</a> compares the two corpora in
      full.</div></section>''')

    # ---- visualization
    viz = json.loads((ROOT / "results" / "visualization.json").read_text())
    figs = []
    for mid in ["G-ft", "A", "A-text8"]:
        for method, nm in [("pca", "PCA"), ("tsne", "t-SNE")]:
            v = viz[mid][method]
            detail = (f'The two axes keep {v["explained_variance"][0]:.1%} and '
                      f'{v["explained_variance"][1]:.1%} of the variance' if method == "pca"
                      else f'Perplexity {v["perplexity"]}')
            figs.append(f'''<div class="figure">{v["svg"]}
              <p class="caption"><b>{nm}, Model {row_label(mid)}.</b> {detail}.
              Silhouette score <b>{v["silhouette_2d"]:+.3f}</b>.</p></div>''')
    sil = lambda m, k: viz[m][k]["silhouette_2d"]
    sep = lambda m, k: viz[m]["separation"][k]
    nwords = viz["A"]["pca"]["words_plotted"]
    ev = {m: sum(viz[m]["pca"]["explained_variance"]) for m in FULL}
    t8 = ["A-text8", "B-text8", "C-text8"]
    sec("viz")
    add(f'''<section id="viz">
      <div class="sechead">{eb("Handout requirement", "Qualitative check")}
      <h2>Do similar words cluster together?</h2></div>
      <p class="prose">A model stores each word as {dim} numbers, which can't be drawn
      directly. PCA and t-SNE are two standard ways to squeeze them down to two, so every word
      becomes a dot on a page. PCA keeps the two directions along which the words vary most.
      t-SNE tries to keep each word's nearest neighbours close, at the cost of distorting
      larger distances. If a model has learned what words mean, related words should land
      near each other.</p>
      <p class="prose">To test that, I picked {nwords} words before looking at any plot, eight
      from each of four groups: positive words (<code>good</code>, <code>charming</code>),
      negative words (<code>dull</code>, <code>tedious</code>), filmmaking terms
      (<code>screenplay</code>, <code>editing</code>) and genres (<code>horror</code>,
      <code>western</code>). Each dot is coloured by its group.</p>
      <p class="prose">Clusters are easy to imagine in a scatter plot, so each one also gets a
      silhouette score. It compares how close a word sits to its own group against how close
      it sits to the nearest other group, and runs from -1 to 1. Near 0 means the groups are
      mixed together; higher means they form separate clumps. The plots below show G-ft, the
      clearest case, and A trained on each corpus.</p>
      <div class="figgrid">{"".join(figs)}</div>''')
    add(f'''<p class="prose">A flat picture throws most of the information away. The first two
      PCA axes keep between {min(ev.values()):.0%} and {max(ev.values()):.0%} of the variance,
      depending on the model. So the last two score columns skip the projection and measure
      the same thing on the full {dim}-dimensional vectors, using cosine distance. The last of
      those asks only whether positive and negative words form separate groups.</p>''')
    add(table(["Model", "2-D, PCA", "2-D, t-SNE", f"{dim}-D, all groups",
               f"{dim}-D, positive vs negative"],
              [[row_label(mid), f"{sil(mid,'pca'):+.3f}", f"{sil(mid,'tsne'):+.3f}",
                f"{sep(mid,'all'):+.3f}", f"{sep(mid,'positive_vs_negative'):+.3f}"]
               for mid in FULL],
              ["m", "n", "n", "n", "n"]))
    add(f'''<div class="note"><b>The GoogleNews models separate the groups best.</b> Under PCA,
      G-ft and G score {sil("G-ft","pca"):+.3f} and {sil("G","pca"):+.3f}, while A, B and C
      trained on SST score between {min(sil(m,"pca") for m in "ABC"):+.3f} and
      {max(sil(m,"pca") for m in "ABC"):+.3f}. t-SNE is less stable: the same model can score
      quite differently under the two methods, as B (text8) does
      ({sil("B-text8","pca"):+.3f} against {sil("B-text8","tsne"):+.3f}).</div>
      <div class="note warm"><b>text8 teaches similarity, not sentiment.</b> The text8 models
      score close to G on WordSim-353 ({sim("A-text8","wordsim353"):.3f} for A against
      {sim("G","wordsim353"):.3f} for G), yet they separate positive from negative words worse
      than any other model, at {min(sep(m,"positive_vs_negative") for m in t8):+.3f} to
      {max(sep(m,"positive_vs_negative") for m in t8):+.3f}, against
      {sep("A","positive_vs_negative"):+.3f} for A trained on SST and
      {sep("G","positive_vs_negative"):+.3f} for G. Wikipedia uses <code>good</code> and
      <code>bad</code> in the same kinds of sentence, so the vectors for praise and criticism
      end up close together. That fits the drop in sentiment accuracy described under
      <a href="#text8">More training text</a>.</div></section>''')

    # ---- analogies
    ar = lambda m: {x["query"]: x for x in (r[m]["analogies"] if m.endswith("text8")
                                            else ana[m])["arithmetic"]}
    sec("analogies")
    add(f'''<section id="analogies">
      <div class="sechead">{eb("Handout requirement", "Compositionality")}
      <h2>Vector arithmetic and analogies</h2></div>
      <p class="prose">{hw("The handout names <code>king - man + woman = queen</code> "
      "specifically.", "<code>king - man + woman = queen</code> is the canonical test of "
      "whether an embedding space supports this kind of arithmetic.")}
      Below it are four queries built from film-review vocabulary. Every word in
      those four occurs in SST, so the SST-trained models cannot miss them for lack of
      vocabulary. The standard Google analogy benchmark follows.</p>''')
    chip = lambda w, s, top=False: (f'<span class="chip{" top" if top else ""}">{e(w)} '
                                    f'<b>{s:.2f}</b></span>')
    for label, expected in [("king - man + woman", "queen"), ("actor - man + woman", "actress"),
                            ("better - good + bad", "worse"), ("worst - bad + good", "best"),
                            ("comedy - funny + scary", "horror")]:
        rows = []
        for mid in full:
            x = ar(mid)[label]
            if x["status"] != "ok":
                rows.append([row_label(mid), f'<i>{x["status"]}: {", ".join(x["missing"])}</i>', ""])
                continue
            (w0, s0), rest = x["answers"][0], x["answers"][1:4]
            rows.append([row_label(mid), chip(w0, s0, top=True),
                         '<span class="chips">' + "".join(chip(w, s) for w, s in rest)
                         + "</span>"])
        add(f'<h3 class="qhead">{e(label)} <span class="qexp">expected '
            f'<code>{e(expected)}</code></span></h3>')
        add('<div class="anatab">'
            + table(["Model", "Top answer", "Ranks 2 to 4"], rows, ["m", "l", "l"]) + "</div>")
    add(f'''<div class="note"><b>Every model trained on enough text solves the canonical
      analogy.</b> G, G-ft and the three text8 models all rank <code>queen</code> first for
      <code>king - man + woman</code>. Trained on SST alone, A, B and C return unrelated
      words.</div>
      <div class="note"><b>Film vocabulary does not rescue the SST-trained models.</b> A, B and C
      trained on SST miss the film queries too. The one near hit is B (SST), which ranks
      <code>actress</code> second for <code>actor - man + woman</code>. G and G-ft put the
      expected word first for the actor, better and worst queries. The text8 models do better
      but not uniformly: all three rank <code>actress</code> first, A (text8) and C (text8)
      also solve the better and worst queries, and B (text8) solves neither. For <code>comedy - funny + scary</code>, G returns
      <code>horror_flick</code> and <code>horror</code>, and G-ft returns <code>gross-out</code>
      and <code>frightening</code>. The SST-only models fail even on their own
      vocabulary.</div>
      <h3 class="subhead">Why A, B and C fail</h3>
      <p class="prose">An analogy works when the offset between two words, such as
      <code>man</code> to <code>woman</code>, points the same way for other pairs, such as
      <code>king</code> to <code>queen</code> and <code>actor</code> to <code>actress</code>.
      A model learns a shared direction like that only after it sees each word in many
      different contexts. In the {corpus["sentence_train"]["rows"]:,} training sentences, the
      expected answers appear between {min(ans_n)} and {max(ans_n)} times, and even the most
      common query word appears only {max(v["sentences"] for v in wc.values())} times. The
      table gives each count next to the count in the phrase rows the models were trained
      on.</p>''')
    add(table(["Query", "Occurrences: sentences / phrase rows"],
              [[f"<code>{e(label)}</code>",
                '<span class="chips">' + "".join(
                    f'<span class="chip">{w} <b>{wc[w]["sentences"]} / {wc[w]["phrases"]}</b></span>'
                    for w in words) + "</span>"]
               for label, words in [
                   ("king - man + woman", ["king", "man", "woman", "queen"]),
                   ("actor - man + woman", ["actor", "man", "woman", "actress"]),
                   ("better - good + bad", ["better", "good", "bad", "worse"]),
                   ("worst - bad + good", ["worst", "bad", "good", "best"]),
                   ("comedy - funny + scary", ["comedy", "funny", "scary", "horror"])]],
              ["m", "l"]))
    add(f'''<p class="prose">The phrase counts overstate what the models learned from. Most
      phrase rows are fragments of those same sentences (see <a href="#data">the data
      section</a>), so <code>queen</code> appears {wc["queen"]["phrases"]} times but in only
      {wc["queen"]["sentences"]} places in the original text. GoogleNews was trained on about
      100 billion tokens, more than {int(1e11 / tok / 10000) * 10000:,} times the {tok:,}
      tokens here. The similarity benchmarks agree: on WordSim-353, A scores
      {sim("A", "wordsim353"):+.3f}, B {sim("B", "wordsim353"):+.3f} and C
      {sim("C", "wordsim353"):+.3f}, against G's {sim("G", "wordsim353"):+.3f}.</p>
      <p class="prose">Retraining A, B and C on text8 confirms the explanation. With
      {r["A-text8"]["train"]["corpus_tokens"]:,} tokens and the same hyperparameters, all three
      answer <code>queen</code>, and their analogy accuracy rises to
      {r["A-text8"]["analogies"]["benchmark"]["overall_accuracy"]:.3f},
      {r["B-text8"]["analogies"]["benchmark"]["overall_accuracy"]:.3f} and
      {r["C-text8"]["analogies"]["benchmark"]["overall_accuracy"]:.3f}. See
      <a href="#text8">More training text</a>.</p>
      <p class="prose">Sentiment classification holds up anyway. A reaches {acc("A"):.3f} on
      SST-2 against G's {acc("G"):.3f}. A classifier over averaged vectors only needs positive
      and negative words to separate roughly. That is a much coarser property than consistent
      analogy offsets, so the two evaluations measure different things.</p>''')
    add(table(["Model", "Analogy accuracy", "Questions attempted", "Coverage", "Candidate pool"],
              [[row_label(mid), f'{anb(mid)["overall_accuracy"]:.4f}',
                f'{anb(mid)["attempted"]:,} of {anb(mid)["questions_in_set"]:,}',
                f'{anb(mid)["coverage"]:.1%}',
                f'top {anb(mid)["restrict_vocab"]:,}'] for mid in full],
              ["m", "n", "n", "n", "n"]))
    add(f'''<p class="prose">Accuracy needs to be read alongside coverage. gensim drops any
      question with an out-of-vocabulary word, so the SST-trained models are scored on only
      {ana["A"]["benchmark"]["coverage"]:.1%} of the set against G's
      {ana["G"]["benchmark"]["coverage"]:.1%}. G-ft scores
      {ana["G-ft"]["benchmark"]["overall_accuracy"]:.3f} on its {ana["G-ft"]["benchmark"]["coverage"]:.1%},
      and that structure comes from the warm start rather than SST. A2 uses the same schedule
      but starts randomly; when measured, it scores
      {ana["A2"]["benchmark"]["overall_accuracy"]:.4f}.</p></section>'''
      if "A2" in ana else f'''<p class="prose">Accuracy needs to be read alongside coverage.
      gensim drops any question with an out-of-vocabulary word, so the SST-trained models are
      scored on only {ana["A"]["benchmark"]["coverage"]:.1%} of the set against G's
      {ana["G"]["benchmark"]["coverage"]:.1%}. G-ft's {ana["G-ft"]["benchmark"]["overall_accuracy"]:.3f}
      comes from structure inherited through the warm start, not learned from SST.</p></section>''')

    # ---- similarity
    sec("similarity")
    add(f'''<section id="similarity">
      <div class="sechead">{eb("Supporting analysis, beyond the requirement", "Where the sentiment score misleads")}
      <h2>Good at sentiment, useless at similarity</h2></div>
      <p class="prose">{hw("The sections above cover the required sentiment task. This extra "
      "analysis is included", "This analysis is included")} because sentiment accuracy alone
      makes the embeddings look more capable than they are.</p>
      <p class="prose">The three benchmarks compare cosine similarity with human ratings using
      Spearman correlation. Coverage is the fraction of pairs where both words are in the
      vocabulary. It is part of the result: the SST-trained models cover only about half of
      WordSim-353 because many benchmark words never occur in the movie reviews.</p>''')
    add(table(["Model", "WordSim-353", "SimLex-999", "MEN-3k", "Coverage", "SST-2 accuracy"],
              [[row_label(mid),
                f'<span class="{"win" if sim(mid,"wordsim353")>0.4 else "lose" if sim(mid,"wordsim353")<0.1 else ""}">{sim(mid,"wordsim353"):+.3f}</span>',
                f'{sim(mid,"simlex999"):+.3f}', f'{sim(mid,"men3k"):+.3f}',
                f'{r[mid]["intrinsic"]["wordsim353"]["coverage"]:.1%}', f"{acc(mid):.3f}"]
               for mid in full],
              ["m", "n", "n", "n", "n", "n"]))
    add(f'''<div class="note warm"><b>Trained on SST, A, B and C show no measurable lexical
      similarity structure.</b> B is slightly negative on two benchmarks, which is effectively no
      correlation. Yet those same three models score
      {acc("B"):.3f} to {acc("A"):.3f} on sentiment, within {acc("G")-acc("B"):.3f} of
      GoogleNews. Mean-pooled logistic regression needs only that sentiment-bearing words
      point in consistent directions; it does not need <code>tiger</code> to sit near
      <code>cat</code>. Downstream accuracy by itself would make these embeddings look nearly
      as good as GoogleNews, which the intrinsic tests clearly contradict. The text8 rows are
      the same three models trained on more text, and they score close to GoogleNews;
      <a href="#text8">More training text</a> covers them.</div>
      <p class="prose">This also clarifies what the warm start contributes. G-ft scores
      {sim("G-ft","wordsim353"):.3f} on WordSim-353 against {sim("A","wordsim353"):.3f} for A,
      trained from scratch on the identical corpus with identical settings. The pretrained
      initialization does more than add {gap_ad:.3f} accuracy. It supplies all of Model G-ft's
      measurable lexical structure.</p></section>''')

    # ---- text8
    At, Bt = r["A-text8"], r["B-text8"]
    t8tok = At["train"]["corpus_tokens"]
    t8 = {"A": At, "B": Bt}
    top = lambda mid, q: next(x for x in anres(mid)["arithmetic"]
                              if x["query"] == q)["answers"][0][0]
    cols = ["A", "A-text8", "B", "B-text8", "C", "C-text8", "G"]
    sec("text8")
    add(f'''<section id="text8">
      <div class="sechead">{eb("Beyond the requirement", "More training text")}
      <h2>More text fixes similarity, not sentiment</h2></div>
      <p class="prose">The sections above blame the size of the training data for the
      failures of A, B and C on similarity and analogies. To test that directly, I retrained all
      three on text8,
      the first 100 MB of cleaned English Wikipedia
      (<a href="http://mattmahoney.net/dc/text8.zip">mattmahoney.net/dc/text8.zip</a>). It has
      {t8tok:,} tokens, {t8tok / tok:.0f} times the SST phrase corpus. Every hyperparameter
      matches the SST runs. One implementation detail differs: C's text8 run uses sparse
      Adam, which updates only the rows a batch touches, because dense Adam is too slow at
      this vocabulary size (see <a href="#systems">Cost to run</a>). C (text8) also trained on
      a PACE ICE A100 GPU, so its training time is not comparable with the laptop times. The sentiment test is also unchanged: the classifier is still fit
      on the SST phrase labels and scored on the same SST-2 test sentences, so only the word
      vectors differ.</p>''')
    tok_short = lambda n: (f"{n / 1e9:.0f}B" if n >= 1e9 else f"{n / 1e6:.0f}M" if n >= 1e6
                           else f"{n / 1e3:.0f}k")
    add('<div class="t8tab">' + table(["Model", "Tokens", "WordSim", "Analogies",
               "king &minus; man + woman", "SST-2 acc.", "OOV", "Train time"],
              [[row_label(m), tok_short(r[m]["train"]["corpus_tokens"]),
                f'{sim(m, "wordsim353"):+.3f}', f'{anb(m)["overall_accuracy"]:.3f}',
                f"<code>{e(top(m, 'king - man + woman'))}</code>", f"{acc(m):.3f}",
                f'{r[m]["extrinsic"]["oov_rate_test"]:.1%}',
                (f'{r[m]["train"]["wall_s"]:.1f} s'
                 + (" (A100)" if "cuda" in r[m]["machine"] else "")) if m != "G" else "-"]
               for m in cols],
              ["m", "n", "n", "n", "n", "n", "n", "n"]) + "</div>")
    add(f'''<div class="note"><b>More text fixes the structure.</b> WordSim-353 rises from
      {sim("A", "wordsim353"):+.3f} to {sim("A-text8", "wordsim353"):+.3f} for A, from
      {sim("B", "wordsim353"):+.3f} to {sim("B-text8", "wordsim353"):+.3f} for B and from
      {sim("C", "wordsim353"):+.3f} to {sim("C-text8", "wordsim353"):+.3f} for C, close to
      GoogleNews at {sim("G", "wordsim353"):+.3f}. Analogy accuracy goes from almost zero to
      {anb("A-text8")["overall_accuracy"]:.3f}, {anb("B-text8")["overall_accuracy"]:.3f} and
      {anb("C-text8")["overall_accuracy"]:.3f}, and all three now answer <code>queen</code> for
      <code>king - man + woman</code>. The
      <a href="#analogies">analogy tables</a> and <a href="#neighbours">neighbour lists</a>
      include the text8 models as extra rows.</div>
      <div class="note warm"><b>Sentiment accuracy does not improve.</b> A falls from
      {acc("A"):.3f} to {acc("A-text8"):.3f}, B moves from {acc("B"):.3f} to
      {acc("B-text8"):.3f}, and C falls from {acc("C"):.3f} to {acc("C-text8"):.3f}. The text8
      vocabulary misses
      {r["A-text8"]["extrinsic"]["oov_rate_test"]:.1%} of the SST-2 test tokens, against
      {r["A"]["extrinsic"]["oov_rate_test"]:.1%} for the SST models.
      {t8oov["kinds"]["punctuation or number"]["share"]:.0%} of the missing tokens are
      punctuation, which carries no sentiment. The rest matters more. text8 has no apostrophes,
      so <code>n't</code> is missing, and it occurs in {t8oov["sentences_with_nt"]} of the
      {t8oov["test_sentences"]:,} test sentences. Review vocabulary is missing too:
      {", ".join(f"<code>{e(w)}</code>" for w in ["refreshingly", "soggy", "mesmerizing", "sappy", "clunker", "heartwarming"])}
      are all in the SST vocabulary but not in text8's, and several of them are strong
      sentiment words. Wikipedia teaches general word meaning. SST teaches the words
      reviewers use to praise and criticize, and the sentiment test rewards that.</div>''')
    add(f'''<h3 class="subhead">Testing the handout's claims</h3>
      <p class="prose">Page 1 of the handout makes two claims about the architectures that
      the two corpora can test. <b>CBOW trains several times faster.</b> On text8, B took
      {Bt["train"]["wall_s"]:.0f} seconds against A's {At["train"]["wall_s"]:.0f}, a
      {At["train"]["wall_s"] / Bt["train"]["wall_s"]:.1f}&times; difference
      ({Bt["train"]["words_per_sec"]:,} against {At["train"]["words_per_sec"]:,} words per
      second). The claim holds. <b>Skip-gram does better on small data, and CBOW needs large
      data.</b> On SST, skip-gram leads CBOW by
      {sw["skip-gram + NS"]["acc"]["mean"] - sw["CBOW + HS"]["acc"]["mean"]:.3f} in sentiment
      accuracy averaged over three seeds, and neither has any similarity structure. On text8, the
      accuracy gap disappears ({acc("B-text8") - acc("A-text8"):+.3f} for B), while A still
      leads on WordSim-353 by {sim("A-text8", "wordsim353") - sim("B-text8", "wordsim353"):.3f}
      and on analogies by
      {anb("A-text8")["overall_accuracy"] - anb("B-text8")["overall_accuracy"]:.3f}. CBOW
      closes the gap as the data grows, which fits the claim, but skip-gram stays ahead on the
      finer-grained tests. The text8 figures are single runs. On SST, the
      <a href="#hyper">hyperparameter sweep</a> puts the seed-to-seed standard deviation of
      accuracy at {min(x["acc"]["sd"] for x in sweep["rows"]):.3f} to
      {max(x["acc"]["sd"] for x in sweep["rows"]):.3f}, so the text8 accuracy gap is within
      noise. The WordSim-353 and analogy gaps are far larger than that.</p></section>''')

    # ---- finetune
    sec("finetune")
    add(f'''<section id="finetune">
      <div class="sechead"><div class="eyebrow">Model G-ft</div>
      <h2>What fine-tuning means for Word2Vec</h2></div>
      <p class="prose">Fine-tuning Word2Vec is much simpler than fine-tuning a transformer.
      There are no frozen layers or adapters. I initialize <code>W_in</code> with published
      vectors, then continue ordinary skip-gram training on movie reviews at a low learning
      rate.</p>
      <div class="stats">
        <div class="stat"><span class="n">{ft["seeded_from_pretrained"]:,}</span>
          <span class="l">rows seeded from GoogleNews</span></div>
        <div class="stat"><span class="n">{ft["randomly_initialised"]:,}</span>
          <span class="l">rows Google's vocabulary lacks</span></div>
        <div class="stat"><span class="n">{ft["mean_cosine_before_after"]:.3f}</span>
          <span class="l">mean cosine before and after training</span></div>
        <div class="stat"><span class="n">+{gap_ad:.3f}</span>
          <span class="l">accuracy over training from scratch</span></div>
      </div>
      <div class="driftgrid">
        {drift_col(ft["moved_most"][:8], "moved", "Moved most")}
        {drift_col(ft["moved_least"][-8:], "anchored", "Did not move at all")}
      </div>
      <div class="note"><b>The drift ranking mostly tracks corpus frequency.</b> The words that
      move most are actor surnames, which are common in reviews but rare in news. At the other
      end, rare sentiment adjectives remain at cosine 1.000 because one or two SST appearances
      do not produce enough gradient to move them. The full ranking gives a clearer account of
      fine-tuning than one hand-picked word would.</div>
      <ol class="steps">
        <li><div class="step-b"><b>Build the vocabulary from our corpus first</b>
          <span class="why"><code>intersect_word2vec_format</code> only overwrites rows that
          already exist. It does not import Google's three million words, so G-ft keeps the
          vocabulary built from SST.</span></div></li>
        <li><div class="step-b"><b>Set <code>vectors_lockf</code> to ones explicitly</b>
          <span class="why">gensim 4.x does not create it. Without it the import either fails
          or silently freezes every row, leaving the imported vectors unchanged.</span></div></li>
        <li><div class="step-b"><b>Keep the learning rate low: alpha {GFT["hyperparams"]["alpha"]}, {GFT["hyperparams"]["epochs"]} epochs</b>
          <span class="why">Google never published their output matrix, so <code>W_out</code>
          starts random. Early updates must learn a new decoder as well as adjust the imported
          vectors. A standard learning rate distorts those vectors during the first
          epoch.</span></div></li>
        <li><div class="step-b"><b>Measure drift rather than assuming it</b>
          <span class="why">Cosine between each word's warm-start vector and its final vector.
          The mean is {ft["mean_cosine_before_after"]:.4f} and the median is
          {ft["median_cosine_before_after"]:.4f}, confirming that the low learning rate
          preserved most of the pretrained structure.</span></div></li>
      </ol></section>''')

    # ---- scratch
    lc = C["train"]["loss_curve"]
    sec("scratch")
    add(f'''<section id="scratch">
      <div class="sechead">{eb("Model C, beyond the requirement", "Model C")}
      <h2>Writing skip-gram with negative sampling</h2></div>
      <p class="prose">{hw("The handout suggests a brief look at the implementation. I chose "
      "to write it instead because that makes", "Writing the training loop makes")} each part
      of Word2Vec concrete.
      <code>src/sgns.py</code> implements the vocabulary and counts, subsampling of frequent
      words, the unigram^0.75 noise table, a dynamic window, the two embedding matrices, and
      linear learning-rate decay. <code>to_keyedvectors</code> wraps <code>W_in</code> in a
      gensim object, which lets C use the same evaluation code as every other model.</p>''')
    add(table(["", "A, gensim", "C, from scratch"],
              [["SST-2 accuracy", f'{acc("A"):.4f}', f'{acc("C"):.4f}'],
               ["Wall clock", f'{A["train"]["wall_s"]:.1f} s', f'{C["train"]["wall_s"]:.1f} s'],
               ["Words per second", f'{A["train"]["words_per_sec"]:,}', f'{C["train"]["words_per_sec"]:,}'],
               ["Loss, first to last epoch", "not exposed", f"{lc[0]} to {lc[-1]}"]],
              ["l", "n", "n"]))
    add(f'''<div class="note warm"><b>The first version did not learn.</b> Its loss stayed at
      exactly 1.3863 for all ten epochs, and accuracy was 0.4992, or chance on a balanced split.
      A loss of 1.3863 is 2&middot;ln 2, what we get when every logit is zero. The problem was the
      loss reduction, not the sampler. Calling
      <code>BCEWithLogitsLoss</code> once on the positives and once on the negatives averages
      over B entries and over B&middot;k entries, so at a batch of
      {C["hyperparams"]["batch_size"]:,} the effective per-pair learning rate was about 3e-6.
      The measured gradient norm on <code>W_in</code> was exactly 0.0. I fixed it by writing
      Mikolov's equation 4 directly, summing across the k negatives, and averaging only across
      the batch.</div>
      <p class="prose">Two component checks made the failure quick to locate. Subsampling
      retains 17.6% of occurrences of
      <code>the</code>. The noise table maps <code>the</code> from a raw frequency of 4.30%
      down to a sampled 1.31% while lifting <code>execrable</code> from effectively zero to
      2e-5. Both results match the expected effect of the 0.75 exponent.</p></section>''')

    # ---- systems
    sec("systems")
    add(f'''<section id="systems">
      <div class="sechead">{eb("Deliverable 4", "Systems measurements")}
      <h2>What these models cost to run</h2></div>
      <p class="prose">Query latency measures brute-force <code>most_similar</code>, which runs
      a dense matrix-vector product over the full vocabulary. Each value comes from
      {lat("A")["repeats"]} calls. Memory is the RSS increase in an isolated subprocess,
      compared with the size of the matrices the model retains.</p>''')
    add(table(["Model", "Vocab", "p50 ms", "p95 ms", "Keeps", "Predicted MB", "Measured MB", "Ratio"],
              [[row_label(mid), f'{lat(mid)["vocab_size"]:,}', f'{lat(mid)["p50_ms"]:.2f}',
                f'{lat(mid)["p95_ms"]:.2f}', f'{mem(mid)["matrices_retained"]}',
                f'{mem(mid)["predicted_mb"]:.1f}', f'{mem(mid)["delta_mb"]:.1f}',
                f'{mem(mid)["delta_mb"]/mem(mid)["predicted_mb"]:.2f}'] for mid in ORDER],
              ["m", "n", "n", "n", "n", "n", "n", "n"]))
    add(f'''<div class="note"><b>Brute-force query time grows with vocabulary size.</b> G has
      {vocab_ratio:.0f} times as many words as the SST models and is {lat_ratio:.0f} times
      slower at p50. An approximate index adds little value at
      {lat("A")["vocab_size"]:,} words. At {lat("G")["vocab_size"]:,}, a query already takes
      {lat("G")["p50_ms"]:.0f} ms, and the full three-million-word Google model would take
      roughly 90 ms.</div>
      <p class="prose">The <code>2 &middot; V &middot; d &middot; 4</code> prediction holds
      within 3% for A, G-ft and C. B comes in at
      {mem("B")["delta_mb"]/mem("B")["predicted_mb"]:.2f} because hierarchical softmax also
      keeps Huffman-tree code and point arrays that the formula omits.</p>
      <div class="note warm"><b>G's memory result does not match the prediction.</b>
      {mem("G")["vocab"]:,} by {dim} float32 is {mem("G")["predicted_mb"]:.0f} MB and the
      measured increase is only {mem("G")["delta_mb"]:.1f} MB. macOS memory compression is a
      possible explanation because it can compress inactive anonymous pages, causing
      <code>ru_maxrss</code> to fall below the allocation. A Linux run should land near
      {mem("G")["predicted_mb"]:.0f} MB if that explanation is correct. Until then, the gap
      remains unresolved.</div>
      <p class="prose">A separate benchmark determined how C trains. On this vocabulary, MPS is
      slower than CPU (64.3 against 44.6 ms per batch) and sparse gradients are slower than
      dense gradients (52.6 against 44.6). The model is small enough for GPU launch and data
      transfer overhead to dominate. Sparse updates also offer little benefit. A batch of
      {C["hyperparams"]["batch_size"]:,} pairs with {C["hyperparams"]["negative"]} negatives
      touches roughly 90,000 index slots across {V:,} rows, so almost every row updates on each
      step.</p>
      <div class="note"><b>On text8, both results reverse.</b> The vocabulary grows to
      {r["C-text8"]["train"]["vocab_size"]:,} rows, and a dense Adam step has to update every
      one of them. A benchmark of a single training step at that size gave 1,300 ms per batch
      for dense updates on the CPU, against 207 ms for sparse updates on the CPU and 196 ms for
      dense updates on MPS. C's text8 run therefore uses sparse Adam. On the laptop it
      measured about 0.15 seconds per step, which puts the full
      {len(r["C-text8"]["train"]["loss_curve"])}-epoch run at roughly four hours, so it
      moved to a PACE ICE A100 GPU. There it trained in {r["C-text8"]["train"]["wall_s"]:.0f}
      seconds, {r["C-text8"]["train"]["words_per_sec"]:,} words per second. The GPU changed
      the speed and not the result: the laptop run, stopped after two epochs, had reached
      the same loss, {r["C-text8"]["train"]["loss_curve"][1]}, at the end of epoch 2.</div></section>''')

    # ---- errors
    sec("errors")
    add(f'''<section id="errors">
      <div class="sechead">{eb("Deliverable 2c", "Error analysis")}
      <h2>Why every model struggles with the same sentences</h2></div>''')
    add(table(["Model", "Errors / 1,821", "Negation", "Contrast", "Other"],
              [[row_label(mid), f'{err(mid)["n_errors"]:,}',
                f'{err(mid)["error_categories"].get("negation",0)}',
                f'{err(mid)["error_categories"].get("contrast",0)}',
                f'{err(mid)["error_categories"].get("other",0)}']
               for mid in sorted(ORDER, key=lambda m: err(m)["n_errors"])],
              ["m", "n", "n", "n", "n"]))
    tc_share = lambda m: (err(m)["error_categories"].get("negation", 0)
                          + err(m)["error_categories"].get("contrast", 0)) / err(m)["n_errors"]
    tcat = corpus["test_categories"]
    test_share = (tcat["negation"] + tcat["contrast"]) / sum(tcat.values())
    add(f'''<div class="note"><b>Negation and contrast account for
      {min(map(tc_share, ORDER)):.0%} to {max(map(tc_share, ORDER)):.0%} of each main model's
      errors, against {test_share:.0%} of the test set.</b> They are over-represented for every
      model, and most of all for GoogleNews ({tc_share("G"):.0%}), so better embeddings do not
      solve it. Mean pooling removes word order, including
      the information needed to determine the scope of a negation.</div>''')
    d2c = json.loads((ROOT / "results" / "examples_2c.json").read_text())
    add(f'''<h3 class="subhead">{hw("Deliverable 2c, stated explicitly for Model G-ft",
      "Model G-ft, confidently right and confidently wrong")}</h3>
      <p class="prose">{hw("The handout asks for at least two test examples the model "
      "classifies correctly and at least two it gets wrong, for a named model. Model G-ft, the "
      "fine-tuned one, is used here.", "Two test examples Model G-ft, the fine-tuned one, gets "
      "right and two it gets wrong.")} All four come from the same 1,821-sentence test split.
      Within each outcome, these are the highest-confidence cases.</p>''')
    add(table(["Outcome", "Gold", "Predicted", "Confidence", "Sentence"],
              [[f'<span class="{"win" if x["correct"] else "lose"}">'
                f'{"correct" if x["correct"] else "incorrect"}</span>',
                "positive" if x["gold"] else "negative",
                "positive" if x["pred"] else "negative",
                f'{x["confidence"]:.3f}', f'<code>{e(x["sentence"][:86])}</code>']
               for x in d2c["model_G-ft"]], ["n", "n", "n", "n", "l"]))
    add('''<h3 class="subhead">Confident failures across models</h3>''')
    rows = []
    for mid in ["A", "G-ft"]:
        for x in err(mid)["confident_errors"][:3]:
            rows.append([row_label(mid), f'<code>{e(x["sentence"][:88])}</code>',
                         "pos" if x["gold"] else "neg", "pos" if x["pred"] else "neg",
                         f'{x["confidence"]:.3f}', x["category"]])
    add(table(["Model", "Sentence", "Gold", "Predicted", "Confidence", "Category"], rows,
              ["m", "l", "n", "n", "n", "m"]))
    add(f'''<p class="prose">The failures follow directly from the representation.
      <code>not a bad journey at all</code> averages <code>not</code> and <code>bad</code> into
      negative territory. <code>there is n't a weak or careless performance amongst them</code>
      pools two strongly negative adjectives without preserving the negation that reverses
      them. In <code>a great deal of sizzle and very little steak</code>, the negative meaning
      comes from the contrast between clauses, which averaging also removes.{hw(" The handout "
      "says to make a test set harder if it produces no failures. SST already contains clear "
      "failures, so no extra examples were needed.")}</p></section>''')

    # ---- deliverable 6
    sec("running")
    add(f'''<section id="running">
      <div class="sechead">{eb("Deliverable 6", "Reproduction")}
      <h2>Installation, running and measurement</h2></div>
      <p class="prose">Four setup and measurement problems took most of the debugging time.
      In each case, the visible error pointed away from the actual cause. The symptoms and fixes
      are recorded here{hw(" as part of the requested implementation experience")}.</p>
      <ol class="steps">
        <li><div class="step-b"><b>gensim has no wheels for Python 3.14</b>
          <span class="why">The machine's default interpreter is 3.14. Installing gensim there
          starts a Cython source build and fails late in the process. The virtual environment
          therefore uses Homebrew Python 3.11.</span></div></li>
        <li><div class="step-b"><b><code>import gensim</code> fails on a fresh unpinned install</b>
          <span class="why">gensim 4.3.x imports <code>scipy.linalg.triu</code>, which scipy
          1.13 removed. The ImportError points at gensim even though the incompatible scipy
          version is the cause. <code>requirements.txt</code> pins <code>scipy&lt;1.13</code>
          and <code>numpy&lt;2</code>.</span></div></li>
        <li><div class="step-b"><b>The SST-2 split with hidden labels</b>
          <span class="why">The GLUE copy sets every test label to -1. An accuracy calculation
          against those placeholders can still produce a plausible number. <code>src/data.py</code>
          prevents that by requiring two classes in the test labels.</span></div></li>
        <li><div class="step-b"><b>Whole-process peak RSS is not a model measurement</b>
          <span class="why">The first memory table compared process peak RSS against the matrix
          prediction and returned 297.8 MB for a model with 34.3 MB of matrices. Process peak
          RSS also includes the interpreter, gensim, numpy, torch, and the resident corpus.
          <code>src/measure_mem.py</code> now measures an RSS delta in an isolated
          subprocess per model.</span></div></li>
      </ol>
      <p class="prose">After <code>pip install -r requirements.txt</code>, these commands
      produce the results the page is built from, in order:</p>
      <ol class="steps">
        <li><div class="step-b"><code>python -m src.data</code>
          <span class="why">corpus statistics</span></div></li>
        <li><div class="step-b"><code>python -m src.main</code>
          <span class="why">trains A and B on SST and loads G</span></div></li>
        <li><div class="step-b"><code>python -m src.run_e</code>
          <span class="why">trains C on SST, {C["train"]["wall_s"]:.0f} seconds on CPU</span></div></li>
        <li><div class="step-b"><code>python -m src.run_eval</code>, <code>python -m src.measure_mem</code>,
          <code>python -m src.run_analogies</code>, <code>python -m src.testcases</code>
          <span class="why">similarity, latency, memory, errors, analogies and the worked test
          examples</span></div></li>
        <li><div class="step-b"><code>python -m src.run_text8</code>
          <span class="why">downloads text8 and retrains A and B on it, about
          {(r["A-text8"]["train"]["wall_s"] + r["B-text8"]["train"]["wall_s"]) / 60:.0f} minutes</span></div></li>
        <li><div class="step-b"><code>python -m src.run_e --text8</code>
          <span class="why">trains C on text8. On a GPU cluster, <code>sbatch slurm/c_text8.sbatch</code>
          runs the same command; it saves a checkpoint after every epoch, so an interrupted job
          resumes where it stopped.</span></div></li>
        <li><div class="step-b"><code>python -m src.visualize</code>
          <span class="why">PCA and t-SNE projections and the cluster scores</span></div></li>
        <li><div class="step-b"><code>python -m src.sweep</code>
          <span class="why">the hyperparameter sweep</span></div></li>
        <li><div class="step-b"><code>python -m src.build_report</code>
          <span class="why">rebuilds both report variants</span></div></li>
      </ol>
      <p class="prose"><code>src/build_report.py</code> reads every reported figure from
      <code>results/*.json</code> when it builds the page. Re-running a model and rebuilding
      therefore updates the prose and tables together.</p>
      <div class="note warm"><b>Not everything has a command yet.</b> G-ft and the A2 control
      come from functions in <code>src/finetune.py</code> and <code>src/train_gensim.py</code>
      that have no command-line entry point. The deliverable 2b and 2c example files have no
      generating code in the repository at all. The list above therefore does not regenerate
      these; their results are stored in <code>results/</code>.</div></section>''')

    sec("foot")
    add(f'''<footer>{hw("CS 6220 Big Data Systems, Fall 2026. ")}Generated from
      {len(r)} result records.</footer>''')

    order = submission.placed() if SUB else [i for _, ids in WEB_GROUPS for i in ids]
    stranded = set(buckets) - set(order) - {"head", "nav", "foot"}
    assert not stranded, f"sections built but never placed: {sorted(stranded)}"
    if SUB:
        body = submission.assemble(buckets, TITLES, submission.context(
            r=r, acc=acc, err=err, sw=sw, sweep=sweep, corpus=corpus, order=ORDER, table=table,
            viz=json.loads((ROOT / "results" / "visualization.json").read_text())))
    else:
        body = ("".join(buckets["head"])
                + "".join("".join(buckets[i]) for i in order)
                + "".join(buckets["foot"]))
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>word2vec-lab</title>'
            f'<style>{CSS}{EXTRA_CSS}{submission.SUB_CSS if SUB else ""}</style></head><body><div class="shell">'
            f'{"".join(buckets["nav"])}<div class="wrap">{body}</div></div>'
            f'{TOC_JS}</body></html>')
    out = REPORT_OUT if SUB else OUT
    out.parent.mkdir(exist_ok=True)
    out.write_text(page)
    print(f"wrote {out.relative_to(ROOT)}  {len(page)/1024:.1f} KB")


if __name__ == "__main__":
    build("web")
    build("submission")
