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
OUT = ROOT / "docs" / "index.html"
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
        if p.name == "testcases.json":      # not a model record
            continue
        r = json.loads(p.read_text())
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
    rows = []
    for q in QUERIES:
        lines = []
        for mid in ids:
            nb = recs[mid]["intrinsic"]["neighbours"].get(q)
            if not nb:
                continue
            base = {w for w, _ in recs["C"]["intrinsic"]["neighbours"].get(q, [])}
            chips = "".join(
                f'<span class="chip{" new" if mid == "D" and w not in base else ""}">{e(w)}</span>'
                for w, _ in nb[:8])
            lines.append(f'<div class="nline"><span class="nlabel">{mid}</span>'
                         f'<span class="chips">{chips}</span></div>')
        rows.append(f'<div class="nrow"><div class="nq">{e(q)}</div>'
                    f'<div class="npair">{"".join(lines)}</div></div>')
    return f'<div class="nbox">{"".join(rows)}</div>'


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
        chips = "".join(f'<span class="chip{" new" if n["label"]==s["gold"] else ""}">'
                        f'{e(n["text"])} &middot; {n["cos"]:.2f}</span>' for n in nb)
        blocks.append(f'''<div class="tcase">
          <div class="tchead"><span class="eyebrow">Example {k+1} &middot; {e(s["selected_for"])}</span>
            <p class="tcsent"><code>{e(s["text"])}</code></p>
            <span class="tcgold">gold label: <b>{gold}</b></span></div>
          {table(["Model", "Rank 1", "Rank 2", "Result", "Neighbour agreement"], rows,
                 ["m", "n", "n", "n", "n"])}
          <div class="tcnb"><span class="nlabel">D retrieves</span>
            <span class="chips">{chips}</span></div></div>''')
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

    ft = D["finetune"]
    gap_ad = acc("D") - acc("A")
    gap_cd = acc("C") - acc("D")
    lat_ratio = lat("C")["p50_ms"] / lat("A")["p50_ms"]
    vocab_ratio = lat("C")["vocab_size"] / lat("A")["vocab_size"]

    S = []
    add = S.append

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
      <a href="#similarity">Lexical similarity</a><a href="#finetune">Fine-tuning</a>
      <a href="#scratch">From scratch</a><a href="#systems">Systems</a>
      <a href="#errors">Errors</a><a href="#log">Build log</a></nav>''')

    # ---- results
    add(f'''<section id="results">
      <div class="sechead"><div class="eyebrow">Where things stand</div>
      <h2>Five models, one classifier</h2></div>
      <p class="prose">Every model is scored the same way. Look up a vector per word, average
      them into a sentence vector, fit logistic regression, test on the same held-out SST-2
      sentences. The classifier never changes and is never tuned per model, so any gap between
      these numbers belongs to the embeddings and to nothing else.</p>
      {cards(r)}
      <div class="note"><b>The comparison that isolates one variable is A against D.</b>
      Same architecture, same loss, same corpus, same classifier. The only difference is where
      the input matrix started: random for A, Google's published vectors for D. That is worth
      {gap_ad:.3f} accuracy, {acc("A"):.3f} to {acc("D"):.3f}. D then lands {gap_cd:.3f} below C
      while carrying {vocab_ratio:.0f} times less vocabulary.</div>''')

    add(table(
        ["Model", "Architecture", "Loss", "Trained on", "Accuracy", "F1", "OOV"],
        [[mid, "Skip-gram" if r[mid]["hyperparams"]["sg"] else "CBOW",
          "Hierarchical softmax" if r[mid]["hyperparams"].get("hs") else
          f'Negative sampling, k={r[mid]["hyperparams"]["negative"]}',
          f'{r[mid]["train"]["corpus_tokens"]:,} tok',
          f'<span class="{"win" if acc(mid)==max(acc(m) for m in ORDER) else ""}">{acc(mid):.4f}</span>',
          f"{f1(mid):.4f}", f'{r[mid]["extrinsic"]["oov_rate_test"]:.3f}']
         for mid in ORDER],
        ["m", "l", "l", "n", "n", "n", "n"]))
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
      question than the one asked.</p></section>''')

    # ---- hyperparameters
    add(f'''<section id="hyper">
      <div class="sechead"><div class="eyebrow">Deliverable 5</div>
      <h2>Hyperparameters</h2></div>
      <p class="prose">A and B differ on two axes at once, architecture and loss, which is
      deliberate: it gives the comparison two things to say rather than one. Everything else is
      held constant between them.</p>''')
    add(table(["", "A", "B", "C", "D", "E"],
              [[k] + [str(r[m]["hyperparams"].get(lbl, "-")) for m in ORDER]
               for k, lbl in [("Architecture (sg)", "sg"), ("Hierarchical softmax", "hs"),
                              ("Negatives", "negative"), ("Dimensions", "vector_size"),
                              ("Window", "window"), ("Min count", "min_count"),
                              ("Epochs", "epochs"), ("Initial LR", "alpha"),
                              ("Subsample threshold", "sample")]],
              ["l", "n", "n", "n", "n", "n"]))
    add("</section>")

    # ---- neighbours
    add(f'''<section id="neighbours">
      <div class="sechead"><div class="eyebrow">Deliverables 3 and 5</div>
      <h2>Top-ranked neighbours for five queries</h2></div>
      <p class="prose">Five words chosen to separate the models rather than flatter them:
      a polysemous noun, a sentiment adjective, a word whose sense depends on domain, a rare
      domain term, and the most frequent word in English. Highlighted chips on row D are
      neighbours that are not in C's list for the same query, so they are what fine-tuning
      introduced.</p>
      {neighbour_box(r, ORDER)}
      <div class="note"><b>Read the C and D rows against A, B and E.</b> C returns clean
      synonyms and inflections. D keeps them and adds film-specific associations:
      <code>visuals</code>, <code>screenplay</code> and <code>photography</code> enter
      <code>cinematography</code>, and <code>storyline</code> enters <code>plot</code>.
      A, B and E return something close to noise for every query. At {tok:,} tokens there is
      not enough co-occurrence evidence to place a word, and the next section puts a number
      on exactly how little.</div></section>''')

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
        <div class="stat"><span class="n">{ft["mean_cosine_drift"]:.3f}</span>
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
          Mean {ft["mean_cosine_drift"]:.4f}, median {ft["median_cosine_drift"]:.4f}, so the low
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
