# HW2 implementation plan

> **Model IDs were renamed on Sep 23 2026.** This document now uses the new letters: C is the from-scratch model (formerly E), G is GoogleNews-300 (formerly C), and G-ft is the fine-tune (formerly D). Commit messages keep the old letters.

Companion to `PLAN.md`, which explains what the assignment is and why. This file is the
ordered build sequence. Read `PLAN.md` first for the reasoning; nothing here re-argues it.

Ordering rule: **every required deliverable is complete and submittable before any extra
work starts.** Phases 0 through 9 produce a finished submission. Phases 10 through 13 are
the extras from `PLAN.md` Part 4, layered on top, and each one is independently droppable.
If Wednesday goes badly, phase 9 already shipped.

Each step lists what it produces, how to tell it worked, and a rough cost. The acceptance
check matters more than the description: a step is not done because code ran, it is done
because the check passed.

---

## Phase 0. Scaffold

### Step 0.1. Python environment

Local default is 3.14 and gensim publishes no wheels for it, so building would mean
compiling Cython by hand. Homebrew has 3.11 installed already. Use it.

```sh
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -V              # expect 3.11.x
```

Known-good pins, written into `requirements.txt`:

```
gensim==4.3.3
numpy<2
scipy<1.13
scikit-learn
matplotlib
pandas
tqdm
```

The scipy pin is not cosmetic. gensim 4.3.x imports `scipy.linalg.triu`, which scipy 1.13
removed, so a fresh unpinned install fails at `import gensim` with an ImportError that
looks like a broken gensim rather than a version conflict. PyTorch is not installed yet; it
arrives in phase 10 with the from-scratch model.

Acceptance: `python -c "import gensim, sklearn, numpy; print(gensim.__version__)"` prints
4.3.3 without a traceback.

Cost: 15 minutes, most of it waiting on downloads.

### Step 0.2. Repository

Create `moriowen/6220-bds` as a private repo, with `HW2/` inside it. Leave `HW1/` where it
is; it predates the one-repo-per-course convention. `.gitignore` covers `.venv/`,
`__pycache__/`, `*.bin`, `*.gz`, `models/`, and `data/text8/`. The large downloads are
reproducible from URLs and must not go into git history, but note that `data/sst2/` is
small and **does** get committed, because deliverable 2 requires shipping the datasets.

Acceptance: `git status` is clean after a fresh `python -c "import gensim"`, meaning no
stray artefacts are being tracked.

Cost: 15 minutes.

### Step 0.3. The results contract

Before any model code, fix the shape of `results/*.json`. Everything downstream reads it,
and changing it later means re-running everything.

```python
{
  "model_id": "A",
  "name": "gensim skip-gram + negative sampling",
  "corpus": "sst2",
  "machine": "m2-air" | "pace-ice-cpu" | "pace-ice-a100",
  "hyperparams": {"sg": 1, "vector_size": 300, "window": 5, "negative": 10,
                  "min_count": 2, "epochs": 10, "alpha": 0.025, "sample": 1e-3},
  "train": {"wall_s": 0.0, "words_per_sec": 0.0, "peak_rss_mb": 0.0,
            "vocab_size": 0, "corpus_tokens": 0},
  "intrinsic": {"neighbours": {"plot": [["storyline", 0.71], ...]},
                "wordsim353_spearman": 0.0},
  "extrinsic": {"sst2_test_acc": 0.0, "sst2_test_f1": 0.0,
                "query_latency_ms": {"p50": 0.0, "p95": 0.0}},
  "errors": [{"sentence": "...", "gold": 1, "pred": 0, "confidence": 0.91,
              "category": "negation"}]
}
```

Two rules that follow from it. First, `machine` is mandatory on every record, for the
reason HW1 established: an A100 wall-clock and a fanless-laptop wall-clock are
indistinguishable in the output and not comparable, and no table may mix them. Second,
every model writes this same schema, which is what lets Model C be added in phase 10 by
registering one more entry rather than by touching the report.

Acceptance: a hand-written stub JSON for a fake model passes a small `validate_result()`
function.

Cost: 30 minutes. Worth every minute; skipping it is how the last day turns into
reconciliation work.

---

## Phase 1. Data

### Step 1.1. SST-2

Download `SetFit/sst2`, which carries real test labels. The GLUE version
(`stanfordnlp/sst2`) sets every test label to -1 because the split is held out for the
leaderboard, and discovering that on Thursday is a bad afternoon.

Write `src/data.py` with `load_sst2()` returning train/dev/test as lists of
`(tokens, label)`. Tokenisation is lowercase plus whitespace split; SST is already
tokenised, so nothing cleverer is needed, and anything cleverer would have to be justified
in the report.

Acceptance, and this is the guard against the wrong-split trap: assert the test labels
contain both 0 and 1 and are roughly balanced. Print split sizes and total token counts.
Expect about 67k train rows and a labelled test split in the low thousands.

Cost: 45 minutes.

### Step 1.2. data/README.md

Source URL, licence, split sizes, token counts, vocabulary size, and which SST variant was
used and why. Deliverable 2 asks for the datasets described and shipped; this file is the
description and `data/sst2/` is the shipping.

Acceptance: every number in it was printed by step 1.1, not estimated.

Cost: 20 minutes.

---

## Phase 2. Models A and B

### Step 2.1. Training wrapper

`src/train_gensim.py`, one function taking a config dict and returning a trained model plus
the `train` block of the results schema. Measure wall clock with `time.perf_counter()` and
peak memory with `resource.getrusage(RUSAGE_SELF).ru_maxrss`, noting that on macOS that
value is bytes while on Linux it is kilobytes, so the conversion is platform-dependent and
the ICE runs will report differently from the laptop if this is not handled.

Model A: `sg=1, negative=10, hs=0, vector_size=300, window=5, min_count=2, epochs=10`.
Model B: `sg=0, hs=1, negative=0`, everything else identical.

Holding everything but architecture and loss constant is what makes the A-versus-B
comparison in deliverable 1 mean anything.

Acceptance: both models train, both report a vocabulary of the expected size, and
`model_a.wv.most_similar("good")` returns words rather than an exception. Neighbour quality
is not judged yet.

Cost: 1 hour including the training runs.

### Step 2.2. Persist

Save to `models/` with `model.save()`, not `save_word2vec_format()`, because the full model
keeps `W_out` and the vocabulary counts, and phase 5's fine-tune needs them. Record every
hyperparameter into the results JSON automatically from the model object rather than by
retyping the config, so the report can never disagree with what actually ran.

Acceptance: a model loads back and produces identical neighbours for `good`.

Cost: 20 minutes.

---

## Phase 3. The sentiment pipeline

This is the step that turns embeddings into the accuracy number every later comparison
rests on.

### Step 3.1. Pooling

`src/pooling.py`. `mean_pool(tokens, kv)` looks up each in-vocabulary token, averages, and
returns a zero vector for a sentence with no known words. Track and report the
out-of-vocabulary rate per model, because it differs sharply between a 15k-word SST
vocabulary and GoogleNews, and it partly explains any accuracy gap.

Add `tfidf_pool` as a second variant. It is a few lines and gives the report a second row
rather than a single point.

Acceptance: pooled matrix has shape `(n_sentences, 300)` with no NaNs. OOV rate is printed.

Cost: 45 minutes.

### Step 3.2. Classifier

`sklearn.linear_model.LogisticRegression(max_iter=1000)`, fit on pooled train, scored on
pooled test. The same function serves every model with no per-model branching. That
uniformity is the experimental control described in `PLAN.md` section 1.3, and any
special-casing here silently invalidates the comparison.

Acceptance: A and B both land somewhere in the 70s or low 80s on SST-2 test. A number below
65 means the pipeline is broken, not that the embeddings are bad. A number above 90 means
label leakage, most likely from evaluating on the training split.

Cost: 30 minutes.

**Monday night ends here.** Phases 0 through 3 are the ugly end-to-end pipeline: data in,
two trained models, one accuracy number each. The assignment now has a spine.

---

## Phase 4. Model G, the pretrained reference

### Step 4.1. Load GoogleNews-300

Via `gensim.downloader.load("word2vec-google-news-300")`, or the raw `.bin` with
`KeyedVectors.load_word2vec_format(path, binary=True, limit=500000)`.

Use the limit on the laptop. The full file is 3.4 GB on disk and wants roughly 4 GB
resident, which is most of an 8 GB machine. The top 500,000 words cover essentially all of
SST. If the limit is used it must be stated in the report, because it changes the
brute-force query latency numbers in phase 7 and those numbers would otherwise be
incomparable to the unlimited case.

Acceptance: `kv.most_similar("plot")` returns conspiracy-flavoured neighbours. That is the
expected result, not a bug, and it is the demonstration that makes the fine-tune in phase 5
legible.

Cost: 45 minutes, mostly download.

### Step 4.2. Score it through the same pipeline

No new code. Same pooling, same classifier.

Acceptance: G's accuracy is recorded and its OOV rate is much lower than A's or B's.

Cost: 15 minutes.

---

## Phase 5. Model G-ft, the fine-tune

The recipe is in `PLAN.md` section 1.4 with the gensim calls spelled out. Three things to
get right, in this order of likely failure:

1. `vectors_lockf` must be explicitly set to ones. gensim 4.x does not create it by default
   and `intersect_word2vec_format` will fail or silently freeze everything without it.
2. `alpha` low, around 0.005, and few epochs, around 5 to 8. `W_out` starts random because
   Google never published theirs, so the first updates are spent re-learning a decoder, and
   a normal learning rate scrambles the imported vectors in the first epoch.
3. Measure drift. For every word present in both G and G-ft, compute cosine between its
   before-vector and its after-vector. Report the mean, and the ten words that moved most.

Acceptance, and this is the real gate: `plot` in Model G-ft must sit closer to storyline senses
than it does in Model G, while a word absent from SST is unchanged. If mean drift is very
high, say above 0.4, the learning rate is too high and the fine-tune is destroying rather
than adapting. Turn `alpha` down and rerun.

The drift measurement is not required by the handout. It is four lines of code and it is
what turns this section of the report from a recipe into a result.

Cost: 1 hour 30 minutes including at least one rerun, which should be expected rather than
hoped against.

---

## Phase 6. Intrinsic evaluation

### Step 6.1. Nearest neighbours for the five queries

`plot`, `terrible`, `star`, `cinematography`, `the`. Rationale for each is in `PLAN.md`
Part 8. Emit top-1, top-5 and top-10 for all four models into the results JSON, which is
deliverable 5's table directly.

Acceptance: the table is generated, not transcribed, and `plot` differs visibly between the
SST-trained models and G.

Cost: 30 minutes.

### Step 6.2. WordSim-353

Download the 353 pairs, compute Spearman between human ratings and cosine similarity, skip
pairs with an OOV word and report how many were skipped. The skip count is part of the
result, since a model that scores well on half the pairs has not scored well.

Acceptance: G lands around 0.6 to 0.7. A and B will be much lower, which is the expected
consequence of a million-token corpus and is worth stating plainly rather than apologising
for.

Cost: 45 minutes.

---

## Phase 7. Measurement, deliverable 4

Training numbers were captured in phase 2. What remains:

- Query latency for `most_similar`, 200 repeats per model, reported at p50 and p95. This
  scales with vocabulary size, so G at 500k words should be visibly slower than A at 15k,
  and that contrast is the finding.
- Classification throughput: sentences per second for pooling plus prediction.
- Peak RSS per model against the predicted `2 * V * d * 4` bytes. Explaining any gap between
  predicted and measured is more interesting than the numbers themselves.

Acceptance: one table with four model rows and every cell populated, `machine` recorded on
each.

Cost: 1 hour.

---

## Phase 8. Error mining, deliverable 2c

Sort test sentences by classifier confidence, take the most confident mistakes, and hand-label
each into negation, contrast, or sarcasm per `PLAN.md` Part 7. Pull at least two correct and
two incorrect examples per model, which is what 2c requires, plus the five training examples
and four test examples deliverable 2b wants, quoted verbatim.

Acceptance: at least two confident errors per model, each with an explanation grounded in
the pooling step rather than a shrug. The handout is explicit that a test set producing no
failures must be made harder, so if this comes up empty, add negation-heavy sentences by
hand until it does not.

Cost: 1 hour, and the hand-labelling is the slow part.

---

## Phase 9. Report and submission

### Step 9.1. Architecture figures

Two hand-drawn SVGs, one per model, showing one-hot input of size V, the `W_in` multiply to
d dimensions, the code layer, the `W_out` multiply back to V, and the output. Skip-gram
shows one input predicting C context slots; CBOW shows C inputs averaged into one
prediction. Include the note from `PLAN.md` section 1.2 about the encoder/decoder framing
fitting word2vec only loosely.

Cost: 1 hour 30 minutes. Do not spend three.

### Step 9.2. Build the report

`report/report.html` to PDF, the same pipeline HW1 already used. Write against the
deliverable checklist in `PLAN.md` Part 8 literally, ticking rows. Every number resolves
from `results/*.json` at build time, so nothing in the prose can go stale when a run is
repeated.

Cost: 3 hours.

### Step 9.3. Ship

Zip with `data/` inside it. Filename matching HW1's convention,
`HW2_P_Mohite_Atharva.pdf`. Submit Thursday night. HW1 closed at 23:55 rather than 23:59
and HW2's grace window is unconfirmed, so the deadline is hard.

Cost: 30 minutes.

**The assignment is complete at the end of phase 9.** Everything below is extra.

---

## Phase 10. Model C, from scratch

The reason this project is worth doing, and now safely on top of a finished submission.
Build order, each step with the trap it carries, from `PLAN.md` section 4.1:

| Step | What | The trap |
|---|---|---|
| 10.1 | Vocab and counts | none |
| 10.2 | Subsampling, `1 - sqrt(t/f) - t/f`, `t=1e-5` | discarding too little; check that `the` loses most of its occurrences |
| 10.3 | Negative sampling table, unigram to the 0.75 | using raw unigram. Loss still falls smoothly and vectors are garbage |
| 10.4 | Dynamic window, uniform from 1 to `window` | forgetting it; it is one line in the original C and easy to miss |
| 10.5 | Two `nn.Embedding` layers, dot product, BCE over 1 positive and k negatives | none |
| 10.6 | Linear learning rate decay | none |

Correctness gate before anything enters the report: on the same corpus with matched
hyperparameters, C's neighbours for `king` and its WordSim-353 correlation must be close to
gensim's. Gate on neighbour quality, never on the loss curve, for the reason in 10.3.

Then register C in the results schema and rerun the harness. Because every model emits the
same JSON, the report picks it up with no edits beyond the prose.

Cost: 4 to 6 hours.

## Phase 11. text8

Download, train A, B and C on it, and compare against the SST-trained versions. This is
where the handout's page-1 claim gets tested: skip-gram is supposed to beat CBOW on small
data and lose on large. Report whichever way it comes out.

Run C on an `ice-gpu` A100 allocation per `PLAN.md` Part 5. 17 million tokens with a 253k
vocabulary is a real GPU workload and is minutes per epoch there against hours locally.

Cost: 2 to 3 hours of attention, plus training time.

## Phase 12. Hogwild scaling study

gensim `workers` at 1, 2, 4, 8, 16 on an `ice-cpu` node, plotting words per second. Find
where it stops scaling. The falloff is usually the single Python thread feeding the C
workers rather than the workers themselves, which makes it a clean concrete example of a
data pipeline bottleneck, which is the course's actual subject.

Cost: 2 hours.

## Phase 13. Hyperparameter sweep

Six to eight configurations, not a grid. Vary window in {2, 5, 10}, dimension in
{50, 100, 300}, negatives in {5, 15}. Plot training time against SST accuracy so the report
answers what the last two points of accuracy cost rather than only which setting won.

Cost: 2 hours plus training time.

---

## Schedule

| When | Phases | Outcome |
|---|---|---|
| Mon evening, 3h | 0 to 3 | pipeline runs end to end, two models, two accuracy numbers |
| Tue, 5h | 4 to 8 | all four models done, every required measurement captured |
| Wed, 5h | 9, then 10 | submission built and ready; from-scratch model started |
| Thu, 4h | finish 10, submit | assignment submitted Thursday night |
| Fri onward | 11 to 13 | extras, on a finished submission |

The submission is ready at the end of Wednesday rather than Thursday, which buys a full day
of slack against Sep 25 also carrying 6242 project teams and 7641 A1 and Q4.

## Cut order

If time runs short, cut in this order, which is `PLAN.md` Part 9 restated against these
phase numbers: phase 13, then 12, then 11, then 10, then step 6.2, then step 3.1's tf-idf
variant. Phases 0 through 5 and 7 through 9 are not cuttable, because each one carries a
numbered deliverable.

---

## Run log

### Sep 22, phases 0 through 3 complete

Environment, data, models A and B, pooling and classifier all working end to end. First
numbers, from `results/A_sst2.json` and `results/B_sst2.json`, on an M2 Air:

| | Model A, skip-gram + NS | Model B, CBOW + HS |
|---|---|---|
| Vocabulary | 7,140 | 7,140 |
| Train wall clock | 2.49 s | 0.93 s |
| Words/sec | 536k | 1.44M |
| SST-2 test accuracy | 0.679 | 0.628 |
| SST-2 test F1 | 0.696 | 0.652 |
| Test OOV rate | 0.095 | 0.095 |

Three observations worth carrying into the report:

**CBOW trains 2.7 times faster and scores 5 points worse.** The handout asserts both halves
of this on page 1 (CBOW is faster, skip-gram does better on small data) and on a 134k-token
corpus both hold. Phase 11 on text8 tests whether the accuracy half reverses at scale, which
is the more interesting question.

**Accuracy is lower than the 70s the plan predicted, and the cause is corpus size, not a
broken pipeline.** The acceptance check in phase 3 said below 65 means broken; B at 62.8 sits
just under that line. The diagnosis against it: both models share an identical pipeline and
differ only in architecture, OOV is a reasonable 9.5%, and the split is balanced at 49.9%
positive, so a broken pipeline would not produce an ordered, explainable gap. 134k tokens is
simply thin. See the size caveat in `data/README.md`.

**Neighbour quality is poor for all five report queries.** `terrible` returns `saves,
glorified, exploiting`, which is close to noise. This is the same corpus-size problem and it
is exactly what Model G will contrast against in phase 4, so it is useful evidence rather
than a failure. It does mean the phrase-level SST rows should be tried before the report is
written.

Next: phase 4, Model G.

### Sep 22, phrase-level corpus and Model G

Two changes. Embedding and classifier training moved from the 6,920 sentence-level rows to
the 67,348 phrase-level rows, 4.7 times more text. Model G added.

| | A, skip-gram + NS | B, CBOW + HS | G, GoogleNews-300 |
|---|---|---|---|
| Training corpus | SST phrases, 634k tok | SST phrases, 634k tok | news, ~100B tok |
| Vocabulary | 14,309 | 14,309 | 500,000 (limited) |
| Wall clock | 11.78 s | 3.58 s | 2.84 s to load |
| Peak RSS | 297.8 MB | 297.8 MB | 432.8 MB |
| SST-2 test accuracy | 0.761 | 0.729 | **0.798** |
| SST-2 test F1 | 0.776 | 0.750 | 0.811 |
| Test OOV | 0.066 | 0.066 | 0.271 |

**The corpus change was worth 8 points to A and 10 to B** (0.679 to 0.761, 0.628 to 0.729),
and doubled vocabulary from 7,140 to 14,309. This confirms the diagnosis in the previous
entry: the earlier numbers were limited by corpus size, not by a broken pipeline.

**Skip-gram still beats CBOW, and CBOW still trains 3.3 times faster.** Both halves of the
handout's page-1 claim hold at 634k tokens as they did at 134k. The accuracy gap narrowed
slightly, from 5.1 points to 3.2, which is the direction the claim predicts as data grows.
text8 in phase 11 is what tests whether it crosses over.

**Model G wins by 3.7 points despite a 27% OOV rate.** The composition of that rate is
measured in `data/README.md`: 40% punctuation, most of the rest very frequent function words
that Google dropped when publishing. None of it carries sentiment. A capitalisation fallback
would recover half the misses and was rejected, because it maps `a` to the letter `A` and
`and` to the acronym `AND`.

**Neighbour quality separates the models more sharply than accuracy does.** G returns
inflectional variants (`plot` gives `plots, Plot, plotting, plotline`) and clean synonyms
(`terrible` gives `horrible, horrendous, dreadful, awful`). A and B return near-noise for
the same queries. Two things follow for the report: 634k tokens is still far too little for
good neighbours even though it is enough for competitive classification accuracy, and the
two evaluation modes are measuring genuinely different things. That gap is a result in its
own right and belongs in the deliverable 5 discussion rather than being smoothed over.

Next: phase 5, Model G-ft, the fine-tune. G's `plot` neighbourhood is currently morphological
rather than conspiratorial, which is a weaker starting contrast than the plan assumed, so
the drift measurement matters more than the single-word anecdote.

### Sep 22, phase 5, Model G-ft

| | A, sg+NS | B, CBOW+HS | G, GoogleNews | G-ft, fine-tuned |
|---|---|---|---|---|
| Vocabulary | 14,309 | 14,309 | 500,000 | 14,309 |
| Wall clock | 11.78 s | 3.58 s | 2.84 s load | 13.80 s |
| SST-2 test accuracy | 0.761 | 0.729 | **0.798** | 0.792 |
| SST-2 test F1 | 0.776 | 0.750 | 0.811 | 0.809 |
| Test OOV | 0.066 | 0.066 | 0.271 | 0.066 |

Of the 14,309 vocabulary entries, 12,032 were seeded from GoogleNews and 2,277 kept their
random initialisation because Google's published vocabulary does not contain them.

**The warm start is worth 3.1 points over training from scratch on the same corpus**, 0.761
for A against 0.792 for G-ft, with identical architecture, identical loss and an identical
classifier. That is the cleanest single comparison in the assignment so far, because the only
difference between A and G-ft is where W_in started.

**G-ft lands 0.6 points below G and this is not a failure.** G has 35 times the vocabulary and
100,000 times the training text. G-ft matches it to within noise while carrying a vocabulary
small enough to hold in memory without limiting, and it does so at a quarter of G's OOV rate.
The interesting comparison is not which number is larger but that they are the same number.

**Drift came out healthy.** Mean cosine between each word's warm-start vector and its
post-training vector is 0.9609, median 0.9731. The plan's failure threshold was a mean
cosine below about 0.6, which would have meant the low alpha was not low enough and the
randomly initialised W_out was scrambling the imported vectors. It did not happen, so no
rerun was needed.

**The words that moved are exactly the ones domain shift predicts.** Ranked by drift, the
top of the list is actor surnames: `kidman` 0.553, `philippe` 0.599, `schwarzenegger` 0.608,
`holm` 0.647, `griffiths` 0.663. These are frequent in movie reviews and rare in news, so
they get heavily retrained. At the other end, cosine 1.000 with no measurable movement at
all: `execrable`, `abysmally`, `drudgery`, `worshipful`, `plod`. Rare sentiment adjectives
that appear once or twice in SST and never accumulate enough gradient to move. The drift
distribution is therefore a direct readout of corpus frequency, and it is a better answer to
"what did fine-tuning do" than any single-word anecdote would have been.

**Neighbour quality is where G-ft separates from G.** For `cinematography`, G returns only
morphological variants (`camerawork, cinematographer, Cinematography, Cinematographer`) while
G-ft returns `camerawork, cinematographer, visuals, screenplay, photography`. `visuals` and
`screenplay` are film-domain associations that were not there before training. `plot` shows
the same effect more weakly, with `storyline` entering the top five. The contrast predicted
in `PLAN.md` section 1.4, that `plot` would start conspiratorial and become narrative, did
not occur: G's `plot` neighbourhood was morphological to begin with. The drift measurement
carries this section instead, which is why it was worth building.

Next: phase 6, intrinsic evaluation. Neighbour tables for all four models are already being
written to `results/`, so what remains is WordSim-353.

### Sep 22, phase 10, Model C from scratch

Taken out of order, ahead of phases 6 to 9, on request.

`src/sgns.py` implements skip-gram with negative sampling directly: vocabulary and counts,
subsampling, the unigram^0.75 noise table, a dynamic window, two embedding matrices, and
linear learning rate decay. `to_keyedvectors` wraps W_in in a gensim `KeyedVectors` so C
runs through the existing pooling, classifier and neighbour code with no changes.

| | A, gensim sg+NS | C, from scratch |
|---|---|---|
| Vocabulary | 14,309 | 14,309 |
| Wall clock | 11.78 s | 160.15 s |
| Peak RSS | 297.8 MB | 728.2 MB |
| SST-2 test accuracy | 0.761 | 0.744 |
| SST-2 test F1 | 0.776 | 0.761 |

C lands 1.7 points below gensim on identical hyperparameters and an identical corpus, which
is inside the correctness gate the plan set. It is 13.6 times slower, which is the expected
cost of Python and batched dense updates against hand-tuned Cython doing per-pair updates.

**Component checks, run before training anything.** Subsampling keeps 17.6% of occurrences
of `the`. The noise table maps `the` from a raw frequency of 4.30% down to a sampled 1.31%
while lifting `execrable` from effectively zero up to 2e-5. Both are the 0.75 exponent doing
what it is supposed to do, and checking them first is what made the later failure easy to
localise.

**The first run did not learn at all, and the failure was worth keeping.** Loss sat at
exactly 1.3863 for all ten epochs and test accuracy came out 0.4992, which is chance on a
balanced split. 1.3863 is 2*ln(2), the loss of a model whose every logit is zero.

The cause was the loss reduction, not the sampling. Using `BCEWithLogitsLoss` twice, once on
the positives and once on the negatives, averages over every element: the positive term over
B entries and the negative term over B*k. Measured gradient norm on W_in was exactly 0.0 and
on W_out 2.4e-4. At a batch size of 8192 that leaves an effective per-pair learning rate of
about 3e-6, so across 2,480 steps nothing moved.

The fix is to write Mikolov's equation 4 directly, summing over the k negatives and averaging
only over the batch, and to use Adam rather than plain SGD. The original C implementation
updates once per training pair, so its 0.025 is a per-pair step; the batched equivalent for
SGD would be roughly `alpha * batch_size`, which is not stable. Adam at 1e-3 with the
original's linear decay kept on top is what the PyTorch reference the handout links
(`Andras7/word2vec-pytorch`) does. Loss then runs 4.2288 down to 2.8681.

Worth noting against the plan's own prediction: the risk register named the wrong-noise-table
bug, where loss falls smoothly and vectors are garbage. The bug that actually happened was
the opposite and much easier to catch, a loss that did not move at all. Both are caught by
the same discipline of gating on neighbour quality and downstream accuracy rather than on the
loss curve, but the plan guessed the wrong failure.

**MPS is slower than CPU here, measured rather than assumed**, so C trains on CPU:

| Device | Sparse gradients | ms/batch | 10 epochs |
|---|---|---|---|
| CPU | no | 44.6 | 1.8 min |
| CPU | yes | 52.6 | 2.2 min |
| MPS | no | 64.3 | 2.7 min |
| MPS | yes | 169.9 | 7.0 min |

Two results in that table. The GPU loses because the model is tiny: 14,309 by 300 is 4.3M
parameters per matrix, so kernel launch and transfer overhead dominate the arithmetic.
Sparse gradients also lose, and the reason is specific: a batch of 8,192 pairs with 10
negatives touches roughly 90,000 index slots over a 14,309-row vocabulary, so nearly every
row is touched on every step and there is no sparsity left to exploit. Both results should
reverse on text8, where the vocabulary is around 253,000 and a batch touches a small
fraction of it. That is the measurement phase 11 exists to take, and it is a better
motivation for text8 than corpus size alone.

Next: phase 6, WordSim-353, which also completes C's correctness gate.

### Sep 22, phases 6 to 8

`src/evaluate.py` and `src/run_eval.py` augment the existing records rather than rewriting
them, so training numbers captured at training time are never recomputed from a later run.
`src/measure_mem.py` runs one model per subprocess.

#### Phase 6, lexical similarity

Spearman between human ratings and cosine. Coverage is the fraction of pairs where both
words were in vocabulary, and it is part of the result rather than a footnote.

| | WordSim-353 | SimLex-999 | MEN-3k | Coverage |
|---|---|---|---|---|
| A, gensim sg+NS | 0.029 | 0.045 | 0.118 | 50.1% |
| B, gensim CBOW+HS | -0.036 | 0.069 | -0.010 | 50.1% |
| C, from scratch | 0.005 | 0.033 | 0.096 | 50.1% |
| G-ft, fine-tuned | 0.585 | 0.405 | 0.702 | 50.1% |
| G, GoogleNews | **0.700** | **0.442** | **0.771** | 100% |

**This is the strongest result in the assignment so far, and it is a negative one.** A, B
and C have no measurable lexical similarity structure at all. B is slightly negative on two
of the three benchmarks, which is a correlation indistinguishable from none. Yet those same
three models score 0.729 to 0.761 on sentiment classification, within 5 points of
GoogleNews.

The two evaluations are measuring different things and it is now quantified rather than
asserted. Mean-pooled logistic regression on SST needs only that sentiment-bearing words
occupy consistent directions; it does not need `tiger` to be near `cat`. 634k tokens is
enough to supply the first and nowhere near enough for the second. Any report that
evaluated only downstream accuracy would conclude these embeddings are nearly as good as
GoogleNews, and it would be wrong.

**The fine-tune is what carries the semantic structure**, at 0.585 against 0.029 for A
trained from scratch on the identical corpus. The pretrained initialisation is not a 3-point
accuracy trick; it is the entire source of lexical similarity in Model G-ft.

Coverage is 50.1% for every SST-trained model, because the benchmarks contain general
vocabulary that 634k tokens of film review never mentions. Their scores are computed over
half the pairs and must be read that way.

#### Phase 7, systems measurement

| | Vocab | p50 ms | p95 ms | Matrices kept | Predicted MB | Measured delta MB | Ratio |
|---|---|---|---|---|---|---|---|
| A | 14,309 | 0.41 | 2.41 | 2 | 34.3 | 35.0 | 1.02 |
| B | 14,309 | 0.36 | 1.40 | 2 | 34.3 | 48.0 | 1.40 |
| G-ft | 14,309 | 0.41 | 3.07 | 2 | 34.3 | 34.8 | 1.01 |
| C | 14,309 | 0.37 | 1.51 | 1 | 17.2 | 17.8 | 1.03 |
| G | 500,000 | 15.08 | 26.94 | 1 | 600.0 | 195.6 | 0.33 |

**Query latency is linear in vocabulary, as brute-force `most_similar` should be.** G has 35
times the vocabulary of the SST models and is 37 times slower at p50. That is the crossover
argument for an approximate index stated as a measurement rather than as received wisdom: at
14k words nobody needs one, at 500k it is 15 ms per query, and at Google's full 3M vocabulary
it would be roughly 90 ms.

**The memory prediction holds for four of five models within 3%.** B is the exception at
1.40, which is hierarchical softmax: it keeps a `syn1` matrix plus the Huffman tree's code
and point arrays, so `2 * V * d * 4` understates it.

**G does not fit the prediction and the reason is not yet established.** 500,000 by 300
float32 is 600 MB and the measured delta is 195.6 MB, a third of it. The likely explanation
is macOS memory compression, which compresses inactive anonymous pages so `ru_maxrss` reports
a footprint below the allocation. This is testable: the same measurement on an ICE Linux node
should come out near 600 MB. Until that is run, the number is reported as an unexplained
discrepancy rather than as a finding.

The first pass at this table compared a whole-process peak RSS against the matrix prediction
and produced 297.8 MB for a model whose matrices are 34.3 MB. Process peak RSS includes the
interpreter, gensim, numpy, torch and the resident corpus, so it measures the script and not
the model. `src/measure_mem.py` replaced it with a per-subprocess RSS delta.

#### Phase 8, error mining, deliverable 2c

| | Errors / 1,821 | negation | contrast | other |
|---|---|---|---|---|
| G, GoogleNews | 368 | 102 | 64 | 202 |
| G-ft, fine-tuned | 379 | 96 | 64 | 219 |
| A, gensim sg+NS | 435 | 101 | 72 | 262 |
| C, from scratch | 466 | 114 | 79 | 273 |
| B, gensim CBOW+HS | 493 | 120 | 62 | 311 |

Negation and contrast together account for 38 to 45% of every model's errors, against roughly
20% of the test set overall, so both categories are over-represented among failures for every
model including GoogleNews. That is the pooling step, not the embeddings: averaging discards
word order, so no amount of embedding quality recovers a scope-of-negation judgement.

Confident mistakes, which are what deliverable 2c asks for:

| Model | Sentence | Gold | Pred | Conf |
|---|---|---|---|---|
| A | `not a bad journey at all .` | pos | neg | 0.958 |
| A | `a well acted and well intentioned snoozer .` | neg | pos | 0.898 |
| A | `first good , then bothersome .` | neg | pos | 0.896 |
| G-ft | `never -lrb- sinks -rrb- into exploitation .` | pos | neg | 0.990 |
| G-ft | `there is n't a weak or careless performance amongst them .` | pos | neg | 0.973 |
| G-ft | `it 's a great deal of sizzle and very little steak .` | neg | pos | 0.952 |

Every one has a mechanical explanation. `not a bad journey` averages `not` and `bad` into
negative territory. `there is n't a weak or careless performance` pools two strongly negative
adjectives that the negation was supposed to invert. `a great deal of sizzle and very little
steak` is an idiom whose negative reading lives entirely in the contrast between clauses.
`a well acted and well intentioned snoozer` carries three positive words and one negative one,
and the average follows the majority.

The handout warns that a test set producing no failures must be made harder. That did not
arise: SST supplies these unaided.

Next: phase 9, the report.

### Sep 23, deliverable 3 covered properly

Prompted by a TA reply confirming that sentiment analysis is expected when the SST corpus is
used. It does not change the approach, since sentiment was already the spine, but comparing
the handout's wording side by side exposed a coverage gap:

> **3.** top K ranked results for **5 test examples** ... for your chosen application
> (e.g. sentiment analysis) **using your test data**
> **5.** top-1, top-5 and top-10 ranked words for **five different word queries**

Deliverable 5 says word queries and was already covered by the neighbour table. Deliverable 3
says test examples and ties them to the sentiment task, which means five test *sentences*.
Both had been collapsed into the one word-query table.

`src/testcases.py` closes it. Five test sentences are chosen by rule rather than by eye, and
the rule that selected each is recorded in the output so the mix is visible. Each carries two
ranked outputs per model: the classifier's ranked labels with probabilities, and the top-5
nearest training sentences by cosine, with the share of those whose gold label matches.

Scores across the five: A 3/5, B 3/5, G 4/5, G-ft 3/5, C 3/5.

**Example 3 separates the pretrained model from everything trained here.**
`finally , a genre movie that delivers -- in a couple of genres , no less .` is positive.
`no less` is an intensifier rather than a negation, and only GoogleNews reads it that way.
Every SST-trained model including the fine-tune calls it negative.

**Example 5 is the one worth being suspicious of.** `no movement , no yuks , not much of
anything .` is called negative by all five models at high confidence, and retrieval explains
why: the nearest training sentence is `no charm , no laughs , no fun`, a near-duplicate of
the same stacked-negation pattern. The models are right lexically, not because they composed
the negation. It is the same mechanism that makes Example 3 fail, pointing the right way by
luck. Retrieval is what makes that visible; the classifier's confidence alone would have read
as competence.

Also reframed the lexical similarity section, which had been titled as the headline result.
It is analysis beyond the requirement and now says so, so the required sentiment work reads
as primary. No measurement changed.

### Sep 23, phase 9 closed out, plus an audit

Audited the tracked tree against the handout's six deliverables rather than against memory.
Three gaps and one defect.

**Deliverable 2b was missing.** The handout asks for five representative training examples
and four test examples, two positive and two negative, quoted verbatim. None were in the
report. `results/examples_2b.json` now holds them, selected by rule with the rule recorded,
and they render into the data section.

**Deliverable 6 was missing.** The installation and running notes existed only in this log.
They are now a section of the report: the Python 3.14 wheel problem, the
`scipy.linalg.triu` pin, the SST-2 hidden-label split, and the peak-RSS mistake. Each is
recorded with the misleading symptom rather than just the fix, which is what makes the
section worth reading.

**A defect in a field name, which had already produced two wrong sentences.**
`finetune.mean_cosine_drift` held 0.9609, a cosine *similarity*, not a drift. Two files
stated the acceptance gate as "drift stays under ~0.4", which is only true under the
opposite convention, where drift is 1 minus cosine. `PLAN.md` made the same slip in reverse,
writing the threshold as a distance in one place and as a similarity in another.

The field is renamed `mean_cosine_before_after` (and `median_...`), which cannot be read
two ways, and the existing record was migrated rather than retrained since the value is
unchanged. Worth recording as a lesson rather than a tidy-up: a name that admits two
conventions will eventually be read in both, and the wrong reading produces prose that looks
fine.

**Two files were not written by this session.** `README.md` and `CLAUDE.md` were produced by
side agents and swept into commits `5551046` and `340fe91` by `git add -A`. Content was
checked against the results and is accurate apart from the drift statement above, now fixed.
Flagged because authorship matters for a graded artefact.

**Submission built.** `HW2_P_Mohite_Atharva.pdf`, 21 pages, rendered from `docs/index.html`
through headless Chrome with a print stylesheet that forces the light palette and keeps
cards, tables and examples off page boundaries. `HW2_P_Mohite_Atharva.zip` is 3.6 MB and
contains the PDF, `data/`, `src/`, `results/`, `docs/` and the three markdown documents. The
zip is gitignored, being a build artefact.

Still open: the repo has no remote, so every commit is local. Model G's memory discrepancy
is unexplained and needs one run on a Linux node. Phases 11 to 13 are untouched.

### Sep 23, addressing an external review

An independent review flagged six items. All six are now closed. Five were genuine coverage
gaps; one was an analytical error, and fixing it changed a headline number.

**The analytical error is the one worth reading.** The report claimed A versus G-ft isolates
initialisation, since both are skip-gram with negative sampling on the same corpus with the
same classifier. It does not. G-ft also runs at `alpha=0.005` for 8 epochs against A's `0.025`
for 10, so the comparison mixed the warm start with the optimiser schedule.

`A2` is the missing control: A re-run with G-ft's exact schedule and random initialisation, so
the starting point is the only difference. It scores **0.6645** against G-ft's **0.7919**.

The warm start is therefore worth **+0.127**, not the +0.031 that A versus G-ft suggested. The
looser comparison understated the effect by a factor of four, because A's more aggressive
schedule was partly compensating for its random start. The review was right that the claim
was unsound, and the corrected claim is stronger than the one it replaces. A2 is reported in
the results table even though it is the lowest score there, since hiding a control defeats
its purpose.

**Embedding visualization** (`src/visualize.py`). PCA and t-SNE projections rendered as
inline SVG so they inherit the page theme. The plotted words are fixed in advance and belong
to four semantic groups, which lets the figure be judged rather than admired, and each
carries the silhouette score of the 2-D layout against those group labels:

| | PCA silhouette | t-SNE silhouette |
|---|---|---|
| G-ft | +0.164 | +0.187 |
| G | +0.120 | +0.121 |
| A | +0.036 | +0.051 |
| B | +0.035 | -0.033 |
| C | -0.004 | +0.110 |

The ordering matches WordSim-353 and analogy accuracy exactly. Worth noting in the report
because the first two PCA components capture under 20% of variance for every model, so the
scatter is a thin slice and is the easiest figure here to over-read.

**Vector arithmetic and analogies** (`src/analogies.py`). Seven arithmetic queries with full
ranked answers, plus the Google analogy set scored per section with `restrict_vocab=50000`.

| | Analogy accuracy | Coverage | `king - man + woman` |
|---|---|---|---|
| G | 0.7561 | 79.0% | queen, monarch, princess |
| G-ft | 0.6750 | 18.2% | queen, princess, prince |
| C | 0.0022 | 18.2% | middle-aged, lion, creek |
| B | 0.0020 | 18.2% | sugar, abbott, ernest |
| A | 0.0006 | 18.2% | scorpion, lion, lear |

Only G and G-ft solve the canonical analogy, and both put `queen` at rank 1. Coverage matters
more than accuracy: gensim drops any question with an out-of-vocabulary term, so the
SST-trained models are scored on 18.2% of the set against G's 79.0%.

**Top-1, top-5 and top-10 stated explicitly.** The neighbour section showed eight chips per
query while the JSON held ten. It is now one table per query, with the three ranks in
separate columns and cosine printed on every chip.

**Deliverable 2a completed.** A field table giving the model, the NIPS reference with arXiv
link, the distribution URL, the training corpus, the test set, and accuracy. On the published
figure: the Google Code archive page is a JavaScript application that serves nothing to a
fetch, and the linked papers report results for models trained in those papers rather than
for this released file. Rather than cite a number that could not be verified, the table
reports our measured 0.7561 with its exact conditions and says plainly why no published
figure is quoted.

**Deliverable 2c stated explicitly.** Two correct and two incorrect test cases for one named
model, Model G-ft, highest-confidence in each category, rather than left implicit in the error
tables.

The review also noted that the SST-2 classifier is a defensible response to an ambiguous
section but not a substitute for the visualization and analogy requirements. That is correct,
and both are now present as their own sections rather than folded into the sentiment work.

### Sep 24, submission blockers

**Mean query latency.** Deliverable 4 asks for average per-query time; the table had p50 and
p95 only. `evaluate.latency` now records `mean_ms`, and `python -m src.run_eval --latency`
re-times every record, text8 included, without touching any other field. Mean, p50 and p95
come from the same 200 calls, after one untimed warm-up call per query word. Two earlier
attempts without the warm-up disagreed with each other by 2 to 5 times at p95 on the text8
and G rows, on a laptop at load average 9 to 11 (mostly the measurement's own BLAS threads).
Stored run, M2 Air:

| | mean ms | p50 ms | p95 ms |
|---|---|---|---|
| A | 0.69 | 0.38 | 2.29 |
| B | 1.27 | 0.45 | 4.50 |
| C | 0.74 | 0.42 | 2.74 |
| G | 29.90 | 19.23 | 72.32 |
| G-ft | 1.15 | 0.49 | 4.06 |

Latency on this machine is noisy; read the ratio between vocabulary sizes, not the third
digit.

**Every stored result now has a command.** `python -m src.finetune` trains G-ft and then A2.
A dry run into scratch directories reproduced G-ft exactly (accuracy 0.7919, mean cosine
0.9609) and put A2 at 0.656 against the stored 0.665, which is four-worker gensim
nondeterminism. `python -m src.examples` regenerates `examples_2b.json` and
`examples_2c.json` byte-identically: 2b is a fixed index list, 2c is the top two confident
correct positives and the top two confident wrong negatives from G-ft's mined errors.

**Submission ZIP.** `STUDENT_ID=... python -m src.package` rebuilds the PDF and zips it with
`src/`, `results/`, `data/` including text8, `README.md`, `requirements.txt` and the sbatch
script: 60 files, 37.7 MB. It refuses to run without `STUDENT_ID`.

**Cleanup.** Direct `stanfordnlp/sst2` link in the data table and references; GoogleNews
epochs print `not reported`; quoted sentences cut at a word boundary with an ellipsis
(testcases now stores full neighbour text); long tables and test-case cards may split across
print pages, which took the PDF from 72 to 54 pages.
