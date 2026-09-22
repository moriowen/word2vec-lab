# HW2 implementation plan

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
every model writes this same schema, which is what lets Model E be added in phase 10 by
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

## Phase 4. Model C, the pretrained reference

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

Acceptance: C's accuracy is recorded and its OOV rate is much lower than A's or B's.

Cost: 15 minutes.

---

## Phase 5. Model D, the fine-tune

The recipe is in `PLAN.md` section 1.4 with the gensim calls spelled out. Three things to
get right, in this order of likely failure:

1. `vectors_lockf` must be explicitly set to ones. gensim 4.x does not create it by default
   and `intersect_word2vec_format` will fail or silently freeze everything without it.
2. `alpha` low, around 0.005, and few epochs, around 5 to 8. `W_out` starts random because
   Google never published theirs, so the first updates are spent re-learning a decoder, and
   a normal learning rate scrambles the imported vectors in the first epoch.
3. Measure drift. For every word present in both C and D, compute cosine between its
   before-vector and its after-vector. Report the mean, and the ten words that moved most.

Acceptance, and this is the real gate: `plot` in Model D must sit closer to storyline senses
than it does in Model C, while a word absent from SST is unchanged. If mean drift is very
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
SST-trained models and C.

Cost: 30 minutes.

### Step 6.2. WordSim-353

Download the 353 pairs, compute Spearman between human ratings and cosine similarity, skip
pairs with an OOV word and report how many were skipped. The skip count is part of the
result, since a model that scores well on half the pairs has not scored well.

Acceptance: C lands around 0.6 to 0.7. A and B will be much lower, which is the expected
consequence of a million-token corpus and is worth stating plainly rather than apologising
for.

Cost: 45 minutes.

---

## Phase 7. Measurement, deliverable 4

Training numbers were captured in phase 2. What remains:

- Query latency for `most_similar`, 200 repeats per model, reported at p50 and p95. This
  scales with vocabulary size, so C at 500k words should be visibly slower than A at 15k,
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

## Phase 10. Model E, from scratch

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
hyperparameters, E's neighbours for `king` and its WordSim-353 correlation must be close to
gensim's. Gate on neighbour quality, never on the loss curve, for the reason in 10.3.

Then register E in the results schema and rerun the harness. Because every model emits the
same JSON, the report picks it up with no edits beyond the prose.

Cost: 4 to 6 hours.

## Phase 11. text8

Download, train A, B and E on it, and compare against the SST-trained versions. This is
where the handout's page-1 claim gets tested: skip-gram is supposed to beat CBOW on small
data and lose on large. Report whichever way it comes out.

Run E on an `ice-gpu` A100 allocation per `PLAN.md` Part 5. 17 million tokens with a 253k
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
is exactly what Model C will contrast against in phase 4, so it is useful evidence rather
than a failure. It does mean the phrase-level SST rows should be tried before the report is
written.

Next: phase 4, Model C.

### Sep 22, phrase-level corpus and Model C

Two changes. Embedding and classifier training moved from the 6,920 sentence-level rows to
the 67,348 phrase-level rows, 4.7 times more text. Model C added.

| | A, skip-gram + NS | B, CBOW + HS | C, GoogleNews-300 |
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

**Model C wins by 3.7 points despite a 27% OOV rate.** The composition of that rate is
measured in `data/README.md`: 40% punctuation, most of the rest very frequent function words
that Google dropped when publishing. None of it carries sentiment. A capitalisation fallback
would recover half the misses and was rejected, because it maps `a` to the letter `A` and
`and` to the acronym `AND`.

**Neighbour quality separates the models more sharply than accuracy does.** C returns
inflectional variants (`plot` gives `plots, Plot, plotting, plotline`) and clean synonyms
(`terrible` gives `horrible, horrendous, dreadful, awful`). A and B return near-noise for
the same queries. Two things follow for the report: 634k tokens is still far too little for
good neighbours even though it is enough for competitive classification accuracy, and the
two evaluation modes are measuring genuinely different things. That gap is a result in its
own right and belongs in the deliverable 5 discussion rather than being smoothed over.

Next: phase 5, Model D, the fine-tune. C's `plot` neighbourhood is currently morphological
rather than conspiratorial, which is a weaker starting contrast than the plan assumed, so
the drift measurement matters more than the single-word anecdote.
