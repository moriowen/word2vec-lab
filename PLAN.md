# CS 6220 HW2, Word2Vec (programming option)

> **Model IDs were renamed on Sep 23 2026.** This document now uses the new letters: C is the from-scratch model (formerly E), G is GoogleNews-300 (formerly C), and G-ft is the fine-tune (formerly D). Commit messages keep the old letters.

Due Fri Sep 25 (see `../../calendar.md`). Plan drafted Sep 22, revised the same day after a
first pass was judged too thin.

---

## Part 1. What the assignment is actually asking, in plain terms

### 1.1 The one-sentence version

Train some word embeddings, look at what words they think are similar, use them to classify
movie reviews as positive or negative, and write up a comparison of two of them.

That is the whole assignment. Everything below is detail.

### 1.2 What a "Word2Vec model" is here

A Word2Vec model is not a classifier and not a chatbot. It is a lookup table. You feed it a
corpus of raw text with no labels, and it learns one dense vector (say 300 numbers) per word,
arranged so that words appearing in similar contexts land near each other in the space. The
output of training is a file mapping `word -> [0.13, -0.88, ...]`. Nothing else.

Two matrices do the work:

- `W_in`, shape `(vocab_size, dim)`. One row per word. This is the thing you keep. The
  handout calls this the encoder, and the row itself is the "code."
- `W_out`, shape `(vocab_size, dim)`. Used only to compute the training loss, then thrown
  away. The handout calls this the decoder.

The training objective is: given a word, predict the words around it (skip-gram), or given
the words around a slot, predict what fills it (CBOW). Nobody cares about the prediction.
The prediction task is a pretext. The embeddings are the product.

Deliverable 1 wants an architecture figure showing input to encoder to code to decoder to
output. For word2vec that figure is small and honest: a one-hot input of size V, a matrix
multiply into d dimensions, another matrix multiply back out to V, a softmax or a set of
sigmoid outputs. Two matrices and no nonlinearity. Say in the report that the
encoder/decoder framing is borrowed from autoencoders and fits word2vec only loosely, and
that a true autoencoder reconstructs its own input while word2vec reconstructs a
neighbouring word. Naming that mismatch is worth more to the grader than pretending it fits.

### 1.3 Where sentiment analysis comes in

Word2Vec cannot do sentiment analysis. It has no labels and no output class. The standard
setup, which is what the handout is gesturing at, is a two-stage pipeline:

```
sentence  ->  look up a vector per word  ->  average them  ->  logistic regression  ->  pos/neg
             (the word2vec model)           (pooling)         (a tiny classifier)
```

The classifier is deliberately trivial. Logistic regression on a 300-dimensional averaged
vector has about 300 parameters and takes a second to fit. Keeping it identical across all
models is the entire point: if Model A scores 79% and Model B scores 83% with the same
classifier, the same pooling, and the same train/test split, the difference is attributable
to the embeddings and nothing else. That is the experimental design, and it is what makes
deliverable 4's "compare the two models" claim defensible instead of hand-waved.

### 1.4 What "fine-tuning a Word2Vec model" means

This is the part of the handout that reads as confusing, because "fine-tuning" normally
brings to mind BERT or an LLM. For word2vec it means something simpler and older.

You download a model somebody else trained on a huge general corpus. GoogleNews-300 was
trained on roughly 100 billion tokens of news text. Its vector for `plot` reflects news
usage, so its neighbours are things like `scheme` and `conspiracy`. Our corpus is movie
reviews, where `plot` means narrative structure and should sit near `storyline` and
`screenplay`.

Fine-tuning is: initialise `W_in` from the pretrained vectors, then keep running the normal
word2vec training loop on the movie-review corpus at a low learning rate for a few epochs.
The vectors start from general English and drift toward film usage. Words that never appear
in the reviews keep their downloaded vectors untouched. In gensim it is roughly:

```python
model = Word2Vec(vector_size=300, window=5, min_count=2, sg=1, negative=10, alpha=0.005)
model.build_vocab(sst_sentences)
model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)   # 1.0 = trainable
model.wv.intersect_word2vec_format(
    "GoogleNews-vectors-negative300.bin", binary=True, lockf=1.0)   # overwrite rows we have
model.train(sst_sentences, total_examples=model.corpus_count, epochs=8)
```

Two things about this recipe belong in the report because they are real and most write-ups
miss them:

1. `intersect_word2vec_format` only overwrites rows for words that are already in the vocab
   built from our corpus. It does not import the whole 3-million-word Google vocabulary.
   That is the behaviour we want, but it needs stating.
2. It loads `W_in` only. `W_out` stays randomly initialised, because Google never published
   their output matrix. So the first few hundred updates are partly spent re-learning a
   decoder from scratch, and that pushes the pretrained input vectors around more than a
   true fine-tune would. The practical fix is a low `alpha` and few epochs. The honest
   framing for the report is that this is a warm start on the encoder rather than a full
   fine-tune, and the drift is measurable: report mean cosine between each word's vector
   before and after, which shows directly how far each word moved.

That second point is a genuine finding produced by paying attention, and it is the kind of
thing that separates a report from a tutorial transcript.

---

## Part 2. The models

Five models, because the handout's numbered deliverables ask for more than two things even
though bullet 1 says "two." Each row exists to satisfy specific graded items.

| ID | Model | How it is obtained | Covers |
|----|-------|--------------------|--------|
| A | Skip-gram + negative sampling, gensim | trained by me on SST | 1, 2b, 3, 4, 5 |
| B | CBOW + hierarchical softmax, gensim | trained by me on SST | 1, 2b, 3, 4, 5 |
| G | GoogleNews-300 | downloaded, frozen | 2a, 3 |
| G-ft | GoogleNews-300 fine-tuned on SST | warm start from G, per section 1.4 | 2b, 2c, 3 |
| C | Skip-gram + negative sampling written from scratch in PyTorch | trained by me on SST and text8 | the extra work, plus 4 and 5 |

The headline comparison for deliverable 1 is **A versus B**. They differ in architecture
(skip-gram versus CBOW) and in loss (negative sampling versus hierarchical softmax), which
gives deliverable 5 two independent axes to talk about instead of one. Both are trained by
me, which satisfies the handout's "ideally both."

G is the required pretrained reference. GoogleNews rather than GloVe, deliberately: GloVe is
a different algorithm (it factorises a co-occurrence matrix) and is not a Word2Vec model at
all, so using it as the pretrained comparison would quietly answer a different question than
the one asked. GoogleNews-300 is genuine skip-gram with negative sampling. GloVe can appear
as a footnote if time allows, labelled as a different method.

G-ft is what deliverable 2b's "your own fine-tuned model from a pretrained Word2Vec model" is
asking for, and it is also where the most interesting nearest-neighbour shifts will show up.

C is the part that is not required. See Part 4.

---

## Part 3. Data

Everything here ships in `data/` inside the submitted zip, which deliverable 2 asks for
explicitly.

**SST-2** is the sentiment corpus. Two versions exist and picking the wrong one wastes an
afternoon. The GLUE version on HuggingFace (`stanfordnlp/sst2`) has 67,349 train / 872
validation / 1,821 test rows, but **the test labels are all -1 because they are held out for
the leaderboard**. Use `SetFit/sst2`, which carries the real test labels, or the original
treebank release from Stanford, which has its own labelled test split of 2,210 sentences.
Whichever is used, `data/README.md` states which one and why, since an accuracy number means
nothing without knowing which split produced it.

Note that SST-2's training rows are mostly phrases rather than full sentences, because the
treebank labels every subtree. That is fine for the classifier and slightly odd for training
embeddings, since it inflates the count of short fragments. Worth one sentence in the
report.

**text8** is the second corpus: the first 100 MB of cleaned English Wikipedia, about 17
million tokens, from `mattmahoney.net/dc/text8.zip`. SST on its own has roughly a million
tokens, which is enough to train something but not enough for the nearest neighbours to look
convincing, and not enough for any throughput measurement to be meaningful. text8 is the
standard word2vec benchmark corpus and makes both of those work.

The SST-versus-text8 contrast is itself a result. The handout asserts on page 1 that
skip-gram does better on small data and CBOW needs large datasets. We can test that claim
rather than repeat it: train A and B on both corpora and report all four. If the claim holds,
we have evidence. If it does not, that is a more interesting paragraph.

**Evaluation sets**, all small and all downloadable:

- WordSim-353, 353 word pairs with human similarity ratings. Score is Spearman correlation
  between human ratings and cosine similarity.
- SimLex-999, same idea but measures strict similarity rather than relatedness, which
  word2vec is known to do badly on. Including it is a way to show a limitation rather than
  only wins.
- The Google analogy set, 19,544 questions of the `king - man + woman = queen` form, split
  into semantic and syntactic sections. This is where deliverable's analogy bullet gets its
  numbers.

---

## Part 4. What we are doing beyond the requirement, and why

Three additions. Each one is scoped so that dropping it still leaves a complete submission.

### 4.1 Writing skip-gram with negative sampling from scratch (Model C)

The assignment says to take "a superficial look at the code." Writing it instead is the
difference between having used word2vec and understanding it, and it is the piece that makes
this defensible on a resume.

What has to be built, in order, because each step has a specific trap:

1. **Vocabulary and counting.** Build word counts, drop anything below `min_count`, and
   store the frequency table. Straightforward.
2. **Subsampling of frequent words.** Each token is discarded with probability
   `1 - sqrt(t/f) - t/f` where `f` is its corpus frequency and `t` is about `1e-5`. This
   throws away most occurrences of `the` and `of`. It is the single largest quality win in
   the original paper and it also speeds training up several times over, because the
   discarded tokens were contributing almost no signal.
3. **The negative sampling distribution.** Negatives are drawn from the unigram distribution
   raised to the power 0.75, not from the raw unigram distribution. The exponent flattens the
   curve so rare words get sampled more often than their frequency alone would allow. Getting
   this wrong is the classic bug: training loss still goes down smoothly and the neighbours
   are garbage. Implement the draw with a precomputed table so it is O(1) per sample.
4. **Dynamic window.** For each target word, the actual window size is drawn uniformly from
   1 to `window`. This weights nearby words more heavily without any explicit weighting term.
   Easy to miss when reading the paper, since it is one line in the C code.
5. **The model itself.** Two `nn.Embedding` layers, one for `W_in` and one for `W_out`.
   Score is a dot product. Loss is binary cross entropy with the true context word as a
   positive and k sampled words as negatives. No softmax over the vocabulary, which is the
   whole reason negative sampling exists.
6. **Linear learning rate decay** from `alpha` to near zero across training, as in the
   original.

The correctness gate, before any of it goes in the report: on text8, `king` must return
something like `queen, prince, emperor`, and analogy accuracy must land within a few points
of gensim trained on the same corpus with the same hyperparameters. If it does not, the bug
is almost always in step 3, then step 2, in that order. Do not debug by watching the loss
curve, because a wrong noise distribution produces a perfectly healthy-looking loss curve.

### 4.2 Treating it as a systems problem

This is CS 6220, and deliverables 4 and 5 are asking for throughput and cost numbers even if
they do not use those words. The measurements to take:

- Training throughput in words per second, and total wall clock, for every model.
- Peak resident memory against vocabulary size. The two embedding matrices are
  `2 * V * d * 4` bytes, so a 300-dimensional model over a 70,000-word vocabulary is about
  168 MB before optimiser state. Measure it and compare against that prediction. A gap
  between the two is worth explaining.
- Scaling across workers. gensim's `workers` parameter uses Hogwild-style lock-free
  asynchronous SGD, where threads write to the shared embedding matrix without locking and
  the resulting races are tolerated because updates are sparse. Measure words/sec at 1, 2,
  4, 8 and 16 workers and find where it stops scaling. The falloff is usually the Python
  producer thread feeding the C workers, not the workers themselves, which makes it a good
  concrete example of a data pipeline bottleneck.
- Query latency at p50 and p95 for `most_similar`, brute force against the full vocabulary
  versus an approximate index. Brute force is a dense matrix-vector product over V rows, so
  it grows linearly with vocabulary and matters at Google's 3 million words in a way it does
  not at SST's 15,000. Showing the crossover point is the useful part.
- A hyperparameter sweep reported as cost versus benefit: for each of window, dimension,
  negative count, and epochs, plot training time against downstream SST accuracy. The
  engineering question is not "which is most accurate," it is "what does the last two points
  of accuracy cost," and almost no student report answers that.

### 4.3 A reproducible harness instead of a notebook

One command runs everything and writes `results/*.json`. Figures and tables in the report
are generated from that JSON, never typed by hand. This is the same invariant HW1 used, and
for the same reason: a hand-copied results table silently goes stale the moment a run is
repeated. HW1's report pipeline already resolves placeholders out of run data at build time,
and that machinery can be lifted over.

---

## Part 5. Using PACE ICE

Same cluster and same account as HW1. The runbook, including the four things that went
wrong the first time, is at `../HW1/docs/PACE-ICE.md`. What differs this time:

**No SSH tunnel.** HW1 needed one because a server (ollama) had to be reachable from the
laptop during interactive runs. HW2 is batch training. Submit with `sbatch`, let it write
checkpoints and JSON to scratch, `rsync` the results back. Simpler and it survives a dropped
connection without any special handling.

**The A100 only helps one of the five models.** Worth being clear about, since requesting a
GPU for work that cannot use it is its own kind of mistake:

- gensim (models A, B, G, G-ft) is Cython and CPU-only. A GPU does nothing for it. What ICE
  does give these models is core count and RAM: 16 or 32 cores make the `workers` scaling
  study possible in a way an M2 Air does not, and loading GoogleNews-300 needs roughly 4 GB
  of RAM before anything else happens. Run these on an `ice-cpu` allocation.
- The from-scratch PyTorch model (C) is where the A100 earns its place. Large batches of
  skip-gram pairs over text8's 17 million tokens with 253,000 vocabulary entries is a real
  GPU workload. On the laptop this is an overnight job at best; on an A100 with batches in
  the tens of thousands it is minutes per epoch. This is what makes training C on the full
  text8 corpus feasible inside the schedule at all, rather than on a truncated slice.

Allocations, adapted from the HW1 runbook:

```sh
# CPU work: gensim sweeps, the workers scaling study
salloc -p ice-cpu --cpus-per-task=16 --mem=64G --time=04:00:00

# GPU work: the from-scratch SGNS on text8
salloc -p ice-gpu --gres=gpu:a100:1 --cpus-per-task=8 --mem=64G --time=04:00:00
```

Carry over from HW1: connect GlobalProtect first or the hostnames do not resolve, the login
node is `login-ice.pace.gatech.edu`, never run work on the login node, and the compute node
name changes with every allocation so nothing may hardcode it.

One thing to add that HW1 established the need for: record which machine produced each
timing. HW1 used a `BDS_BACKEND` environment variable written into every trace header,
because wall-clock from an A100 is not comparable to wall-clock from a fanless laptop and
nothing in the output distinguishes them otherwise. Same rule here. Any timing in the report
carries the machine it came from, and no table mixes the two.

---

## Part 6. Repository layout

```
HW2/
  README.md               how to run it
  PLAN.md                 this file
  data/                   ships inside the submitted zip (deliverable 2)
    README.md             source URLs, sizes, token counts, split sizes, licences
    sst2/
    text8/
    eval/                 wordsim353, simlex999, google analogies
  src/
    corpus.py             ingest, tokenise, vocab, subsample, negative-sampling table
    train_gensim.py       models A, B, and the fine-tune G-ft
    sgns.py               model C, from scratch
    pooling.py            sentence vectors: mean, and tf-idf weighted
    evaluate.py           intrinsic + extrinsic + systems benchmarks -> results/*.json
    figures.py            every plot and table, generated from the JSON
  results/                JSON and SVG, committed
  report/
    report.html           source of truth, same pipeline as HW1
    HW2_P_Mohite_Atharva.pdf
  slurm/                  sbatch scripts for the ICE runs
```

Git: start `moriowen/6220-bds` (private) with `HW2/` inside it, matching how `dva/` and
`infosec/` are organised. HW1 is a standalone per-assignment repo and predates that
convention, so leave it where it is. HW3 through HW5 are coming and each wants a folder, not
a repo.

---

## Part 7. Schedule

Four working sessions, Monday evening through Thursday, with Friday as buffer only.

**Monday evening, tonight, about 3 hours. Get an ugly version of everything working.**
Repo skeleton, virtual environment, download SST-2, train a gensim skip-gram model on it,
average the vectors, fit logistic regression, print one accuracy number. End to end and
unpolished. The point is that the assignment is then already complete in outline on night
one, and every remaining session improves something that works rather than racing to finish
something broken.

**Tuesday, about 5 hours. Models A through G-ft, plus the from-scratch implementation.**
Morning: the gensim four, including the fine-tune, with the before-and-after cosine drift
measurement from section 1.4. Afternoon: write `sgns.py` and get it through the correctness
gate on SST. Start the text8 run on an ICE GPU allocation before stopping for the day so it
trains overnight.

**Wednesday, about 5 hours. Measurement.**
The full evaluation harness. The workers scaling study on an ICE CPU node. The
hyperparameter sweep, kept to six or eight configurations rather than a grid. Error mining
for deliverable 2c. Every number lands in `results/*.json` and every figure is generated.
Nothing is typed into the report by hand.

**Thursday, about 4 hours. Write and submit.**
Both architecture figures, drawn as SVG by hand. Then write against the checklist in Part 8
literally, ticking rows. Build the PDF, assemble the zip with `data/` inside it, submit.
HW1 closed at 23:55 rather than 23:59, and HW2's grace window is unconfirmed, so treat the
deadline as hard and aim to be done Thursday night.

### Finding the negative examples deliverable 2c requires

Worth calling out separately because the handout is emphatic about it: if all the test
results come out positive, more difficult examples must be added until there are real
failures to report. This will not be a problem, and the reason is predictable in advance.
Averaging word vectors discards word order, so anything whose sentiment depends on structure
breaks:

- Negation. "not bad" averages the vector for `not` with the vector for `bad` and lands in
  negative territory, while the sentence means mildly positive.
- Contrast. "great cast, terrible script" averages to roughly neutral, and the label depends
  on which clause the annotator weighted.
- Sarcasm and faint praise. "a triumph of production design over storytelling."

The method is to sort test sentences by classifier confidence, pull the most confident
mistakes, and check them against those three categories. Confident errors make a better
table than random ones, because each comes with an explanation grounded in the pooling step
rather than a shrug.

---

## Part 8. Deliverable checklist

Written from the handout's six numbered items so the report can be checked against it row by
row on Thursday.

| # | Requirement | Where it is satisfied |
|---|-------------|----------------------|
| 1 | Two Word2Vec models compared, architecture figure each, input through code to output, at least one trained by me | A versus B, both trained by me. Two hand-drawn SVGs. Note on the autoencoder framing per section 1.2. |
| 2 | Training and test datasets described and shipped in a subdirectory | `data/` with `data/README.md` giving URLs, token counts, split sizes |
| 2a | Pretrained model: reference, URL, reported accuracy, dataset links | Model G, GoogleNews-300 |
| 2b | Fine-tuned model: 5 representative training examples, 4 test examples, 2 positive and 2 negative | Model G-ft, examples quoted verbatim from SST |
| 2c | Performance table with at least 2 correct and 2 incorrect predictions | error mining, per the section above |
| 3 | Top-K results for 5 test examples, compared against a pretrained model | nearest-neighbour tables plus the sentiment comparison against G |
| 4 | Performance measurement for training and for testing, average per query | the systems benchmarks in section 4.2 |
| 5 | Hyperparameter analysis; top-1, top-5, top-10 for five word queries across both models | sweep table plus the query table |
| 6 | Notes on installation, running, and measurement | written as work happens, not reconstructed at the end |

### The five word queries for deliverable 5

Chosen so the four models disagree in interpretable ways rather than all returning the same
list:

| Query | Why this word |
|-------|---------------|
| `plot` | polysemous. SST-trained models should give narrative senses, GoogleNews should give conspiracy senses. The clearest single demonstration of what training corpus does. |
| `terrible` | sentiment-bearing. Word2vec famously places antonyms close together because they share contexts, so `terrible` returning `wonderful` is expected and is a limitation worth reporting. |
| `star` | polysemous across domains: celebrity in SST, astronomy in GoogleNews. |
| `cinematography` | domain-specific and moderately rare. Tests whether the small SST corpus has enough occurrences to place it sensibly. |
| `the` | very frequent. With subsampling on it should have a weak, near-arbitrary neighbourhood, which demonstrates what subsampling does. |

---

## Part 9. Cut lines

Named now, in order, so that a bad Tuesday is a scope decision rather than a panic.

1. t-SNE. Use PCA only. t-SNE is slow, its output is sensitive to perplexity, and the plot
   is decorative either way.
2. SimLex-999 and the Google analogy set. Keep WordSim-353 only.
3. The workers scaling study. Report single-worker throughput instead. Cut this before
   cutting the from-scratch model: it is the most impressive piece but the least required.
4. text8. Run everything on SST only. The pipeline still works end to end and the findings
   get thinner.
5. Model C on text8, falling back to Model C on SST only.

Never cut: the from-scratch model, since it is the reason this is worth doing. The error
table, since 2c is explicitly graded. The shipped `data/` directory, since 2 is explicitly
graded.

---

## Part 10. Risks

**A quiet bug in the from-scratch model.** The failure mode is specific: a wrong negative
sampling distribution still gives a smoothly decreasing loss and useless vectors. Gate on
nearest-neighbour quality and on analogy accuracy against gensim, never on the loss curve.

**Fine-tuning that destroys the pretrained vectors instead of adapting them.** Because
`W_out` starts random (section 1.4), too high a learning rate will scramble the imported
vectors in the first epoch and Model G-ft will score worse than Model G. Guard against it by
measuring mean cosine drift per word, and by checking a handful of words that do not appear
in SST are still intact. If drift is large, drop `alpha` and epochs.

**Picking the SST split with hidden labels.** The GLUE version's test labels are all -1.
Check the label distribution immediately after loading and fail loudly if it is not
balanced.

**GoogleNews-300 is 3.4 GB and wants about 4 GB of RAM to load.** On the M2 Air with 8 GB
this competes with everything else. Load it on ICE, or use `limit=500000` to load only the
most frequent half-million words, which is enough for everything here and cuts the footprint
by more than five times. If the limit is used, say so in the report, since it changes the
brute-force query latency numbers.

**Thursday and Friday are loaded.** Sep 25 also carries 6242 project teams and 7641 A1 and
Q4. Thursday is report-only by design, and nothing that can slip is scheduled there.
