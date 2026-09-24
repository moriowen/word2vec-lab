# AGENTS.md

Working agreement for coding agents in this repo. Every agent reads this file directly.

## What this is

Five Word2Vec models — trained, downloaded, or written from scratch — all scored through
one shared pipeline on SST-2 sentiment, lexical similarity, and what they cost to run. It
began as CS 6220 HW2, the Word2Vec programming option, and still ships as that. It also has
a second audience now, which is the constraint most likely to trip you up; see *Two
audiences, one builder*.

`PLAN.md` holds what the assignment asks for and the reasoning behind each design choice.
`IMPLEMENTATION.md` is the ordered build sequence with acceptance checks per step, and a
dated results log at the bottom. New results go in that log, not copied into this file or
`README.md`.

Deployed from `public/` to Vercel. The remote is `moriowen/word2vec-lab`, private.

## Commands

```sh
source .venv/bin/activate   # Python 3.11 (Homebrew) — gensim 4.3.3 has no 3.14 wheels
python -m src.data          # corpus stats for both training-corpus variants
python -m src.main          # train A, B; score G; write results/*.json
python -m src.run_e         # train C, the from-scratch PyTorch implementation
python -m src.finetune      # train G-ft, warm-started from GoogleNews
python -m src.run_eval      # intrinsic benchmarks, latency and memory, error mining
python -m src.run_analogies # vector arithmetic and the Google analogy benchmark
python -m src.testcases     # five worked test examples across all models
python -m src.visualize     # PCA and t-SNE projections, emitted as inline SVG
python -m src.measure_mem   # per-model RSS, one subprocess per model
python -m src.build_report  # both report variants, from results/*.json
```

Order matters. The `run_*` scripts augment records that `main`, `run_e` and `finetune`
wrote, so those have to exist first.

There is no test suite. Correctness is judged by the acceptance checks in
`IMPLEMENTATION.md`: `most_similar("good")` returns sensible neighbours, SST-2 accuracy
lands in a known range, and the fine-tune's mean cosine between warm-start and final
vectors stays above roughly 0.6.

## Two audiences, one builder

The same measurements get published twice, and the difference is not cosmetic.

`src/build_report.py` emits both. `build("web")` writes `public/index.html`: the experiment
on its own terms, findings first, with no mention of the course, the handout, or any
deliverable. `build("submission")` writes `report/report.html`: the same sections wrapped in the
course's AI-assisted HW template by `src/submission.py`. Each handout deliverable is one
question with Step 1 (the report sections that answer it) and Step 2 (the AI interaction
record: tool, what was asked, a round-by-round table, strategy, critique), then references
and appendices. `python -m src.build_pdf` renders it to `HW2_P_Mohite_Atharva.pdf`. The
Student ID is injected from the `STUDENT_ID` environment variable into a temporary copy only,
and the PDF is gitignored, so the ID never reaches a tracked file.

Course-specific text goes through the `hw()` and `eb()` helpers so it can only reach the
submission variant. Write a bare course reference into the prose and it ships to the public
page. Two consequences before you edit:

- **Section order differs per variant.** Sections build into named buckets and get placed
  by `WEB_GROUPS` or by `QUESTIONS` and `APPENDIX` in `src/submission.py`. An assertion
  fails the build when a section is not placed exactly once in each, so a new section has to
  join `TITLES`, `WEB_GROUPS` and a question or appendix.
- **Submission-only code must not touch the web page.** `src/submission.py` is used only in
  the submission branch, and its CSS is appended only there. After editing either, check
  that `public/index.html` is unchanged.

`.vercelignore` keeps `/report`, `/src`, `/data`, `/results` and the PDFs from being
uploaded at all. Only `public/` is ever deployed.

## Architecture

**Every model funnels through one shared pipeline**, which is the entire experimental
design: `src/data.py` (load) → a model-specific trainer or loader → `src/pooling.py`
(mean-pool word vectors into a sentence vector) → `src/classify.py` (one uniform
`LogisticRegression`, no per-model branching) → `src/schema.py` (validate and write
`results/<model_id>_<corpus>.json`). Never special-case the classifier or the pooling per
model. That invalidates the A-versus-B and C-versus-D comparisons, which are the point.

**The models**, each emitting the same results-JSON shape (`schema.py` `REQUIRED`):

| | What it is | Where |
|---|---|---|
| A | gensim skip-gram + negative sampling, trained here | `train_gensim.py` |
| B | gensim CBOW + hierarchical softmax, trained here | `train_gensim.py` |
| C | skip-gram + NS written from scratch in PyTorch | `sgns.py`, `run_e.py` |
| G | GoogleNews-300, downloaded and frozen | `pretrained.py` |
| G-ft | G warm-started, then trained on SST | `finetune.py` |
| A2 | a control, not a sixth model | `train_gensim.py` |

A, B and C also have text8 runs (`run_text8.py`, `run_e.py --text8`); the report keys them
`A-text8`, `B-text8`, `C-text8`. IDs were renamed on Sep 23 2026 (E became C, C became G,
D became G-ft); commit messages before that use the old letters.

Everything except `sg`, `hs` and `negative` is held identical between A and B on purpose.
A2 re-runs A with G-ft's exact learning-rate schedule and epochs but random initialisation,
which is what separates the warm start from the optimiser settings. A against G-ft confounds
the two. A2 stays out of the headline comparison and belongs only where the warm-start
claim is made.

G loads through `KeyedVectors` with `limit=500_000`; the full file wants about 4 GB of RAM.
It expects the `.bin.gz` under `~/gensim-data/word2vec-google-news-300/`.

G-ft warm-starts `W_in` through `intersect_word2vec_format`, which overwrites only the rows
for words already in our vocabulary, and which loads `W_in` alone. Google never published
their output matrix, so `W_out` starts random and can scramble the imported vectors. That
is why G-ft reports mean and median cosine drift per word, and why the drift is an acceptance
check rather than a curiosity.

**`src/data.py` has two training-corpus modes**, selected by `load_sst2(train_on=...)`:
`"sentences"` (6,920 rows, SetFit) or `"phrases"` (67,349 rows from the GLUE train split's
labelled subtrees, one dropped as a test-set leak, used by default). Dev and test always
come from `SetFit/sst2` and never the GLUE copy, whose test labels are all `-1` because
they are held out for the leaderboard. `load_sst2` asserts both classes appear in the test
labels and raises loudly otherwise. Leave that assertion alone. It is what stops the whole
thing from grading against a placeholder split and reporting a plausible-looking number.

**`src/schema.py` is the results contract.** Its `machine()` label, `m2-air` against an ICE
node, is mandatory on every record: wall-clock across those machines is not comparable and
nothing else in the JSON tells them apart. Any new training or measurement code has to
record it. `peak_rss_mb()` handles `ru_maxrss` being bytes on macOS and kilobytes on Linux.
Hyperparameters are read back off the trained model object (`getattr(model, k, ...)`)
rather than retyped from the config dict, so the report cannot disagree with what actually
ran. Follow that pattern for any new model.

**Reports and figures are generated from `results/*.json`, never hand-typed**, the same
invariant as HW1. If a number appears in the prose, it came from a record at build time.

Artifacts and data are separated deliberately. The SST-2 JSON and parquet under
`data/sst2/` ship with the submission, and `.gitignore` does not exclude them. `models/`,
`data/text8/` and the downloaded vectors are excluded and treated as reproducible from
source and config rather than from git history.

## Gotchas worth knowing before editing

- `scipy` is pinned `<1.13`. gensim 4.3.x imports `scipy.linalg.triu`, which 1.13 removed,
  so an unpinned install dies at `import gensim` with an error that reads like a broken
  gensim rather than a version conflict.
- A and B are persisted with `model.save()` rather than `save_word2vec_format()`, so
  `W_out` and the vocab counts survive for G-ft's fine-tune to use later.
- Negative sampling must draw from the unigram distribution raised to the 0.75 power. A
  raw-unigram bug still produces a smoothly falling loss with garbage vectors, so gate
  correctness on nearest-neighbour and analogy quality, never on the loss curve.
- Mean pooling discards word order, which is why negation and contrast account for 38 to
  45% of every model's errors, GoogleNews included. That belongs to the pooling step, not
  to embedding quality, and no better embedding fixes it.

## Academic-integrity boundary

This is graded coursework, so the workspace rules in `~/dev/acads/AGENTS.md` apply. The one
that matters most here: never write a GTID, a GTID-derived hash, or any personal identifier
into a tracked file, a commit, or a transcript. The submission cover page carries a name
and session by design, and it stays confined to the submission variant.
