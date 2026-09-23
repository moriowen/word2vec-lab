# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CS 6220 HW2 (Word2Vec, programming option). `PLAN.md` explains what the assignment asks
for and the reasoning behind every design decision — read it before touching anything here.
`IMPLEMENTATION.md` is the ordered build sequence with acceptance checks per step, plus a
running log of results at the bottom (append to the log there, don't duplicate numbers into
this file).

## Commands

```sh
source .venv/bin/activate          # Python 3.11 (Homebrew) — gensim 4.3.3 has no 3.14 wheels
python -m src.main                 # trains A, B; scores C; writes results/*.json
python -m src.data                 # print corpus stats for both training-corpus variants
```

There is no test suite. Correctness is judged by the acceptance checks in `IMPLEMENTATION.md`
(e.g. `model.wv.most_similar("good")` returns sensible neighbours, SST-2 accuracy lands in a
specific range, mean cosine between the fine-tune's warm-start and final vectors is above ~0.6) — not by unit tests.

## Architecture

**Every model funnels through one shared pipeline**, which is the entire experimental
design: `src/data.py` (load) → a model-specific trainer/loader → `src/pooling.py`
(mean-pool word vectors into a sentence vector) → `src/classify.py` (one uniform
`LogisticRegression`, no per-model branching) → `src/schema.py` (validate + write
`results/<model_id>_<corpus>.json`). Never special-case the classifier or pooling per model —
doing so invalidates the A-vs-B (and C-vs-D) comparisons that are the point of the assignment.

**Models**, each producing the same results-JSON shape (`schema.py` `REQUIRED` fields):
- `src/train_gensim.py` — A (skip-gram + negative sampling) and B (CBOW + hierarchical
  softmax). `CONFIGS` holds both; everything except `sg`/`hs`/`negative` is held identical
  between them on purpose.
- `src/pretrained.py` — C, GoogleNews-300, loaded frozen via `KeyedVectors` with
  `limit=500_000` (full file wants ~4 GB RAM). Expects the `.bin.gz` under
  `~/gensim-data/word2vec-google-news-300/`.
- `src/finetune.py` — D, warm-starts `W_in` from C via `intersect_word2vec_format` then
  keeps training on SST at low `alpha`. Reports mean/median cosine drift per word since
  `W_out` starts random and can otherwise scramble the imported vectors.

**`src/data.py` has two training-corpus modes**, selected by `load_sst2(train_on=...)`:
`"sentences"` (6,920 rows, SetFit) or `"phrases"` (67,348 rows, GLUE train split's labelled
subtrees — used by default). Dev/test always come from `SetFit/sst2`, never the GLUE copy,
because GLUE's test labels are all `-1` (held out for the leaderboard) — `load_sst2` asserts
both classes appear in the test labels and raises loudly if not. Don't swap this assertion
out; it's the guard against silently grading on a placeholder split.

**`src/schema.py` is the results contract.** Its `machine()` label (`m2-air` vs an ICE node)
is mandatory on every record because wall-clock across those machines isn't comparable and
nothing else in the JSON distinguishes them — any new training/measurement code must record
it. `peak_rss_mb()` handles the fact that `ru_maxrss` is bytes on macOS and KB on Linux.
Hyperparameters in each result record are read back off the trained model object
(`getattr(model, k, ...)`), never retyped from the config dict, so the report can't disagree
with what actually ran — follow that pattern for any new model.

**Reports and figures are generated from `results/*.json`, never hand-typed** (same
invariant as HW1). `src/build_report.py` emits two variants of the same measurements:
`build("web")` writes `public/index.html` (the deployed page, findings-first order, no
mention of the course, deliverable or handout) and `build("submission")` writes
`report/report.html` (cover page, deliverable labels and order, course footer — this is
what the PDF renders from). Anything course-specific goes through the `hw()`/`eb()`
helpers so it can only reach the submission variant; `python -m src.build_report` builds
both. Git commit subjects name deliverables, so the web build omits the commit list.

Committed model artifacts (`models/*.model`) and data are separated: SST-2
JSON/parquet under `data/sst2/` ships with the submission (`.gitignore` does not exclude it);
`models/`, `data/text8/`, and downloaded vectors are excluded and treated as reproducible from
source/config, not from git history.

## Repo-specific gotchas worth knowing before editing

- `scipy` is pinned `<1.13` — gensim 4.3.x imports `scipy.linalg.triu`, removed in 1.13, so an
  unpinned install fails at `import gensim` with an error that looks like a broken gensim.
- `model.save()` is used over `save_word2vec_format()` for A/B so `W_out` and vocab counts
  survive for D's fine-tune to use later.
- Negative sampling draws must use the unigram distribution to the 0.75 power. A raw-unigram
  bug still produces a smoothly decreasing loss with garbage neighbours — gate correctness on
  nearest-neighbour/analogy quality, never on the loss curve.
