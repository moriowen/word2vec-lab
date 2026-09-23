# CS 6220 HW2 — Word2Vec

Train word embeddings five different ways, evaluate them intrinsically (nearest neighbours,
WordSim-353/SimLex-999/MEN-3k) and extrinsically (SST-2 sentiment classification), and
compare. Programming option for CS 6220 Big Data Systems. Full reasoning behind every design
decision is in [`PLAN.md`](PLAN.md); the ordered build steps and a running results log are in
[`IMPLEMENTATION.md`](IMPLEMENTATION.md).

## Models

| ID | What it is | Built by |
|---|---|---|
| A | Skip-gram + negative sampling, trained from scratch on SST | `src/train_gensim.py` (gensim) |
| B | CBOW + hierarchical softmax, trained from scratch on SST | `src/train_gensim.py` (gensim) |
| C | GoogleNews-300, pretrained, frozen | `src/pretrained.py` |
| D | GoogleNews-300 warm-started, fine-tuned on SST | `src/finetune.py` |
| E | Skip-gram + negative sampling, implemented from scratch in PyTorch | `src/sgns.py` |

Every model funnels through the same pipeline so comparisons are attributable to the
embeddings alone: `src/data.py` (load SST-2) → model-specific trainer/loader →
`src/pooling.py` (mean-pool word vectors into a sentence vector) → `src/classify.py` (one
shared `LogisticRegression`) → `src/schema.py` (validate + write `results/<id>_<corpus>.json`).

## Layout

```
src/           pipeline code (see AGENTS.md for the module-by-module architecture notes)
data/          SST-2 splits, phrase-level training corpus, WordSim/SimLex/MEN eval sets
models/        trained model artifacts (gitignored, reproducible from src/ + data/)
results/       one JSON per model, the only source for report numbers/figures
report/        write-up deliverables
slurm/         cluster job scripts
PLAN.md        what the assignment asks for and why each design choice was made
IMPLEMENTATION.md   step-by-step build log with acceptance checks and a dated results log
```

## Running it

```sh
source .venv/bin/activate            # Python 3.11 — gensim 4.3.3 has no 3.14 wheels
python -m src.data                   # print corpus stats for both training-corpus variants
python -m src.main                   # train A, B; score C; write results/*.json
python -m src.run_e                  # train E (from-scratch PyTorch skip-gram)
python -m src.finetune               # train D (GoogleNews warm start)
python -m src.run_eval               # phases 6-8: intrinsic benchmarks, latency/memory, error mining
```

There is no test suite. Correctness is judged by the acceptance checks in
`IMPLEMENTATION.md` — e.g. `most_similar("good")` returns sensible neighbours, SST-2 accuracy
lands in a known range, fine-tune mean cosine between warm-start and final vectors stays above ~0.6 — never by the training loss
curve alone (see the gotchas in `AGENTS.md`).

## Headline results (phrase-level corpus, 67k rows)

| | A, sg+NS | B, CBOW+HS | C, GoogleNews | D, fine-tuned | E, sg+NS from scratch |
|---|---|---|---|---|---|
| SST-2 test accuracy | 0.761 | 0.729 | 0.798 | 0.792 | 0.744 |
| SST-2 test F1 | 0.776 | 0.750 | 0.811 | 0.809 | 0.761 |
| Vocabulary | 14,309 | 14,309 | 500,000 | 14,309 | 14,309 |

Full discussion — corpus-size effects, neighbour-quality comparisons, fine-tune drift
analysis, the from-scratch training-loss bug and fix, CPU-vs-MPS timing — is in the dated log
at the bottom of `IMPLEMENTATION.md`.
