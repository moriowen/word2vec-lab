# Datasets

## SST-2 (Stanford Sentiment Treebank, binary)

Source: `SetFit/sst2` on the HuggingFace hub,
`https://huggingface.co/datasets/SetFit/sst2`, files `train.jsonl`, `dev.jsonl`,
`test.jsonl`, downloaded Sep 22 2026. Underlying corpus is Socher et al. (2013),
`https://nlp.stanford.edu/sentiment/`.

**Why this variant.** The GLUE copy (`stanfordnlp/sst2`) sets every test label to -1,
because that split is held out for the leaderboard. `SetFit/sst2` carries real test labels,
so accuracy on it is a number rather than a placeholder. `src/data.py` asserts the test
labels contain both classes and raises if they do not, which fails loudly if the wrong
variant is ever substituted.

| Split | Sentences | Tokens | Positive | Positive fraction |
|---|---|---|---|---|
| train | 6,920 | 133,555 | 3,610 | 0.522 |
| dev | 872 | 17,046 | 444 | 0.509 |
| test | 1,821 | 35,023 | 909 | 0.499 |

All three splits are balanced, so accuracy is interpretable against a 50% baseline.

**Size caveat, and it matters.** This is the sentence-level release: 6,920 full sentences,
about 134,000 tokens. The phrase-level release expands every labelled subtree of the
treebank into its own row and reaches roughly 67,000 rows, which is an order of magnitude
more text to train embeddings on. 134k tokens is very little for word2vec, and the measured
consequences are visible in `results/`: vocabulary is only 7,140 types at `min_count=2`, and
the nearest neighbours for the five report queries are weak. Moving embedding training to
the phrase-level rows while keeping classification on these labelled splits is the obvious
next improvement.

Tokenisation is lowercase plus whitespace split. SST ships pre-tokenised, with punctuation
already separated, so anything more elaborate would need justifying rather than assuming.

## text8

Not yet downloaded. Planned for phase 11, from `http://mattmahoney.net/dc/text8.zip`:
the first 100 MB of cleaned English Wikipedia, about 17 million tokens. Excluded from git
via `.gitignore` and reproducible from the URL.
