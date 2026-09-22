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

## Phrase-level training corpus

Source: the GLUE copy `stanfordnlp/sst2`, file `data/train-00000-of-00001.parquet`,
67,349 rows. Saved as `data/sst2/train_phrases.parquet`.

These are the labelled subtrees of the same treebank sentences, so they are train-split only
and carry real labels. They are used for both embedding training and classifier training,
while dev and test stay on the SetFit sentence-level splits. Only the train split of the
GLUE copy is used, so its hidden test labels never come into it.

| | Rows | Tokens | Mean length | Positive fraction |
|---|---|---|---|---|
| sentence-level train | 6,920 | 133,555 | 19.3 | 0.522 |
| phrase-level train | 67,348 | 633,720 | 9.4 | 0.558 |

Checked for leakage against the evaluation splits by exact string match: zero overlap with
dev, and one row overlapping test, the fragment `what 's next ?`, which appears both as a
training subtree and as a short test sentence. `src/data.py` drops it, which is where the
67,349 becomes 67,348.

**Size caveat on the sentence-level release.** This is the sentence-level release: 6,920 full sentences,
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

## Out-of-vocabulary behaviour of GoogleNews-300

Model C shows a 27.1% token OOV rate on the SST test split, against 6.6% for the
SST-trained models. Measured composition of those 9,487 OOV tokens:

| Cause | Share |
|---|---|
| Punctuation tokens (`.`, `,`, `...`, `--`, `-lrb-`, `-rrb-`) | 39.8% |
| Very frequent function words absent from the published vocabulary (`a`, `and`, `of`, `to`, `'s`) | most of the remainder |

So the high rate is not a preprocessing fault. Google dropped the highest-frequency function
words when the vectors were published, and SST's punctuation is tokenised into separate
tokens that no word2vec vocabulary would carry. What is missing carries no sentiment, which
is why Model C still scores highest despite the rate.

A capitalisation fallback would nominally recover 50.5% of the OOV tokens, and it was
rejected: `a` would resolve to the vector for the letter `A`, `and` to the acronym `AND`.
That trades a harmless miss for a wrong vector. Lookup stays exact and lowercase.
