"""SST-2 loading. SetFit/sst2 rather than the GLUE copy, whose test labels are all -1.

Two training corpora are available and they differ by an order of magnitude:

  sentences  6,920 rows / 134k tokens, the sentence-level release (SetFit train.jsonl)
  phrases   67,349 rows / 634k tokens, the phrase-level release (GLUE train split)

The phrase rows are labelled subtrees of the same treebank, so they are train-split only and
carry real labels. They serve both as embedding training text and as classifier training
data. `sentences` is kept because the first run used it and the comparison is reported.
"""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "sst2"


def _read_jsonl(split: str) -> list[tuple[list[str], int]]:
    rows = []
    for line in (DATA / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append((obj["text"].lower().split(), int(obj["label"])))
    return rows


def _read_phrases(exclude: set[str]) -> list[tuple[list[str], int]]:
    import pandas as pd
    df = pd.read_parquet(DATA / "train_phrases.parquet")
    rows = []
    for text, lab in zip(df.sentence, df.label):
        t = text.strip().lower()
        if t in exclude:   # one incidental collision with the test split; see data/README.md
            continue
        rows.append((t.split(), int(lab)))
    return rows


def load_sst2(train_on: str = "phrases") -> dict[str, list]:
    """SST ships pre-tokenised, so lowercase plus whitespace split is the honest choice."""
    d = {"dev": _read_jsonl("dev"), "test": _read_jsonl("test")}
    labels = {lab for _, lab in d["test"]}
    # Guard against the GLUE split, where every test label is -1.
    if labels != {0, 1}:
        raise ValueError(f"test labels are {labels}, expected both 0 and 1 -- wrong SST variant?")

    if train_on == "phrases":
        held = {" ".join(t) for t, _ in d["test"]} | {" ".join(t) for t, _ in d["dev"]}
        d["train"] = _read_phrases(held)
    elif train_on == "sentences":
        d["train"] = _read_jsonl("train")
    else:
        raise ValueError(train_on)
    return d


def sentences(rows) -> list[list[str]]:
    return [toks for toks, _ in rows]


def stats(d: dict) -> dict:
    out = {}
    for split, rows in d.items():
        toks = sum(len(t) for t, _ in rows)
        pos = sum(lab for _, lab in rows)
        out[split] = {"rows": len(rows), "tokens": toks, "positive": pos,
                      "positive_frac": round(pos / len(rows), 3)}
    return out


if __name__ == "__main__":
    for mode in ("sentences", "phrases"):
        d = load_sst2(mode)
        print(mode, json.dumps(stats(d)["train"]))
