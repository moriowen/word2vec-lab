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


ANALOGY_WORDS = ("king queen man woman actor actress good better bad worse worst best "
                 "comedy funny scary horror").split()


def corpus_report(min_count: int = 2) -> dict:
    """How much of the phrase corpus is new text, measured against the sentence release."""
    from collections import Counter
    ph, se = load_sst2("phrases"), load_sst2("sentences")["train"]
    blob = "\n".join(f" {' '.join(t)} " for t, _ in se)
    inside = sum(1 for t, _ in ph["train"] if f" {' '.join(t)} " in blob)
    cp = Counter(w for t, _ in ph["train"] for w in t)
    cs = Counter(w for t, _ in se for w in t)
    vp = {w for w, c in cp.items() if c >= min_count}
    vs = {w for w, c in cs.items() if c >= min_count}
    gained = vp - vs
    splits = stats(ph)
    for s in splits.values():
        s["mean_len"] = round(s["tokens"] / s["rows"], 1)
    from .evaluate import categorise
    test_cats = Counter(categorise(t) for t, _ in ph["test"])
    return {
        "splits": splits,
        "test_categories": dict(test_cats),
        "sentence_train": {**stats({"t": se})["t"], f"vocab_min{min_count}": len(vs)},
        "phrase_train": {f"vocab_min{min_count}": len(vp), "rows_inside_sentences": inside,
                         "rows_inside_frac": round(inside / len(ph["train"]), 4),
                         "token_ratio": round(splits["train"]["tokens"]
                                              / sum(len(t) for t, _ in se), 2)},
        "vocab_gain": {"gained": len(gained),
                       "gained_singletons": sum(1 for w in gained if cs.get(w) == 1)},
        "word_counts": {w: {"sentences": cs.get(w, 0), "phrases": cp.get(w, 0)}
                        for w in ANALOGY_WORDS},
    }


if __name__ == "__main__":
    for mode in ("sentences", "phrases"):
        d = load_sst2(mode)
        print(mode, json.dumps(stats(d)["train"]))
    out = Path(__file__).resolve().parent.parent / "results" / "corpus.json"
    out.write_text(json.dumps(corpus_report(), indent=2))
    print(f"wrote {out.relative_to(out.parent.parent)}")
