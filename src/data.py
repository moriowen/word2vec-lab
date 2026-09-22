"""SST-2 loading. SetFit/sst2 rather than the GLUE copy, whose test labels are all -1."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "sst2"


def _read(split: str) -> list[tuple[list[str], int]]:
    rows = []
    for line in (DATA / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append((obj["text"].lower().split(), int(obj["label"])))
    return rows


def load_sst2() -> dict[str, list]:
    """SST is already tokenised, so lowercase plus whitespace split is the honest choice."""
    d = {s: _read(s) for s in ("train", "dev", "test")}
    labels = {lab for _, lab in d["test"]}
    # Guard against the GLUE split, where every test label is -1.
    if labels != {0, 1}:
        raise ValueError(f"test labels are {labels}, expected both 0 and 1 -- wrong SST variant?")
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
    d = load_sst2()
    print(json.dumps(stats(d), indent=2))
    vocab = {w for toks, _ in d["train"] for w in toks}
    print("train vocab types:", len(vocab))
