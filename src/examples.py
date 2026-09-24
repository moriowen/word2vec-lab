"""Deliverables 2b and 2c: the representative examples quoted in the report.

    python -m src.examples   # after src.finetune and src.run_eval

2b is a hand-picked set, one row per property worth showing, so the choice lives here as
indices into the loaded splits rather than as retyped text. The text and gold label are
read back from the data, so a changed split cannot leave a stale sentence in the report.

2c is chosen by rule from G-ft's mined errors: the two most confident correct positives and
the two most confident wrong negatives.
"""
import json

from . import data, schema

TRAIN_2B = [(2, "positive, plain"), (3, "negative, plain"), (11, "negation"),
            (29, "contrastive clause"), (473, "film-domain vocabulary")]
TEST_2B = [(4, "positive"), (30, "positive"), (8, "negative"), (0, "negative")]


def ex2b(d) -> dict:
    row = lambda split, i, why: {"i": i, "why": why, "text": " ".join(d[split][i][0]),
                                 "label": int(d[split][i][1])}
    return {"train": [row("train", i, why) for i, why in TRAIN_2B],
            "test": [row("test", i, why) for i, why in TEST_2B]}


def ex2c(rec, n=2) -> dict:
    err = rec["errors"]
    pick = lambda rows, gold, ok: [{k: x[k] for k in ("sentence", "gold", "pred", "confidence")}
                                   | {"correct": ok} for x in rows if x["gold"] == gold][:n]
    return {f"model_{rec['model_id']}": pick(err["confident_correct"], 1, True)
            + pick(err["confident_errors"], 0, False)}


def main():
    d = data.load_sst2("phrases")
    gft = json.loads((schema.RESULTS / "G-ft_sst2-phrases.json").read_text())
    for name, out in (("examples_2b", ex2b(d)), ("examples_2c", ex2c(gft))):
        (schema.RESULTS / f"{name}.json").write_text(json.dumps(out, indent=2))
        print(f"wrote results/{name}.json")


if __name__ == "__main__":
    main()
