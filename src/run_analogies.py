"""Score vector arithmetic and the analogy benchmark for every model."""
import json

from . import analogies, schema
from .run_eval import load_all_kv


def main():
    out = {}
    for mid, kv, stem in load_all_kv():
        bench = analogies.benchmark(kv)
        out[mid] = {"arithmetic": analogies.arithmetic(kv), "benchmark": bench}
        c = f'{bench["coverage"]:.1%}' if bench else "n/a"
        a = f'{bench["overall_accuracy"]:.4f}' if bench else "n/a"
        print(f"{mid}: analogy accuracy {a} over {c} of the set "
              f"({bench['attempted']:,} of {bench['questions_in_set']:,} questions)")
        ok = [x for x in out[mid]["arithmetic"] if x["status"] == "ok"]
        print(f"    arithmetic answerable: {len(ok)}/{len(out[mid]['arithmetic'])}")
        for x in ok[:2]:
            print(f"      {x['query']:28s} -> {', '.join(w for w, _ in x['answers'][:3])}")
    (schema.RESULTS / "analogies.json").write_text(json.dumps(out, indent=2))
    print("wrote results/analogies.json")


if __name__ == "__main__":
    main()
