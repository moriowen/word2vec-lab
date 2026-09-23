"""Vector arithmetic and analogy evaluation.

The handout asks to "experiment with vector math analogies (e.g., King - Man + Woman =
Queen)". Two things here: the standard Google analogy benchmark scored per section, and a
small set of arithmetic queries reported with their full ranked answers.

Note on coverage: the benchmark's vocabulary is general English (capitals, currencies,
family terms), and the SST-trained models have a 14k vocabulary drawn from film reviews.
Most questions are simply unanswerable for them. Coverage is reported alongside accuracy,
because an accuracy computed over 2% of the questions is not comparable to one computed
over 99%.
"""
import json
from pathlib import Path

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval"
QUESTIONS = EVAL / "questions-words.txt"
RESTRICT = 50_000

# General analogies, the classic ones the handout names.
GENERAL = [
    (["king", "woman"], ["man"], "king - man + woman"),
    (["paris", "italy"], ["france"], "paris - france + italy"),
    (["bigger", "small"], ["big"], "bigger - big + small"),
    (["walking", "swim"], ["walk"], "walking - walk + swim"),
]
# Film-domain analogies, which the SST-trained models have a chance at.
DOMAIN = [
    (["comedy", "scary"], ["funny"], "comedy - funny + scary"),
    (["actor", "woman"], ["man"], "actor - man + woman"),
    (["worst", "good"], ["bad"], "worst - bad + good"),
    (["better", "bad"], ["good"], "better - good + bad"),
]


def arithmetic(kv, topn=5):
    """Ranked answers for each arithmetic query, or a note on which term was missing."""
    out = []
    for pos, neg, label in GENERAL + DOMAIN:
        missing = [w for w in pos + neg if w not in kv]
        if missing:
            out.append({"query": label, "status": "out of vocabulary",
                        "missing": missing, "answers": []})
            continue
        res = kv.most_similar(positive=pos, negative=neg, topn=topn)
        out.append({"query": label, "status": "ok", "missing": [],
                    "answers": [[w, round(float(s), 3)] for w, s in res]})
    return out


def benchmark(kv):
    """Google analogy set, scored per section by gensim's own implementation."""
    if not QUESTIONS.exists():
        return None
    # restrict_vocab caps the candidate answer set to the N most frequent words. The original
    # word2vec paper uses 30k for the same reason: without a cap, scoring one question against
    # GoogleNews means a 500k-row matrix-vector product, and the full set takes minutes.
    score, sections = kv.evaluate_word_analogies(
        str(QUESTIONS), case_insensitive=True, restrict_vocab=RESTRICT)
    per = {}
    for s in sections:
        n = len(s["correct"]) + len(s["incorrect"])
        if n and s["section"] != "Total accuracy":
            per[s["section"]] = {"accuracy": round(len(s["correct"]) / n, 4), "attempted": n}
    total = next(s for s in sections if s["section"] == "Total accuracy")
    attempted = len(total["correct"]) + len(total["incorrect"])
    # gensim silently drops any question with an out-of-vocabulary term, so attempted is the
    # honest denominator and the raw question count is the honest reference point.
    asked = sum(1 for l in QUESTIONS.read_text().splitlines() if l and not l.startswith(":"))
    return {"overall_accuracy": round(float(score), 4), "attempted": attempted,
            "restrict_vocab": RESTRICT,
            "questions_in_set": asked, "coverage": round(attempted / asked, 4),
            "sections": per}
