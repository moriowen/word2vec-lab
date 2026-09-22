"""The results contract. Every model emits this shape so the report never branches per model."""
import json, os, platform, resource, sys, time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

REQUIRED = ("model_id", "name", "corpus", "machine", "hyperparams", "train")


def machine() -> str:
    """Provenance label. An A100 wall-clock is not a laptop wall-clock and nothing else distinguishes them."""
    return os.environ.get("BDS_MACHINE", "m2-air" if platform.system() == "Darwin" else "linux")


def peak_rss_mb() -> float:
    """ru_maxrss is bytes on macOS and kilobytes on Linux."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e6 if sys.platform == "darwin" else raw / 1e3


def validate(rec: dict) -> dict:
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        raise ValueError(f"result record missing {missing}")
    if not rec["train"].get("vocab_size"):
        raise ValueError("train.vocab_size is zero or absent")
    return rec


def write(rec: dict) -> Path:
    validate(rec)
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{rec['model_id']}_{rec['corpus']}.json"
    path.write_text(json.dumps(rec, indent=2))
    return path


def load_all() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]


class timed:
    """with timed() as t: ...   then t.wall_s"""
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.wall_s = time.perf_counter() - self.t0
