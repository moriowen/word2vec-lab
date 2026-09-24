"""Build the submission ZIP: report PDF, source, both training corpora, test data, results.

    STUDENT_ID=... python -m src.package

Rebuilds the PDF first, from the same environment variable, so the ZIP can never carry a PDF
with a blank ID line. The handout asks for the training and test data in a subdirectory, so
text8 goes in as well; it is fetched if missing. The GoogleNews training corpus was never
released and the vectors are 1.6 GB, so G ships as a URL in the report instead. Trained
models are left out: they are rebuilt by the commands in README.md.
"""
import os
import sys
import zipfile
from pathlib import Path

from . import build_pdf
from .run_text8 import fetch

ROOT = Path(__file__).resolve().parent.parent
NAME = "HW2_P_Mohite_Atharva"
ZIP = ROOT / f"{NAME}.zip"


def members():
    yield build_pdf.PDF
    yield from (ROOT / f for f in ("README.md", "requirements.txt", "slurm/c_text8.sbatch",
                                   "data/README.md"))
    for pattern in ("src/*.py", "results/*.json", "data/sst2/*", "data/eval/*"):
        yield from sorted(ROOT.glob(pattern))
    yield fetch()


def main():
    if not os.environ.get("STUDENT_ID", "").strip():
        sys.exit("STUDENT_ID is not set; refusing to package a PDF with a blank ID line")
    build_pdf.main()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in members():
            z.write(f, f"{NAME}/{f.relative_to(ROOT)}")
    with zipfile.ZipFile(ZIP) as z:
        n = len(z.namelist())
    print(f"wrote {ZIP.name}  {n} files  {ZIP.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
