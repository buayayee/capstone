import csv, re
from pathlib import Path

SUPPORTED_IMAGE_GLOBS = ["*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.tif","*.webp"]

def get_files(case_dir):
    files = sorted(case_dir.glob("*.pdf"))
    for g in SUPPORTED_IMAGE_GLOBS:
        files += sorted(case_dir.glob(g))
    return [f.name for f in files]

rows = list(csv.DictReader(open("output/results.csv", encoding="utf-8")))
for row in rows:
    case_dir = Path("submissions") / row["case"]
    if case_dir.is_dir():
        fnames = get_files(case_dir)
        row["filename"] = ", ".join(fnames) if fnames else row["case"]

with open("output/results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["case","overall_verdict","filename","doc_verdict","reasons","notes"])
    w.writeheader()
    w.writerows(rows)

print(f"Updated {len(rows)} rows. Sample:")
for r in rows[:3]:
    print(" ", r["case"], "->", r["filename"])
