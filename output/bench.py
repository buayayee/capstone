import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "preprocessing")

PDF = "directives/Main_MP-D 401-01-2005A Deferment Disruption and Exemption Policy for NSmen_050620.pdf"

# --- pdfplumber ---
t0 = time.perf_counter()
import pdfplumber
with pdfplumber.open(PDF) as pdf:
    text_plumber = "\n".join(p.extract_text() or "" for p in pdf.pages)
t_plumber = time.perf_counter() - t0

# --- docling ---
t1 = time.perf_counter()
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert(PDF)
text_docling = result.document.export_to_markdown()
t_docling = time.perf_counter() - t1

out = "\n".join([
    "=== BENCHMARK RESULTS ===",
    f"pdfplumber : {t_plumber:.2f}s | {len(text_plumber):,} chars | {len(text_plumber.splitlines())} lines",
    f"docling    : {t_docling:.2f}s | {len(text_docling):,} chars | {len(text_docling.splitlines())} lines",
    f"Speed diff : docling is {t_docling/t_plumber:.1f}x slower",
    "",
    "--- pdfplumber preview ---",
    text_plumber[:400],
    "",
    "--- docling preview ---",
    text_docling[:400],
])

with open("output/benchmark.txt", "w", encoding="utf-8") as f:
    f.write(out)

print(out)
