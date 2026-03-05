import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "preprocessing")

IMG = "submissions/case17/New Employment Local 1_Page_1.jpg"

# --- Tesseract ---
t0 = time.perf_counter()
import pytesseract
from PIL import Image
text_tess = pytesseract.image_to_string(Image.open(IMG))
t_tess = time.perf_counter() - t0

# --- Docling ---
t1 = time.perf_counter()
from docling.document_converter import DocumentConverter
result = DocumentConverter().convert(IMG)
text_docling = result.document.export_to_markdown()
t_docling = time.perf_counter() - t1

out = "\n".join([
    "=== OCR QUALITY COMPARISON (scanned JPG) ===",
    f"Tesseract : {t_tess:.2f}s | {len(text_tess):,} chars | {len(text_tess.splitlines())} lines",
    f"Docling   : {t_docling:.2f}s | {len(text_docling):,} chars | {len(text_docling.splitlines())} lines",
    "",
    "--- Tesseract output (first 600 chars) ---",
    text_tess[:600],
    "",
    "--- Docling output (first 600 chars) ---",
    text_docling[:600],
])

with open("output/ocr_compare.txt", "w", encoding="utf-8") as f:
    f.write(out)

print(out)
