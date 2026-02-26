"""
extract_text.py
---------------
Extracts raw text from either a PDF or a Word (.docx) document.
Scanned PDFs (no text layer) fall back to Tesseract OCR automatically.

Cross-platform: works on Windows and macOS.
Tesseract and Poppler paths are detected automatically per OS.

Usage:
    python extract_text.py --input path/to/document.pdf
    python extract_text.py --input path/to/instructions.docx
"""

import argparse
import os
import platform
import re

from PIL import Image
import pdfplumber
import pytesseract
from docx import Document as DocxDocument
from pdf2image import convert_from_path


# ── Platform-aware external tool paths ───────────────────────────────────────

def _configure_tesseract():
    """
    Set the Tesseract executable path based on the current OS.
    Install instructions:
      macOS  : brew install tesseract
      Windows: https://github.com/UB-Mannheim/tesseract/wiki
    """
    system = platform.system()
    if system == "Windows":
        candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(candidate):
            pytesseract.pytesseract.tesseract_cmd = candidate
    elif system == "Darwin":  # macOS
        # Homebrew on Apple Silicon vs Intel
        for candidate in ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]:
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break
    # On Linux, tesseract is typically on PATH already — no override needed


def _get_poppler_path() -> str | None:
    """
    Return the Poppler bin path for pdf2image on Windows.
    On macOS/Linux, Poppler (installed via brew/apt) is on PATH — return None.
    Install instructions:
      macOS  : brew install poppler
      Windows: https://github.com/oschwartz10612/poppler-windows/releases
               Extract and add the bin/ folder to your system PATH,
               OR place it at C:/poppler/Library/bin
    """
    if platform.system() == "Windows":
        candidates = [
            r"C:\poppler\Library\bin",
            r"C:\Program Files\poppler\Library\bin",
            r"C:\tools\poppler\Library\bin",
        ]
        for path in candidates:
            if os.path.isdir(path):
                return path
        # If not found in known locations, assume it's on PATH
        return None
    return None  # macOS/Linux: rely on PATH


# Configure Tesseract at import time
_configure_tesseract()


# ── Text helpers ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-printable characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    return text.strip()


# ── Extractors ───────────────────────────────────────────────────────────────

def extract_from_docx(filepath: str) -> str:
    """Extract all paragraph text from a Word (.docx) document."""
    doc = DocxDocument(filepath)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also extract text from tables inside the document
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())
    return clean_text("\n".join(paragraphs))


def extract_from_digital_pdf(filepath: str) -> str:
    """Extract text from a digital (text-layer) PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)


def extract_from_scanned_pdf(filepath: str) -> str:
    """
    Fallback OCR path for scanned PDFs (no embedded text layer).
    Converts each page to an image via Poppler, then runs Tesseract OCR.
    Requires Poppler installed:
      Windows : https://github.com/oschwartz10612/poppler-windows/releases
      macOS   : brew install poppler
    """
    print(f"  [OCR Fallback] Converting pages to images: {filepath}")
    poppler_path = _get_poppler_path()
    kwargs = {"dpi": 300}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    try:
        pages = convert_from_path(filepath, **kwargs)
    except Exception:
        raise RuntimeError(
            "Poppler is not installed or not on PATH.\n"
            "  Windows : download from https://github.com/oschwartz10612/poppler-windows/releases\n"
            "            extract the zip, then add the 'Library\\bin' folder to your system PATH.\n"
            "  macOS   : brew install poppler\n"
            "  Note    : Digital (non-scanned) PDFs work fine without Poppler."
        )
    text = ""
    for i, page_img in enumerate(pages):
        page_text = pytesseract.image_to_string(page_img)
        text += page_text + "\n"
        print(f"  [OCR] Page {i+1}/{len(pages)} done.")
    return clean_text(text)


# ── Public interface ──────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def extract_from_image(filepath: str) -> str:
    """
    Run Tesseract OCR directly on an image file (JPG, PNG, BMP, TIFF, etc.).
    Requires Tesseract installed:
      Windows : https://github.com/UB-Mannheim/tesseract/wiki
      macOS   : brew install tesseract
    """
    print(f"  [OCR Image] Running OCR on: {filepath}")
    img = Image.open(filepath)
    text = pytesseract.image_to_string(img)
    return clean_text(text)


def extract_text(filepath: str) -> str:
    """
    Extract text from a PDF (.pdf), Word document (.docx), or image file
    (.jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp).
    PDF files automatically fall back to OCR if no text layer is detected.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".docx":
        return extract_from_docx(filepath)
    elif ext == ".pdf":
        text = extract_from_digital_pdf(filepath)
        if len(text.split()) < 20:  # scanned PDF — use OCR
            text = extract_from_scanned_pdf(filepath)
        return text
    elif ext in IMAGE_EXTENSIONS:
        return extract_from_image(filepath)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: .pdf, .docx, .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF or Word document.")
    parser.add_argument("--input", required=True, help="Path to the .pdf or .docx file")
    args = parser.parse_args()

    result = extract_text(args.input)
    print("\n=== Extracted Text ===")
    print(result)
