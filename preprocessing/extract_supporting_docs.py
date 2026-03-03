"""
extract_supporting_docs.py
--------------------------
Backend-only supporting document extraction pipeline.

Reads documents from an input folder (simulating DB-stored supporting docs),
extracts structured content, and writes per-document outputs:

- metadata.json   : extraction metadata and status
- content.txt     : plain text output (for downstream classifier)
- content.md      : markdown output (Docling when available)
- layout.json     : layout clusters (Docling when available)

Supported inputs:
- PDF, JPG, JPEG, PNG, BMP, TIFF, TIF, WEBP, DOCX

Usage:
  python preprocessing/extract_supporting_docs.py \
    --input-folder submissions \
    --output-folder output/supporting_docs
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Reuse existing text extraction fallback in this repository
from extract_text import extract_text

try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions, TableFormerMode
except Exception:
    DocumentConverter = None
    PdfFormatOption = None
    InputFormat = None
    PdfPipelineOptions = None
    RapidOcrOptions = None
    TableFormerMode = None


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".docx",
}


@dataclass
class ExtractionResult:
    input_file: str
    status: str
    used_docling: bool
    used_fallback_text_extractor: bool
    pages: int | None
    markdown_length: int
    text_length: int
    layout_items: int
    error: str | None = None


def _sanitize_name(file_path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in file_path.stem)


def _collect_files(folder: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [p for p in folder.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    return sorted(files)


def _build_docling_converter(enable_ocr: bool, enable_tables: bool, mode: str) -> Any:
    if DocumentConverter is None:
        return None

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = enable_tables
    if enable_tables:
        pipeline_options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if mode == "Accurate" else TableFormerMode.FAST
        )

    pipeline_options.do_ocr = enable_ocr
    if enable_ocr:
        pipeline_options.ocr_options = RapidOcrOptions(
            backend="onnxruntime",
            force_full_page_ocr=True,
        )

    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.do_picture_classification = True

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            InputFormat.IMAGE: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _extract_with_docling(converter: Any, file_path: Path) -> tuple[str, str, list[dict[str, Any]], int | None]:
    result = converter.convert(str(file_path))
    markdown = result.document.export_to_markdown()

    layout_info: list[dict[str, Any]] = []
    for page_no, page in enumerate(result.pages, 1):
        if page.predictions.layout:
            for cluster in page.predictions.layout.clusters:
                layout_info.append(
                    {
                        "page": page_no,
                        "label": cluster.label,
                        "bbox": [cluster.bbox.l, cluster.bbox.t, cluster.bbox.r, cluster.bbox.b],
                        "confidence": getattr(cluster, "confidence", None),
                    }
                )

    plain_text = markdown.replace("#", " ").strip()
    page_count = len(result.document.pages) if getattr(result.document, "pages", None) else None
    return markdown, plain_text, layout_info, page_count


def _write_outputs(output_root: Path, source_file: Path, markdown: str, text: str, layout: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    doc_dir = output_root / _sanitize_name(source_file)
    doc_dir.mkdir(parents=True, exist_ok=True)

    (doc_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (doc_dir / "content.txt").write_text(text or "", encoding="utf-8")
    (doc_dir / "content.md").write_text(markdown or "", encoding="utf-8")
    (doc_dir / "layout.json").write_text(json.dumps(layout or [], indent=2), encoding="utf-8")


def run_batch_extraction(
    input_folder: Path,
    output_folder: Path,
    recursive: bool,
    enable_ocr: bool,
    enable_tables: bool,
    mode: str,
) -> list[ExtractionResult]:
    files = _collect_files(input_folder, recursive=recursive)
    if not files:
        return []

    converter = _build_docling_converter(enable_ocr=enable_ocr, enable_tables=enable_tables, mode=mode)
    results: list[ExtractionResult] = []

    for file_path in files:
        used_docling = False
        used_fallback = False
        markdown = ""
        text = ""
        layout: list[dict[str, Any]] = []
        pages = None

        try:
            if converter is not None and file_path.suffix.lower() != ".docx":
                markdown, text, layout, pages = _extract_with_docling(converter, file_path)
                used_docling = True
            else:
                text = extract_text(str(file_path))
                markdown = text
                used_fallback = True

            record = ExtractionResult(
                input_file=str(file_path),
                status="success",
                used_docling=used_docling,
                used_fallback_text_extractor=used_fallback,
                pages=pages,
                markdown_length=len(markdown),
                text_length=len(text),
                layout_items=len(layout),
            )

        except Exception as exc:
            # Last-chance fallback to existing extractor if Docling path fails
            try:
                text = extract_text(str(file_path))
                markdown = text
                used_fallback = True
                record = ExtractionResult(
                    input_file=str(file_path),
                    status="success_with_fallback",
                    used_docling=used_docling,
                    used_fallback_text_extractor=used_fallback,
                    pages=pages,
                    markdown_length=len(markdown),
                    text_length=len(text),
                    layout_items=0,
                    error=str(exc),
                )
            except Exception as fallback_exc:
                record = ExtractionResult(
                    input_file=str(file_path),
                    status="failed",
                    used_docling=used_docling,
                    used_fallback_text_extractor=False,
                    pages=pages,
                    markdown_length=0,
                    text_length=0,
                    layout_items=0,
                    error=f"Docling error: {exc}; Fallback error: {fallback_exc}",
                )

        metadata = asdict(record)
        _write_outputs(
            output_root=output_folder,
            source_file=file_path,
            markdown=markdown,
            text=text,
            layout=layout,
            metadata=metadata,
        )
        results.append(record)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch extract supporting documents from folder input.")
    parser.add_argument("--input-folder", required=True, help="Folder containing supporting documents")
    parser.add_argument("--output-folder", default="output/supporting_docs", help="Folder to write extracted outputs")
    parser.add_argument("--recursive", action="store_true", help="Recursively read subfolders")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR in Docling pipeline")
    parser.add_argument("--no-tables", action="store_true", help="Disable table extraction in Docling pipeline")
    parser.add_argument("--mode", choices=["Fast", "Accurate"], default="Fast", help="Docling table mode")
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.exists() or not input_folder.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {input_folder}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    results = run_batch_extraction(
        input_folder=input_folder,
        output_folder=output_folder,
        recursive=args.recursive,
        enable_ocr=not args.no_ocr,
        enable_tables=not args.no_tables,
        mode=args.mode,
    )

    summary = {
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "total_files": len(results),
        "success": sum(1 for r in results if r.status == "success"),
        "success_with_fallback": sum(1 for r in results if r.status == "success_with_fallback"),
        "failed": sum(1 for r in results if r.status == "failed"),
        "results": [asdict(r) for r in results],
    }

    (output_folder / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Supporting Document Extraction Summary ===")
    print(f"Input folder : {summary['input_folder']}")
    print(f"Output folder: {summary['output_folder']}")
    print(f"Total files  : {summary['total_files']}")
    print(f"Success      : {summary['success']}")
    print(f"Fallback     : {summary['success_with_fallback']}")
    print(f"Failed       : {summary['failed']}")
    print(f"Summary JSON : {output_folder / '_summary.json'}")


if __name__ == "__main__":
    main()
