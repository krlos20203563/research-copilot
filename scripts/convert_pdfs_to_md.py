"""
convert_pdfs_to_md.py
---------------------
Conversión única y reproducible de los PDFs de `papers/` a Markdown en
`papers_md/`, usando pymupdf4llm. Los .md generados son la fuente que lee
el pipeline de ingesta (src/ingestion.py); los PDFs originales se conservan
solo como referencia.

Cada .md conserva el mismo nombre base que su PDF para que el matching
contra papers/papers.json siga funcionando.

Uso:
    python scripts/convert_pdfs_to_md.py            # convierte los faltantes
    python scripts/convert_pdfs_to_md.py --force    # reconvierte todo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import fitz  # PyMuPDF
import pymupdf4llm

# La consola de Windows usa cp1252 por defecto; forzamos UTF-8 en la salida
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "papers"
MD_DIR = ROOT / "papers_md"

# Si el Markdown queda por debajo de esta fracción del texto plano del PDF,
# la conversión probablemente perdió contenido y se marca como sospechosa.
MIN_SIZE_RATIO = 0.3


def plain_text_size(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    n = sum(len(page.get_text("text")) for page in doc)
    doc.close()
    return n


def convert(force: bool = False) -> None:
    MD_DIR.mkdir(exist_ok=True)
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No se encontraron PDFs en {PDF_DIR}")

    converted, skipped, suspicious = 0, 0, []
    for pdf in pdfs:
        out = MD_DIR / (pdf.stem + ".md")
        if out.exists() and not force:
            print(f"  skip  {out.name}")
            skipped += 1
            continue

        md = pymupdf4llm.to_markdown(str(pdf), show_progress=False)
        out.write_text(md, encoding="utf-8")
        converted += 1

        ratio = len(md) / max(plain_text_size(pdf), 1)
        flag = ""
        if ratio < MIN_SIZE_RATIO:
            flag = "  ⚠ sospechosamente corto"
            suspicious.append(out.name)
        print(f"  ok    {out.name}  ({len(md):,} chars, ratio {ratio:.2f}){flag}")

    print(f"\n{len(pdfs)} PDFs — {converted} convertidos, {skipped} omitidos → {MD_DIR}")
    if suspicious:
        print("Revisar manualmente (posible pérdida de contenido):")
        for name in suspicious:
            print(f"  - {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Reconvierte aunque el .md ya exista")
    args = parser.parse_args()
    convert(force=args.force)
