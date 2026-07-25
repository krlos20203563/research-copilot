"""
ingestion.py
------------
Reads the Markdown versions of the papers (papers_md/, generated once from
the PDFs via scripts/convert_pdfs_to_md.py) and combines them with metadata
from papers.json. Running this module directly loads and summarizes all papers.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Default paths (relative to project root)
ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "papers"
PAPERS_MD_DIR = ROOT / "papers_md"
PAPERS_JSON = PAPERS_DIR / "papers.json"


def load_metadata(json_path: Path = PAPERS_JSON) -> dict[str, dict]:
    """Return a dict keyed by filename → metadata record."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["filename"]: p for p in data["papers"]}


def read_markdown(md_path: Path) -> str:
    """Read the full text of a Markdown paper."""
    return md_path.read_text(encoding="utf-8")


def _strip_ext(fname: str) -> str:
    """Remove a trailing .pdf or .md extension (case-insensitive)."""
    lower = fname.lower()
    for ext in (".pdf", ".md"):
        if lower.endswith(ext):
            return fname[: -len(ext)]
    return fname


def _find_metadata(fname: str, metadata: dict[str, dict]) -> Optional[dict]:
    """
    Find metadata for a filename using exact match first, then fuzzy prefix match.

    Handles two common mismatches:
      1. JSON filenames are truncated (e.g. 'Author - 2020 - Title....pdf')
         while actual files have the full name.
      2. Actual files are .md while JSON keys still end in .pdf.
    """
    # 1. Exact match
    if fname in metadata:
        return metadata[fname]

    # 2. Compare without extensions (.md files vs .pdf keys)
    fname_noext = _strip_ext(fname)

    for key, meta in metadata.items():
        key_noext = _strip_ext(key)
        # Remove trailing ellipsis from JSON key if present
        key_noext = key_noext.rstrip(".")

        # Match if either name starts with a common prefix (first 30 chars)
        prefix_len = min(30, len(fname_noext), len(key_noext))
        if prefix_len >= 15 and fname_noext[:prefix_len] == key_noext[:prefix_len]:
            return meta

    return None


def _find_markdown_files(papers_md_dir: Path = PAPERS_MD_DIR) -> list[Path]:
    """Return all Markdown papers in papers_md_dir, sorted by name."""
    if not papers_md_dir.is_dir():
        raise FileNotFoundError(
            f"{papers_md_dir} no existe. Genera los Markdown con: "
            "python scripts/convert_pdfs_to_md.py"
        )
    return sorted(papers_md_dir.glob("*.md"))


def load_papers(
    papers_dir: Path = PAPERS_MD_DIR,
    json_path: Path = PAPERS_JSON,
    verbose: bool = True,
) -> list[dict]:
    """
    Load all papers, reading their Markdown text and merging metadata.

    Returns a list of dicts, each with:
        id, title, authors, year, venue, doi, topics, abstract,
        filename, text, num_chars
    """
    metadata = load_metadata(json_path)
    results = []

    paper_files = _find_markdown_files(papers_dir)
    if verbose:
        logger.info(f"Found {len(paper_files)} paper files in {papers_dir}")

    for md_path in paper_files:
        fname = md_path.name
        meta = _find_metadata(fname, metadata)

        if meta is None:
            logger.warning(f"No metadata found for: {fname} — skipping")
            continue

        try:
            text = read_markdown(md_path)
        except Exception as exc:
            logger.error(f"Failed to read {fname}: {exc}")
            continue

        record = {
            **meta,
            "filename": fname,
            "text": text,
            "num_chars": len(text),
            "filepath": str(md_path),
        }
        results.append(record)

        if verbose:
            logger.info(
                f"  [{meta['id']}] {meta['title'][:60]} — {len(text):,} chars"
            )

    if verbose:
        logger.info(f"Loaded {len(results)} papers successfully.")
    return results


def get_paper_by_id(paper_id: str, papers: list[dict]) -> Optional[dict]:
    """Return the paper dict matching the given id, or None."""
    for p in papers:
        if p.get("id") == paper_id:
            return p
    return None


# ---------------------------------------------------------------------------
# CLI entry-point: python -m src.ingestion
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    papers = load_papers(verbose=True)
    print(f"\nSummary: {len(papers)} papers loaded.")
    total_chars = sum(p["num_chars"] for p in papers)
    print(f"Total text: {total_chars:,} characters across all papers.")
