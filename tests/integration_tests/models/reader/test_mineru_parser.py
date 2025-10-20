import os
from pathlib import Path

from graphgen.models.reader.pdf_reader import MinerUParser


def test_check_bin():
    """Ensure mineru CLI is available."""
    MinerUParser()


def test_parse_pdf():
    """Parse a real PDF and verify basic structure."""
    repo_root = Path(__file__).resolve().parents[4]

    sample_pdf = os.path.join(repo_root, "resources", "input_examples", "pdf_demo.pdf")
    parser = MinerUParser()
    blocks = parser.parse_pdf(sample_pdf, device="cpu", method="auto")

    assert isinstance(blocks, list)
    assert blocks, "At least one block expected"

    text_blocks = [b for b in blocks if b.get("type") == "text"]
    assert text_blocks, "No text block found"

    first = text_blocks[0]
    assert "text" in first
    assert isinstance(first["content"], str)
    assert first["content"].strip(), "Empty text content"


def test_empty_pdf(tmp_path: Path) -> None:
    """Gracefully handle blank PDF."""
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"%PDF-1.4\n%%EOF\n")  # syntactically valid, no content

    parser = MinerUParser()
    blocks = parser.parse_pdf(empty, device="cpu")

    # Empty list or list with empty text block are both acceptable
    assert isinstance(blocks, list)
