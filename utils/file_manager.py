from __future__ import annotations

from datetime import datetime
from pathlib import Path


OUTPUT_DIR = Path("outputs")


def ensure_output_dir(output_dir: Path = OUTPUT_DIR) -> Path:
    """Create and return the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_output_path(source_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Build an automatic output path using the source stem and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_stem = source_path.stem.replace(" ", "_")
    return ensure_output_dir(output_dir) / f"{safe_stem}_{timestamp}.txt"


def save_transcript(text: str, source_path: Path, output_dir: Path = OUTPUT_DIR) -> Path:
    """Persist transcript text and return the generated output path."""
    output_path = build_output_path(source_path, output_dir)
    output_path.write_text(text, encoding="utf-8")
    return output_path
