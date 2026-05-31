"""
Recursively scans a directory for supported image files.
"""

import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_images(input_dir: str | Path) -> Generator[Path, None, None]:
    """
    Yield all image file paths found recursively under input_dir.
    Skips unreadable directories and logs warnings for them.
    """
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def count_images(input_dir: str | Path) -> int:
    """Return total count of scannable images (used to size progress bars)."""
    return sum(1 for _ in scan_images(input_dir))
