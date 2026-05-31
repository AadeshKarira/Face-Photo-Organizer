"""
Copies or moves source images into per-person output folders.

An image that contains N detected faces is placed in N folders (one per
person).  When `move=True` and the same image belongs to multiple people,
it is copied to all folders except the last, then moved to the last one —
this avoids leaving the original in place while still not duplicating the
file unnecessarily (the copies go to the extra folders).

Folder layout
-------------
<output>/
    Person_1/
        photo_001.jpg
        photo_002.jpg
    Person_2/
        …
    Unknown/          ← faces that DBSCAN marked as noise
        …
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .clustering import ClusteringResult
from .face_detector import FaceResult

logger = logging.getLogger(__name__)

# Folder name used for DBSCAN noise faces (unassigned to any cluster)
UNKNOWN_FOLDER = "Unknown"


def organize_images(
    clustering: ClusteringResult,
    output_dir: str | Path,
    move: bool = False,
) -> dict[str, list[Path]]:
    """
    Create output folders and populate them.

    Parameters
    ----------
    clustering   Result from cluster_faces().
    output_dir   Root output directory (will be created if absent).
    move         If True, move files; otherwise copy.  Images that belong to
                 multiple clusters are always copied to the extra destinations
                 and only moved (or left in place on copy-mode) for the last.

    Returns
    -------
    Mapping of folder name → list of destination paths written.
    """
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Build a reverse map: image_path → set of person folder names
    # ---------------------------------------------------------------
    image_to_folders: dict[Path, set[str]] = defaultdict(set)

    for cluster_id, faces in clustering.clusters.items():
        folder_name = f"Person_{cluster_id + 1}"
        for face in faces:
            image_to_folders[face.image_path].add(folder_name)

    for face in clustering.noise:
        image_to_folders[face.image_path].add(UNKNOWN_FOLDER)

    # ---------------------------------------------------------------
    # Create all needed output folders up-front
    # ---------------------------------------------------------------
    all_folders = {f for folders in image_to_folders.values() for f in folders}
    for folder in all_folders:
        (out_root / folder).mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Copy / move each image into its destination(s)
    # ---------------------------------------------------------------
    written: dict[str, list[Path]] = defaultdict(list)
    total = len(image_to_folders)

    with tqdm(total=total, desc="Organising images", unit="img") as pbar:
        for src_path, folders in image_to_folders.items():
            folder_list = sorted(folders)

            for i, folder in enumerate(folder_list):
                dst_path = _unique_dest(out_root / folder, src_path.name)
                is_last = i == len(folder_list) - 1

                try:
                    if move and is_last:
                        shutil.move(str(src_path), str(dst_path))
                    else:
                        shutil.copy2(src_path, dst_path)
                    written[folder].append(dst_path)
                except Exception as exc:
                    logger.error(
                        "Failed to %s %s → %s: %s",
                        "move" if (move and is_last) else "copy",
                        src_path,
                        dst_path,
                        exc,
                    )

            pbar.update(1)

    return dict(written)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_dest(folder: Path, filename: str) -> Path:
    """
    Return a destination path that does not collide with existing files.
    Appends _2, _3, … before the suffix when needed.
    """
    candidate = folder / filename
    if not candidate.exists():
        return candidate

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 2
    while True:
        candidate = folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1
