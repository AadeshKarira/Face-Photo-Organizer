"""
photo_organizer — entry point
==============================

Usage examples
--------------
# Basic run (copies images):
python main.py --input ~/Photos --output ~/Sorted

# Stricter identity matching, move files instead of copy:
python main.py --input ~/Photos --output ~/Sorted --threshold 0.55 --move

# Very lenient matching (one person per folder even if slightly different):
python main.py --input ~/Photos --output ~/Sorted --threshold 0.30
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from src.clustering import cluster_faces
from src.face_detector import FaceDetector, FaceResult
from src.organizer import organize_images
from src.reporter import generate_report
from src.scanner import count_images, scan_images


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            # File handler is added once we know the output directory
        ],
    )


def _add_file_handler(output_dir: Path) -> None:
    log_path = output_dir / "photo_organizer.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)
    logging.getLogger(__name__).info("Logging to file: %s", log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo_organizer",
        description="Group photos by face identity into per-person folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="DIR",
        help="Source directory containing photos (scanned recursively).",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="DIR",
        help="Destination directory where Person_N folders will be created.",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.40,
        metavar="FLOAT",
        help=(
            "Cosine similarity threshold for grouping faces as the same person. "
            "Range [0, 1].  Higher = stricter (more folders, purer clusters). "
            "Typical sweet spot: 0.35–0.50."
        ),
    )
    parser.add_argument(
        "--move",
        action="store_true",
        default=False,
        help="Move files instead of copying them (source files are removed).",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        metavar="PX",
        help="Detector input resolution (square).  Larger = better recall, slower.",
    )
    parser.add_argument(
        "--det-thresh",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Minimum face-detection confidence score [0, 1].",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    """Execute the full pipeline.  Returns exit code (0 = success)."""
    log = logging.getLogger(__name__)
    start_time = time.monotonic()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _add_file_handler(output_dir)

    log.info("Input  : %s", input_dir)
    log.info("Output : %s", output_dir)
    log.info("Threshold: %.2f  |  Move: %s", args.threshold, args.move)

    # ------------------------------------------------------------------
    # Phase 1 — scan & count
    # ------------------------------------------------------------------
    log.info("Counting images …")
    try:
        total_images = count_images(input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        log.error("%s", exc)
        return 1

    if total_images == 0:
        log.warning("No supported images found in %s", input_dir)
        return 0

    log.info("Found %d image(s) to process.", total_images)

    # ------------------------------------------------------------------
    # Phase 2 — detect faces & extract embeddings
    # ------------------------------------------------------------------
    detector = FaceDetector(
        det_size=(args.det_size, args.det_size),
        det_thresh=args.det_thresh,
    )

    all_faces: list[FaceResult] = []
    images_with_no_face: list[Path] = []
    total_faces_detected = 0

    with tqdm(total=total_images, desc="Detecting faces", unit="img") as pbar:
        for img_path in scan_images(input_dir):
            faces = detector.extract_faces(img_path)
            if faces:
                all_faces.extend(faces)
                total_faces_detected += len(faces)
            else:
                images_with_no_face.append(img_path)
            pbar.update(1)
            pbar.set_postfix(faces=total_faces_detected, refresh=False)

    log.info(
        "Detection complete: %d faces in %d images (%d images had no detectable face).",
        total_faces_detected,
        total_images - len(images_with_no_face),
        len(images_with_no_face),
    )

    if not all_faces:
        log.warning("No faces detected — nothing to organise.")
        return 0

    # ------------------------------------------------------------------
    # Phase 3 — cluster embeddings
    # ------------------------------------------------------------------
    clustering = cluster_faces(all_faces, similarity_threshold=args.threshold)

    # ------------------------------------------------------------------
    # Phase 4 — organise files
    # ------------------------------------------------------------------
    written = organize_images(clustering, output_dir, move=args.move)

    # ------------------------------------------------------------------
    # Phase 5 — report
    # ------------------------------------------------------------------
    images_per_person: dict[str, int] = {}
    for folder, paths in written.items():
        # Count unique source images (a multi-face image contributes 1)
        unique_sources = len({p.name for p in paths})
        images_per_person[folder] = unique_sources

    noise_image_count = len({f.image_path for f in clustering.noise})

    generate_report(
        total_images=total_images,
        total_faces=total_faces_detected,
        unique_people=len(clustering.clusters),
        images_per_person=images_per_person,
        noise_images=noise_image_count,
        output_dir=output_dir,
        elapsed_seconds=time.monotonic() - start_time,
    )

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if not (0.0 < args.threshold < 1.0):
        print("ERROR: --threshold must be between 0 and 1 (exclusive).", file=sys.stderr)
        sys.exit(1)

    sys.exit(run(args))


if __name__ == "__main__":
    main()
