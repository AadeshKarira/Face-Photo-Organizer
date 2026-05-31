"""
Generates and prints the post-run summary report.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_report(
    total_images: int,
    total_faces: int,
    unique_people: int,
    images_per_person: dict[str, int],
    noise_images: int,
    output_dir: Path,
    elapsed_seconds: float,
) -> dict:
    """
    Build a summary dict, print it to stdout, and save it as JSON.

    Returns the summary dict so callers can inspect / test it.
    """
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(elapsed_seconds, 1),
        "total_images_scanned": total_images,
        "total_faces_detected": total_faces,
        "unique_people_found": unique_people,
        "unassigned_images": noise_images,
        "images_per_person": images_per_person,
    }

    _print_report(report)
    _save_report(report, output_dir)
    return report


def _print_report(report: dict) -> None:
    line = "─" * 52
    print(f"\n{line}")
    print("  PHOTO ORGANIZER  —  SUMMARY REPORT")
    print(line)
    print(f"  Run completed at : {report['generated_at']}")
    print(f"  Elapsed time     : {report['elapsed_seconds']}s")
    print(f"  Images scanned   : {report['total_images_scanned']}")
    print(f"  Faces detected   : {report['total_faces_detected']}")
    print(f"  Unique people    : {report['unique_people_found']}")
    print(f"  Unassigned imgs  : {report['unassigned_images']}")

    if report["images_per_person"]:
        print(f"\n  {'Person':<20} {'Images':>6}")
        print(f"  {'------':<20} {'------':>6}")
        for person, count in sorted(
            report["images_per_person"].items(),
            key=lambda kv: -kv[1],
        ):
            print(f"  {person:<20} {count:>6}")

    print(line)


def _save_report(report: dict, output_dir: Path) -> None:
    report_path = output_dir / "report.json"
    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n  Full report saved to: {report_path}\n")
    except Exception as exc:
        logger.warning("Could not save report to %s: %s", report_path, exc)
