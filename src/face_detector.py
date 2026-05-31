"""
Face detection and embedding extraction using InsightFace.

Each call to `extract_faces` returns a list of FaceResult objects, one per
detected face in the image.  Embeddings are L2-normalised 512-d float32
vectors suitable for cosine/Euclidean comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """A single detected face within one image."""

    image_path: Path
    face_index: int          # 0-based index among faces in this image
    embedding: np.ndarray    # normalised 512-d float32 vector
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) in pixels
    det_score: float         # detection confidence [0, 1]


class FaceDetector:
    """
    Wraps an InsightFace ArcFace model for detection + embedding.

    The model is loaded once and reused across all images.  We use the
    buffalo_l pack which ships both a RetinaFace detector and an ArcFace
    recognition head.
    """

    def __init__(self, det_size: tuple[int, int] = (640, 640), det_thresh: float = 0.5):
        """
        Parameters
        ----------
        det_size    Resolution at which the detector runs.  Larger → better
                    recall on small faces, slower.  (640, 640) is a good default.
        det_thresh  Minimum face-detection confidence score to keep.
        """
        import insightface
        from insightface.app import FaceAnalysis

        self._det_thresh = det_thresh
        logger.info("Loading InsightFace model (buffalo_l) …")
        self._app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],  # swap to CUDAExecutionProvider if GPU available
        )
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)
        logger.info("InsightFace model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_faces(self, image_path: Path) -> list[FaceResult]:
        """
        Detect all faces in *image_path* and return their embeddings.

        Returns an empty list when:
        - the file cannot be decoded as an image
        - no faces meet the detection threshold

        Never raises — errors are logged and an empty list is returned so
        that a single bad file does not abort the whole run.
        """
        try:
            img = self._load_image(image_path)
        except Exception as exc:
            logger.warning("Could not load image %s: %s", image_path, exc)
            return []

        try:
            faces = self._app.get(img)
        except Exception as exc:
            logger.warning("InsightFace failed on %s: %s", image_path, exc)
            return []

        results: list[FaceResult] = []
        for idx, face in enumerate(faces):
            if face.embedding is None:
                continue
            emb = _l2_normalize(face.embedding)
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            results.append(
                FaceResult(
                    image_path=image_path,
                    face_index=idx,
                    embedding=emb,
                    bbox=(x1, y1, x2, y2),
                    det_score=float(face.det_score),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        """Read image from disk; raises on failure."""
        img = cv2.imread(str(path))
        if img is None:
            # cv2 returns None instead of raising for unrecognised formats
            raise ValueError("cv2.imread returned None")
        # InsightFace expects BGR (OpenCV default) — no conversion needed.
        return img


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return (v / norm).astype(np.float32)
