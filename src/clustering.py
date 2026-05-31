"""
Clusters face embeddings into identity groups.

Strategy
--------
1. Collect all (image_path, face_index, embedding) triples produced by the
   detector pass.
2. Run DBSCAN on the embedding matrix.  DBSCAN requires no pre-specified
   cluster count and marks genuine outliers as noise (label == -1).
3. Return a mapping  { cluster_id: [FaceResult, …] }  plus the set of
   noise faces that could not be assigned to any person.

Similarity metric
-----------------
Embeddings are L2-normalised, so Euclidean distance == sqrt(2 - 2·cosine_sim).
A cosine threshold of 0.40 (good strangers differ by > 0.5) translates to
an eps of roughly sqrt(2 * (1 - 0.40)) ≈ 1.095.  We expose `--threshold`
as cosine similarity so the user thinks in familiar terms.
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import numpy as np
from sklearn.cluster import DBSCAN

from .face_detector import FaceResult

logger = logging.getLogger(__name__)


class ClusteringResult(NamedTuple):
    # cluster_id (0-based) → list of face results
    clusters: dict[int, list[FaceResult]]
    # faces DBSCAN could not assign to any cluster
    noise: list[FaceResult]


def cluster_faces(
    face_results: list[FaceResult],
    similarity_threshold: float = 0.40,
    min_samples: int = 1,
) -> ClusteringResult:
    """
    Group face embeddings by identity using DBSCAN.

    Parameters
    ----------
    face_results          Flat list of all FaceResult objects from the scan.
    similarity_threshold  Cosine similarity above which two faces are
                          considered the same person.  Range [0, 1].
                          Higher → stricter (fewer, purer clusters).
    min_samples           Minimum faces needed to form a core cluster point.
                          Keep at 1 to avoid dismissing people who appear
                          in only one photo.

    Returns
    -------
    ClusteringResult with a dict mapping integer cluster IDs to face lists,
    plus a separate list of noise/unassigned faces.
    """
    if not face_results:
        logger.warning("No face embeddings to cluster.")
        return ClusteringResult(clusters={}, noise=[])

    # Convert similarity threshold → Euclidean distance eps
    # cos_sim = 1 - (euclidean^2 / 2)  →  euclidean = sqrt(2 * (1 - cos_sim))
    eps = math.sqrt(2.0 * (1.0 - float(similarity_threshold)))
    logger.info(
        "Clustering %d faces  (cosine threshold=%.2f → DBSCAN eps=%.4f)",
        len(face_results),
        similarity_threshold,
        eps,
    )

    embeddings = np.stack([fr.embedding for fr in face_results])  # (N, 512)

    db = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,  # use all CPU cores
    )
    labels: np.ndarray = db.fit_predict(embeddings)

    clusters: dict[int, list[FaceResult]] = {}
    noise: list[FaceResult] = []

    for face, label in zip(face_results, labels):
        if label == -1:
            noise.append(face)
        else:
            clusters.setdefault(int(label), []).append(face)

    # Re-key clusters from 0 to N so they map directly to Person_1, Person_2 …
    sorted_ids = sorted(clusters.keys(), key=lambda k: -len(clusters[k]))
    renumbered: dict[int, list[FaceResult]] = {
        new_id: clusters[old_id]
        for new_id, old_id in enumerate(sorted_ids)
    }

    logger.info(
        "Clustering complete: %d people found, %d unassigned face(s).",
        len(renumbered),
        len(noise),
    )
    return ClusteringResult(clusters=renumbered, noise=noise)
