"""Lowe-ratio RootSIFT matching + homography RANSAC, limited to Top-N."""
import time
import cv2
import numpy as np
from .local import rootsift

def geometric_inliers(query, candidate, ratio=.75, threshold=4.0, seed=2026):
    """Verify a candidate with RootSIFT matching and homography RANSAC.
    query and candidate are (xy float32 [M,2], uint8 descriptors [M,128]) pairs.
    Return an integer inlier count. Use ratio test, unique candidate keypoints,
    and cv2.findHomography with the supplied pixel threshold and fixed seed.
    Fewer than four usable matches or failed/degenerate estimation returns 0.
    Allowed: OpenCV BFMatcher and findHomography."""
    # TODO(P6a): implement according to the contract above.
    raise NotImplementedError("P6a: geometric_inliers")

def rerank(ranks, store, top_n=100):
    """Reorder only the first top_n results of each query using geometric_inliers.
    Input ranks [N,Q] is a complete permutation per column; do not mutate it.
    store.get(i, query=True/False) loads local features. top_n must be positive.
    Return (new_ranks [N,Q], sorted_inlier_counts [min(N,top_n),Q], mean_ms_per_query).
    Use decreasing inlier count with stable ties preserving original retrieval order.
    The tail after Top-N must stay unchanged; every result remains a permutation."""
    # TODO(P6b): implement according to the contract above.
    raise NotImplementedError("P6b: rerank")
