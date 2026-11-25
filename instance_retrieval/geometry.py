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
    inlier_count = 0

    matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2)
    q_xy, q_desc = query
    t_xy, t_desc = candidate
    q_desc_root = rootsift(q_desc)
    t_desc_root = rootsift(t_desc)
    matches = matcher.knnMatch(q_desc_root, t_desc_root, k=2)
    good_matches = []

    # apply Lowe ratio test
    for pair in matches:
        m, n = pair
        # the nearest neighbor should be close enough
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return 0
    
    # homography RANSAC 
    src_pts = np.float32([q_xy[m.queryIdx] for m in good_matches]).reshape(-1 , 1, 2)
    dst_pts = np.float32([t_xy[m.trainIdx] for m in good_matches]).reshape(-1 , 1, 2)
    cv2.setRNGSeed(seed)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=threshold)
    if M is not None:
        inlier_count = int(np.sum(mask))
    else:
        inlier_count = 0
    return inlier_count

def rerank(ranks, store, top_n=100):
    """Reorder only the first top_n results of each query using geometric_inliers.
    Input ranks [N,Q] is a complete permutation per column; do not mutate it.
    store.get(i, query=True/False) loads local features. top_n must be positive.
    Return (new_ranks [N,Q], sorted_inlier_counts [min(N,top_n),Q], mean_ms_per_query).
    Use decreasing inlier count with stable ties preserving original retrieval order.
    The tail after Top-N must stay unchanged; every result remains a permutation."""
    # TODO(P6b): implement according to the contract above.
    if top_n <= 0:
        raise ValueError(f"top_n should be positive")

    # rank[i][q] is q-th query's i-th similar image's index
    # e.g. rank[0][1] is the 1 query finds the most similar image's index
    N, Q = ranks.shape
    K = min(N, top_n)
    new_ranks = ranks.copy()
    times = [] # each query's time
    sorted_inlier_counts = np.zeros((K, Q), dtype=np.uint32)

    for q in range(Q):
        t0 = time.perf_counter()
        q_feat = store.get(q, query=True)
        top_n_results = ranks[:K, q]
        inlier_counts = []

        # compute inlier count
        for idx in top_n_results:
            t_feat = store.get(idx, query=False)
            count = geometric_inliers(q_feat, t_feat)
            inlier_counts.append(count)

        # sort by decreasing inlier counts
        order = sorted(
            range(K),
            key=lambda i: -inlier_counts[i],
        )
        new_ranks[:K, q] = top_n_results[order]
        sorted_inlier_counts[:, q] = [inlier_counts[i] for i in order]

        times.append((time.perf_counter() - t0) * 1000)

    mean_ms_per_query = float(np.mean(times))
    return new_ranks, sorted_inlier_counts, mean_ms_per_query
            
