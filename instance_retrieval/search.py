"""Exact cosine retrieval and actual single-query CPU latency."""
import time
import numpy as np
from .utils import normalize

def cosine_search(database, queries):
    """Rank queries by cosine similarity.
    Inputs: database [N,D], queries [Q,D], possibly not L2-normalized.
    Returns: ranks [N,Q] integer indices and scores [N,Q] float32 values.
    scores[i,j] is the similarity of database image i to query j, not reordered by ranks.
    Use decreasing similarity, stable ties (database index order), and complete rankings.
    Handle zero vectors without NaNs; do not mutate inputs. The supplied normalize helper is available."""
    # TODO(P2): implement according to the contract above.
    raise NotImplementedError("P2: cosine_search")

def benchmark_search(database, queries, repeats=3):
    database, queries = normalize(database), normalize(queries)
    _ = cosine_search(database, queries[:1])
    times = []
    for _ in range(repeats):
        for q in queries:
            start = time.perf_counter()
            cosine_search(database, q[None, :])
            times.append((time.perf_counter() - start) * 1000)
    return {"search_ms_mean": float(np.mean(times)),
            "search_ms_p95": float(np.percentile(times, 95)),
            "search_timing": "single query CPU retrieval function; includes normalization, excludes image encoding"}
