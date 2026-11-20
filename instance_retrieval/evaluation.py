"""Protocol adapter; AP and P@10 are computed ONLY by official revisitop."""
import numpy as np
from .vendor.evaluate import compute_map

def protocol_ground_truth(gnd, protocol):
    """Convert revisited labels into official evaluator dictionaries.
    Args: gnd is a list of easy/hard/junk index arrays; protocol is medium or hard.
    Return: one dict per query, with integer arrays named ok and junk.
    Follow the assignment protocol; reject unsupported protocols with ValueError.
    Do not mutate gnd and do not implement AP yourself."""
    # TODO(P1): implement according to the contract above.
    raise NotImplementedError("P1: protocol_ground_truth")

def evaluate(ranks, gnd):
    ranks = np.asarray(ranks)
    if ranks.ndim != 2 or ranks.shape[1] != len(gnd):
        raise ValueError("ranks must have shape [database_size, query_count]")
    if not np.issubdtype(ranks.dtype, np.integer):
        raise ValueError("rank indices must be integers")
    expected = np.arange(len(ranks))[:, None]
    if not np.array_equal(np.sort(ranks, axis=0), np.broadcast_to(expected, ranks.shape)):
        raise ValueError("Each query ranking must be a complete database permutation")
    result = {}
    for protocol in ("medium", "hard"):
        truth = protocol_ground_truth(gnd, protocol)
        if not any(len(g["ok"]) for g in truth):
            raise ValueError(f"No positive queries for {protocol}")
        mean_ap, aps, precision, per_query = compute_map(ranks, truth, kappas=[10])
        result[protocol] = {
            "mAP": float(mean_ap * 100),
            "mP@10": float(precision[0] * 100),
            "AP": [float(a * 100) if np.isfinite(a) else None for a in aps],
            "P@10": [float(a * 100) if np.isfinite(a) else None for a in per_query[:, 0]],
            "valid_queries": int(np.isfinite(aps).sum()),
        }
    return result
