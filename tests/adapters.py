"""Public test boundary, following cs336's tests/adapters.py.

If you reorganize implementations, update these imports; keep test expectations.
Models/weights are not loaded by this module.
"""
from instance_retrieval.evaluation import protocol_ground_truth as run_protocol
from instance_retrieval.search import cosine_search as run_search
from instance_retrieval.deep import pool_dino_tokens as run_pool
from instance_retrieval.local import (
    extract_sift as run_sift, rootsift as run_rootsift,
    sample_descriptors as run_sample, train_vocabulary as run_vocabulary,
    encode_vlad as run_vlad,
)
from instance_retrieval.geometry import geometric_inliers as run_inliers, rerank as run_rerank
