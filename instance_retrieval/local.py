"""Response-ranked uint8 SIFT caches, RootSIFT, Faiss vocabulary, VLAD."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import faiss
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from .utils import normalize, save_npz, implementation_id

MAX_KEYPOINTS = 1000

def extract_sift(image: Image, max_keypoints=MAX_KEYPOINTS):
    """Extract at most max_keypoints SIFT features from a PIL image.
    Return xy [M,2] float32 in original image coordinates and desc [M,128] uint8.
    Keep the strongest responses even if OpenCV returns extra tied features.
    Round and clip descriptors to [0,255] before uint8 storage.
    A blank image must return arrays with shapes (0,2) and (0,128).
    Allowed: OpenCV SIFT_create and detectAndCompute."""
    # TODO(P4a): implement according to the contract above.
    gray = np.asarray(image.convert("L"))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if len(keypoints) == 0:
        return np.empty((0,2), np.float32), np.empty((0,128),np.uint8)
    
    indices = sorted(range(len(keypoints)), 
                 key=lambda i: keypoints[i].response, 
                 reverse=True) # rank by response

    indices = indices[:max_keypoints]

    xy = np.array([keypoints[i].pt for i in indices], dtype=np.float32)   # (M, 2)
    desc = np.clip(np.rint(descriptors[indices]), 0, 255).astype(np.uint8)         # (M, 128)

    return xy, desc

def rootsift(descriptors):
    """Transform uint8 SIFT [M,128] to RootSIFT float32 [M,128].
    Use the RootSIFT definition from the handout.
    All-zero rows remain zero, empty inputs preserve shape, and input must not change."""
    # TODO(P4b): implement according to the contract above.
    l1_norm = descriptors.sum(axis=1, keepdims=True) + 1e-7
    root = np.sqrt(descriptors / l1_norm).astype(np.float32)
    return root


class LocalStore:
    def __init__(self, dataset, cache_dir):
        self.dataset = dataset
        self.directory = Path(cache_dir) / dataset.name / f"sift1000_{dataset.signature[:16]}_{implementation_id('local.py', 'data.py')}"

    def path(self, index, query=False):
        return self.directory / ("query" if query else "database") / f"{index:05d}.npz"

    def get(self, index, query=False):
        with np.load(self.path(index, query)) as z:
            return z["xy"], z["descriptors"]

    def extract(self, workers=8):
        for query in (False, True):
            n = len(self.dataset.qimlist) if query else len(self.dataset.imlist)
            def work(i):
                path = self.path(i, query)
                if not path.exists():
                    xy, descriptors = extract_sift(self.dataset.image(i, query))
                    save_npz(path, xy=xy, descriptors=descriptors)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(tqdm(pool.map(work, range(n)), total=n, desc=f"SIFT {self.dataset.name} query={query}"))
        return self

    @property
    def bytes_per_image(self):
        # Raw descriptor + xy payload; excludes NPZ header overhead.
        total = 0
        for i in range(len(self.dataset.imlist)):
            xy, desc = self.get(i)
            total += xy.nbytes + desc.nbytes
        return total / len(self.dataset.imlist)

def sample_descriptors(stores: LocalStore, count=1_000_000, seed=2026):
    """Sample up to count database descriptors, without replacement, then convert to RootSIFT.
    Inputs: LocalStore instances exposing dataset.imlist and get(index, query=False).
    Return float32 [min(count,total_database_descriptors),128], deterministic for seed.
    Every descriptor across both databases must have equal inclusion probability.
    Do not sample query images or concatenate all descriptors into a giant array.
    An empty pool raises ValueError. A two-pass or reservoir approach is acceptable."""
    # TODO(P4c): implement according to the contract above.
    rng = np.random.default_rng(seed)
    # we use a one pass reservoir method
    reservoir = np.empty((count, 128), dtype=np.float32)
    seen = 0 # count how many elements we have seen

    for store in stores:
        for i in range(len(store.dataset.imlist)):
            _, desc = store.get(i, query = False) # do not sample query image
            if desc is None or len(desc) == 0:
                continue

            for row in desc:
                if seen < count:
                    # stage 1: store elements
                    reservoir[seen] = rootsift(row)
                else:
                    # stage 2: replace elements
                    j = rng.integers(0, seen + 1)
                    if j < count:
                        reservoir[j] = rootsift(row)
                seen += 1

    if seen == 0:
        raise ValueError(f"The pool is empty")
    return reservoir[:seen]

def train_vocabulary(sample, k, seed=2026, iterations=25):
    """Train an unsupervised vocabulary from float32 RootSIFT sample [M,128].
    Return float32 centers [k,128]. Require M >= k; otherwise raise ValueError.
    Use seed, iterations, and Faiss Kmeans (allowed); control the number of CPU threads.
    Configure Faiss sampling so the requested ~1M descriptors are actually used.
    Do not use query relevance labels."""
    # TODO(P4d): implement according to the contract above.
    M, D = sample.shape
    if M < k:
        raise ValueError(f"M should be bigger than k, current value, M: {M}, k: {k}")
    
    kmeans = faiss.Kmeans(D, k, niter=iterations, verbose=True, gpu=False, seed=seed)
    kmeans.train(sample)
    return kmeans.centroids.astype(np.float32)

def encode_vlad(descriptors, centers):
    """Encode raw uint8 SIFT [M,128] using RootSIFT vocabulary centers [K,128].
    Return one float32 vector [K*128]. First obtain RootSIFT descriptors.
    Assign each descriptor to its nearest L2 center and aggregate residuals.
    Apply intra-normalization, signed square root and global L2 normalization.
    Empty images return all zeros; empty clusters must remain finite.
    You may use Faiss assignment; implement aggregation and normalization yourself."""
    # TODO(P5): implement according to the contract above.
    root_desc = rootsift(descriptors) # (M, D)
    K, D = centers.shape
    dists = (
        np.sum(root_desc**2, axis=1, keepdims=True) # (M, 1)
        - 2 * root_desc @ centers.T # (M, K)
        + np.sum(centers**2, axis=1, keepdims=True).T # (1, K)
    )
    # (M, K) a^2 - 2ab + b^2
    nearest_center_labels = np.argmin(dists, axis=1) # (M)

    residuals = np.zeros_like(centers, dtype=np.float32) # (K, D)

    # aggregate residuals for k-th clusters
    for k in range(K):
        mask = nearest_center_labels==k # (M)
        if np.any(mask):
            residuals[k] += np.sum(root_desc[mask] - centers[k], axis=0)

    # inner cluster normalization
    norms = np.linalg.norm(residuals, axis=1, keepdims=True)
    norms[norms == 0] = 1 # prevent 0 division
    residuals = residuals / norms

    # signed square root
    residuals = np.sign(residuals) * np.sqrt(np.abs(residuals))

    # global normalization
    residuals_flat = residuals.reshape(-1)
    global_norm = np.linalg.norm(residuals_flat)
    if global_norm > 0:
        residuals_flat = residuals_flat / global_norm

    return residuals_flat

def encode_store(store, centers, query=False):
    n = len(store.dataset.qimlist) if query else len(store.dataset.imlist)
    return np.stack([encode_vlad(store.get(i, query)[1], centers) for i in tqdm(range(n), desc=f"VLAD K={len(centers)}")])
