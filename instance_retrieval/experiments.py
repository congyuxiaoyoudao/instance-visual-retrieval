"""Experiments E1/E2 and feature-only reproducible evaluation."""
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import DATASETS, RevisitedDataset
from .deep import DeepConfig, DeepExtractor
from .evaluation import evaluate
from .geometry import rerank
from .local import LocalStore, sample_descriptors, train_vocabulary, encode_store
from .search import cosine_search, benchmark_search
from .utils import fingerprint, save_npz, seed_everything, write_json, implementation_id

DINO_REV = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
CLIP_REV = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CONFIGS = {
    **{f"dino_s14_{r}_{p}": DeepConfig(resolution=r, pooling=p, revision=DINO_REV)
       for r in (224, 518) for p in ("cls", "gem")},
    "clip_b32_224": DeepConfig(family="clip", model_id="openai/clip-vit-base-patch32",
                               resolution=224, pooling="projection", revision=CLIP_REV, resize="official_processor"),
}
PAPER = {"roxford5k": {"medium": 68.8, "hard": 43.2},
         "rparis6k": {"medium": 84.6, "hard": 68.5}}

def hardware():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

def _save_feature_run(root, dataset, method, database, queries, feature_path, config, extra=None):
    root = Path(root)
    ranks, scores = cosine_search(database, queries)
    timing = benchmark_search(database, queries)
    record = {
        "dataset": dataset.name, "method": method, "seed": 2026,
        "database_size": len(database), "queries": len(queries), "query_crop": True,
        "annotation_sha256": dataset.signature, "dimension": database.shape[1],
        "bytes_per_image": database.shape[1] * database.dtype.itemsize,
        "feature_path": str(Path(feature_path).relative_to(root)), "config": config,
        "hardware": hardware(), **timing, **(extra or {}),
        "metrics": evaluate(ranks, dataset.gnd),
    }
    output = root / "results" / "coursework" / f"{dataset.name}__{method}"
    save_npz(output.with_suffix(".npz"), ranks=ranks, scores=scores)
    write_json(output.with_suffix(".json"), record)
    print(json.dumps({"dataset": dataset.name, "method": method,
                      "mAP": {p: record["metrics"][p]["mAP"] for p in ("medium", "hard")}}), flush=True)
    return record, ranks

def run_deep(root, methods=None, datasets=DATASETS, device=None):
    root = Path(root).resolve()
    seed_everything()
    for method in (methods or list(CONFIGS)):
        config = CONFIGS[method]
        extractor = DeepExtractor(config, device)
        for name in datasets:
            ds = RevisitedDataset(root / "data", name)
            db, q, path = extractor.extract(ds, root / "cache" / "coursework")
            metadata = json.loads(path.with_suffix(".json").read_text())
            _save_feature_run(root, ds, method, db, q, path, asdict(config),
                              {"query_encoding_ms_mean": metadata["query_encoding_ms_mean"],
                               "resolved_revision": extractor.resolved_revision})
        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def run_local(root, sizes=(64, 256), top_n=100, workers=8):
    root = Path(root).resolve()
    seed_everything()
    stores = [LocalStore(RevisitedDataset(root / "data", name), root / "cache" / "coursework").extract(workers) for name in DATASETS]
    signature = fingerprint({"datasets": [s.dataset.signature for s in stores], "implementation": implementation_id("local.py", "data.py", "utils.py")})
    sample = None
    for k in sizes:
        vocab = root / "cache" / "coursework" / f"vocabulary_k{k}_{signature}.npz"
        if vocab.exists():
            with np.load(vocab) as z:
                centers = z["centers"]
        else:
            if sample is None:
                sample = sample_descriptors(stores)
            centers = train_vocabulary(sample, k)
            save_npz(vocab, centers=centers)
            write_json(vocab.with_suffix(".json"), {"seed": 2026, "samples": len(sample), "k": k,
                       "iterations": 25, "training_data": "database-only, both datasets; no query labels"})
        for store in stores:
            ds = store.dataset
            method = f"rootsift_vlad{k}"
            path = root / "cache" / "coursework" / ds.name / f"{method}_{signature}.npz"
            if path.exists():
                with np.load(path) as z:
                    db, queries = z["database"], z["queries"]
            else:
                db = encode_store(store, centers)
                start = time.perf_counter()
                queries = encode_store(store, centers, query=True)
                query_encode_ms = (time.perf_counter() - start) * 1000 / len(queries)
                save_npz(path, database=db, queries=queries)
                write_json(path.with_suffix(".json"), {"query_vlad_encoding_ms_mean": query_encode_ms,
                           "note": "excludes SIFT extraction; SIFT cache reused"})
            config = {"k": k, "max_keypoints": 1000, "descriptor_storage": "uint8",
                      "encoding": "RootSIFT, VLAD, intra/signed-sqrt/L2 normalization",
                      "vocabulary_path": str(vocab.relative_to(root))}
            record, ranks = _save_feature_run(root, ds, method, db, queries, path, config)
            reranked, inliers, elapsed = rerank(ranks, store, top_n)
            record.update(method=method + "_ransac", rerank_top_n=top_n, rerank_ms_mean=elapsed,
                          local_bytes_per_image=store.bytes_per_image,
                          metrics=evaluate(reranked, ds.gnd))
            output = root / "results" / "coursework" / f"{ds.name}__{record['method']}"
            save_npz(output.with_suffix(".npz"), ranks=reranked, inliers=inliers)
            write_json(output.with_suffix(".json"), record)

def reevaluate(root):
    """Recompute every metric from cached features, including local geometric reranking."""
    root = Path(root).resolve()
    paths = sorted((root / "results" / "coursework").glob("*__*.json"))
    if not paths:
        raise FileNotFoundError("No result manifests; run extract experiments first.")
    for path in paths:
        record = json.loads(path.read_text())
        ds = RevisitedDataset(root / "data", record["dataset"])
        if record["annotation_sha256"] != ds.signature:
            raise ValueError(f"Ground-truth changed for {path}")
        with np.load(root / record["feature_path"]) as z:
            database, queries = z["database"], z["queries"]
        ranks, scores = cosine_search(database, queries)
        record.update(benchmark_search(database, queries))
        arrays = {"ranks": ranks, "scores": scores}
        if record["method"].endswith("_ransac"):
            ranks, inliers, elapsed = rerank(ranks, LocalStore(ds, root / "cache" / "coursework"), record["rerank_top_n"])
            record["rerank_ms_mean"] = elapsed
            arrays = {"ranks": ranks, "inliers": inliers}
        record["metrics"] = evaluate(ranks, ds.gnd)
        save_npz(path.with_suffix(".npz"), **arrays)
        write_json(path, record)
    return tables(root)

def tables(root):
    root = Path(root)
    rows, comparison = [], []
    for path in sorted((root / "results" / "coursework").glob("*__*.json")):
        r = json.loads(path.read_text())
        for protocol, metrics in r["metrics"].items():
            rows.append({"dataset": r["dataset"], "method": r["method"], "protocol": protocol,
                         "mAP": metrics["mAP"], "mP@10": metrics["mP@10"], "dimension": r["dimension"],
                         "bytes/image": r["bytes_per_image"], "local_bytes/image": r.get("local_bytes_per_image", 0),
                         "encode_ms": r.get("query_encoding_ms_mean", np.nan),
                         "search_ms": r["search_ms_mean"], "rerank_ms": r.get("rerank_ms_mean", 0)})
            if r["method"].startswith("dino"):
                paper = PAPER[r["dataset"]][protocol]
                comparison.append({"dataset": r["dataset"], "method": r["method"], "protocol": protocol,
                                   "paper_mAP": paper, "measured_mAP": metrics["mAP"],
                                   "delta_pp": metrics["mAP"] - paper,
                                   "needs_analysis": abs(metrics["mAP"] - paper) > 5})
    frame = pd.DataFrame(rows, columns=["dataset", "method", "protocol", "mAP", "mP@10",
                                               "dimension", "bytes/image", "local_bytes/image",
                                               "encode_ms", "search_ms", "rerank_ms"])
    out = root / "results" / "coursework"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "E1.csv", index=False)
    if len(frame):
        frame[frame.method.str.startswith("rootsift")].to_csv(out / "E2.csv", index=False)
    pd.DataFrame(comparison).to_csv(out / "baseline_comparison.csv", index=False)
    return frame
