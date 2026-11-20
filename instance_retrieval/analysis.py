"""Evidence-backed failure selection; human labels are kept separate from metrics."""
import json
from pathlib import Path
import pandas as pd
from .data import DATASETS, RevisitedDataset
from .evaluation import protocol_ground_truth
from .visualization import load_rankings, top10_figure

CATEGORIES = ("viewpoint", "lighting", "occlusion", "similar_building", "instance_detail", "semantic_motif", "annotation", "other")

def prepare_failures(root, methods=("dino_s14_518_cls", "clip_b32_224"), per_dataset=3):
    root = Path(root)
    records = []
    for name in DATASETS:
        ds = RevisitedDataset(root / "data", name)
        ranks, scores = load_rankings(root, name, methods)
        truth = protocol_ground_truth(ds.gnd, "hard")
        candidates = []
        for q, g in enumerate(truth):
            wrong = {m: [int(i) for i in ranks[m][:10, q] if i not in g["ok"] and i not in g["junk"]] for m in methods}
            if any(wrong.values()):
                candidates.append((sum(map(len, wrong.values())), q, wrong))
        for _, q, wrong in sorted(candidates, reverse=True)[:per_dataset]:
            output = root / "results" / "coursework" / "figures" / f"{name}_q{q}_hard.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig = top10_figure(ds, q, ranks, "hard", scores)
            fig.savefig(output, dpi=130)
            import matplotlib.pyplot as plt
            plt.close(fig)
            for method in methods:
                for index in wrong[method]:
                    records.append({"dataset": name, "protocol": "hard", "query_index": q,
                                    "query_name": ds.qimlist[q], "method": method, "image_index": index,
                                    "image_name": ds.imlist[index], "category": "", "explanation": "",
                                    "figure": str(output.relative_to(root))})
    path = root / "results" / "coursework" / "failure_annotations.csv"
    fresh = pd.DataFrame(records, columns=["dataset", "protocol", "query_index", "query_name",
                                                      "method", "image_index", "image_name", "category",
                                                      "explanation", "figure"])
    if path.exists():
        old = pd.read_csv(path).fillna("")
        keys = ["dataset", "protocol", "query_index", "method", "image_index"]
        old = old.set_index(keys)
        for i, row in fresh.iterrows():
            key = tuple(row[k] for k in keys)
            if key in old.index:
                fresh.loc[i, ["category", "explanation"]] = old.loc[key, ["category", "explanation"]].values
    fresh.to_csv(path, index=False)
    return fresh

def failure_summary(root):
    path = Path(root) / "results" / "coursework" / "failure_annotations.csv"
    data = pd.read_csv(path).fillna("")
    reviewed = data[data.category.isin(CATEGORIES) & data.explanation.ne("")]
    summary = reviewed.groupby(["method", "category"]).size().unstack(fill_value=0)
    summary.to_csv(path.with_name("E3_error_categories.csv"))
    return summary, {"candidate_errors": len(data), "reviewed_errors": len(reviewed),
                     "unreviewed_errors": len(data) - len(reviewed),
                     "scope": "selected failure queries only; not an unbiased dataset-wide estimate"}
