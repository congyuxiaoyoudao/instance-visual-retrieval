"""Top-10 comparisons, with green positives, red negatives, gray ignored."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .evaluation import protocol_ground_truth

def top10_figure(dataset, query_index, rankings, protocol="hard", scores=None):
    truth = protocol_ground_truth(dataset.gnd, protocol)[query_index]
    positive, ignored = set(truth["ok"]), set(truth["junk"])
    fig, axes = plt.subplots(len(rankings), 11, figsize=(24, 2.7 * len(rankings)), squeeze=False)
    for row, (method, ranks) in enumerate(rankings.items()):
        axes[row, 0].imshow(dataset.image(query_index, query=True))
        axes[row, 0].set_title(f"Query {query_index}\n{method}", fontsize=8)
        for col, index in enumerate(ranks[:10, query_index], 1):
            ax = axes[row, col]
            ax.imshow(dataset.image(int(index)))
            label = "correct" if index in positive else "ignored" if index in ignored else "wrong"
            color = "#1a9850" if index in positive else "#888888" if index in ignored else "#d73027"
            value = f" {scores[method][index, query_index]:.3f}" if scores and method in scores else ""
            ax.set_title(f"#{col} {label}{value}", fontsize=8)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3)
                spine.set_edgecolor(color)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{dataset.name} / {protocol.upper()} / green=positive, red=negative, gray=ignored")
    fig.tight_layout()
    return fig

def load_rankings(root, dataset, methods):
    ranks, scores = {}, {}
    for method in methods:
        with np.load(Path(root) / "results" / "coursework" / f"{dataset}__{method}.npz") as z:
            ranks[method] = z["ranks"]
            if "scores" in z:
                scores[method] = z["scores"]
    return ranks, scores
