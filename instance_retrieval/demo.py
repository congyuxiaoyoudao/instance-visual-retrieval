"""Local Gradio UI, also launchable inside the assignment notebook."""
import json
import time
from pathlib import Path
import numpy as np
from .data import RevisitedDataset
from .deep import DeepExtractor
from .experiments import CONFIGS
from .search import cosine_search

class RetrievalDemo:
    def __init__(self, root, dataset="roxford5k", method="dino_s14_518_cls", device=None):
        self.root = Path(root)
        self.dataset = RevisitedDataset(self.root / "data", dataset)
        self.record = json.loads((self.root / "results" / "coursework" / f"{dataset}__{method}.json").read_text())
        self.extractor = DeepExtractor(CONFIGS[method], device=device)
        with np.load(self.root / self.record["feature_path"]) as z:
            self.database = z["database"]
        self.method = method

    def search(self, image):
        if image is None:
            raise ValueError("Please upload a query image.")
        start = time.perf_counter()
        feature = self.extractor.encode(image.convert("RGB"))
        ranks, scores = cosine_search(self.database, feature[None])
        gallery = [(self.dataset.image(int(i)), f"#{j + 1}  cosine={scores[i, 0]:.4f}")
                   for j, i in enumerate(ranks[:10, 0])]
        elapsed = (time.perf_counter() - start) * 1000
        return gallery, {"method": self.method, "dataset": self.dataset.name,
                         "response_ms_including_encoding_and_image_loading": round(elapsed, 1),
                         "benchmark_M_mAP": round(self.record["metrics"]["medium"]["mAP"], 1),
                         "benchmark_H_mAP": round(self.record["metrics"]["hard"]["mAP"], 1),
                         "note": "mAP is benchmark-wide, not a score for the uploaded query."}

    def build(self):
        import gradio as gr
        def on_search(image):
            gallery, info = self.search(image)
            metrics = (f"### 检索性能\n\n方法：**{self.method}** · {self.dataset.name}\n\n"
                       f"| 指标 | 数值 |\n|---|---:|\n"
                       f"| 响应时间（含编码和结果图加载） | {info['response_ms_including_encoding_and_image_loading']:.1f} ms |\n"
                       f"| Medium mAP | {info['benchmark_M_mAP']:.1f}% |\n"
                       f"| Hard mAP | {info['benchmark_H_mAP']:.1f}% |\n\n"
                       "mAP 为完整基准的评测结果，不代表这张上传图片的正确率。")
            return gallery, metrics
        with gr.Blocks(title="Instance Visual Retrieval") as app:
            gr.Markdown("# Instance Visual Retrieval\n上传目标图片，查看 Top-10 相似实例。基准查询已按标注框裁剪。")
            with gr.Row():
                query = gr.Image(type="pil", label="Query image", height=360)
                stats = gr.Markdown("### 检索性能\n上传查询图片并点击搜索，即可查看响应时间和基准指标。")
            button = gr.Button("Search Top-10", variant="primary")
            gallery = gr.Gallery(label="Retrieved images and cosine scores", columns=5, height="auto")
            button.click(on_search, inputs=query, outputs=[gallery, stats])
            example = self.root / "results" / "coursework" / "figures" / f"{self.dataset.name}_demo_query.jpg"
            example.parent.mkdir(parents=True, exist_ok=True)
            self.dataset.image(0, query=True).save(example)
            gr.Examples([[str(example)]], inputs=query)
        return app
