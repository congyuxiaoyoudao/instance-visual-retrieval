"""Local Gradio UI, also launchable inside the assignment notebook."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import ImageOps
from .data import DATASETS, RevisitedDataset
from .deep import DeepExtractor
from .experiments import CONFIGS
from .evaluation import protocol_ground_truth
from .search import cosine_search

class RetrievalDemo:
    def __init__(self, root, dataset="roxford5k", method="dino_s14_518_cls", device=None):
        self.root = Path(root)
        self.method = method
        self._datasets = {}
        self._query_indices = {}
        self.dataset, self.record, self.database = self._dataset_data(dataset)
        self.extractor = DeepExtractor(CONFIGS[method], device=device)

    def _dataset_data(self, name):
        """Reuse features for each dataset without changing another session's selection."""
        if name not in DATASETS:
            raise ValueError(f"Unknown dataset: {name}")
        if name not in self._datasets:
            dataset = RevisitedDataset(self.root / "data", name)
            record = json.loads((self.root / "results" / "coursework" / f"{name}__{self.method}.json").read_text())
            with np.load(self.root / record["feature_path"]) as z:
                database = z["database"]
            self._datasets[name] = dataset, record, database
        return self._datasets[name]

    @staticmethod
    def _image_key(image):
        image = image.convert("RGB")
        return image.size, hashlib.sha256(image.tobytes()).digest()

    def _query_index(self, image, dataset):
        """Match exact official crops; edited or unknown uploads have no ground truth."""
        if dataset.name not in self._query_indices:
            indices = {}
            for index in range(len(dataset.qimlist)):
                key = self._image_key(dataset.image(index, query=True))
                # Identical crops with multiple query identities are ambiguous.
                indices[key] = None if key in indices else index
            self._query_indices[dataset.name] = indices
        return self._query_indices[dataset.name].get(self._image_key(image))

    def search(self, image, dataset=None, protocol="hard"):
        if image is None:
            raise ValueError("Please upload a query image.")
        if protocol not in ("medium", "hard"):
            raise ValueError(f"Unsupported protocol: {protocol}")
        dataset, record, database = self._dataset_data(dataset or self.dataset.name)
        start = time.perf_counter()
        query_index = self._query_index(image, dataset)
        positive, ignored = set(), set()
        if query_index is not None:
            truth = protocol_ground_truth(dataset.gnd, protocol)[query_index]
            positive, ignored = set(truth["ok"]), set(truth["junk"])
        feature = self.extractor.encode(image.convert("RGB"))
        ranks, scores = cosine_search(database, feature[None])
        gallery = []
        for j, i in enumerate(ranks[:10, 0]):
            result = dataset.image(int(i))
            label = "未标注"
            if query_index is not None:
                if i in positive:
                    label, color = "正例", "#1a9850"
                elif i in ignored:
                    label, color = "忽略", "#888888"
                else:
                    label, color = "负例", "#d73027"
                border = max(4, round(min(result.size) * 0.025))
                result = ImageOps.expand(result, border=border, fill=color)
            gallery.append((result, f"#{j + 1}  {label}  cosine={scores[i, 0]:.4f}"))
        elapsed = (time.perf_counter() - start) * 1000
        return gallery, {"method": self.method, "dataset": dataset.name,
                         "query_index": query_index, "protocol": protocol,
                         "response_ms_including_encoding_and_image_loading": round(elapsed, 1),
                         "benchmark_M_mAP": round(record["metrics"]["medium"]["mAP"], 1),
                         "benchmark_H_mAP": round(record["metrics"]["hard"]["mAP"], 1),
                         "note": "mAP is benchmark-wide, not a score for the uploaded query."}

    def build(self):
        import gradio as gr
        def idle_stats(name):
            return f"### 检索性能\n当前数据集：**{name}**。选择或上传查询图片，再点击搜索。"

        def on_search(image, name, protocol):
            gallery, info = self.search(image, name, protocol)
            annotation = (f"Query {info['query_index']} · {protocol.capitalize()} 协议"
                          if info["query_index"] is not None else "当前图片未匹配到官方 query，检索结果未标注。")
            metrics = (f"### 检索性能\n\n方法：**{self.method}** · {info['dataset']}\n\n"
                       f"{annotation}\n\n"
                       f"| 指标 | 数值 |\n|---|---:|\n"
                       f"| 响应时间（含编码和结果图加载） | {info['response_ms_including_encoding_and_image_loading']:.1f} ms |\n"
                       f"| Medium mAP | {info['benchmark_M_mAP']:.1f}% |\n"
                       f"| Hard mAP | {info['benchmark_H_mAP']:.1f}% |\n\n"
                       "mAP 为完整基准的评测结果，不代表这张上传图片的正确率。")
            return gallery, metrics
        with gr.Blocks(title="Instance Visual Retrieval") as app:
            gr.Markdown("# Instance Visual Retrieval\n上传目标图片，查看 Top-10 相似实例。基准查询已按标注框裁剪。")
            with gr.Row():
                dataset_selector = gr.Dropdown(list(DATASETS), value=self.dataset.name,
                                               label="数据集", interactive=True)
                display_mode = gr.Radio(["图片", "标题"], value="图片", label="Query 显示方式")
                protocol_selector = gr.Radio([("Medium (M)", "medium"), ("Hard (H)", "hard")],
                                             value="hard", label="边框标注协议")
            with gr.Row():
                query = gr.Image(type="pil", label="Query image", height=360)
                stats = gr.Markdown(idle_stats(self.dataset.name))
            button = gr.Button("Search Top-10", variant="primary")
            gr.Markdown("**边框：** 🟩 正例 · 🟥 负例 · 灰色为忽略项。普通上传图片没有官方标注时不着色。")
            gallery = gr.Gallery(label="Retrieved images and cosine scores", columns=5, height="auto",
                                 object_fit="contain")
            search_inputs = [query, dataset_selector, protocol_selector]
            search_event = button.click(on_search, inputs=search_inputs, outputs=[gallery, stats],
                                        concurrency_id="demo_search")

            def change_protocol(image, name, protocol):
                return on_search(image, name, protocol) if image is not None else ([], idle_stats(name))

            protocol_event = protocol_selector.change(change_protocol, inputs=search_inputs,
                                                       outputs=[gallery, stats], concurrency_id="demo_search")
            example_views = {}
            for name in DATASETS:
                dataset, _, _ = self._dataset_data(name)
                example_dir = self.root / "results" / "coursework" / "figures" / "demo_queries" / name
                example_dir.mkdir(parents=True, exist_ok=True)
                examples, labels = [], []
                for index, image_name in enumerate(dataset.qimlist):
                    example = example_dir / f"{index:02d}_{image_name}.png"
                    dataset.image(index, query=True).save(example)
                    examples.append([str(example)])
                    labels.append(f"Query {index}: {image_name}")
                for mode in ("图片", "标题"):
                    view = gr.Examples(examples, inputs=query,
                                       example_labels=labels if mode == "标题" else None,
                                       examples_per_page=6, cache_examples=False,
                                       visible=name == self.dataset.name and mode == "图片",
                                       label=f"Examples — {name} ({len(examples)} queries)")
                    example_views[name, mode] = view.dataset

            def show_examples(name, mode):
                return [gr.update(visible=key == (name, mode)) for key in example_views]

            def change_dataset(name, mode):
                return [None, [], idle_stats(name), *show_examples(name, mode)]

            dataset_selector.change(change_dataset, inputs=[dataset_selector, display_mode],
                                    outputs=[query, gallery, stats, *example_views.values()],
                                    cancels=[search_event, protocol_event], queue=False)
            display_mode.change(show_examples, inputs=[dataset_selector, display_mode],
                                outputs=list(example_views.values()), queue=False)
        return app
