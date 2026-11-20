"""Frozen DINOv2 / CLIP descriptors. No label-guided training."""
import time
from dataclasses import asdict, dataclass
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from .utils import fingerprint, normalize, save_npz, write_json, implementation_id

@dataclass(frozen=True)
class DeepConfig:
    family: str = "dinov2"
    model_id: str = "facebook/dinov2-small"
    resolution: int = 518
    pooling: str = "cls"
    gem_p: float = 3.0
    revision: str = "main"
    resize: str = "long_side"

    @property
    def name(self):
        if self.family == "clip":
            return "clip_b32_224"
        suffix = "" if self.resize == "long_side" else f"_{self.resize}"
        return f"dino_s14_{self.resolution}_{self.pooling}{suffix}"

class DeepExtractor:
    def __init__(self, config=DeepConfig(), device=None):
        from transformers import AutoModel, CLIPVisionModelWithProjection, CLIPImageProcessor
        from huggingface_hub import try_to_load_from_cache
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Reuse a complete pinned cache without repeated remote HEAD requests.
        weights = [try_to_load_from_cache(config.model_id, name, revision=config.revision)
                   for name in ("model.safetensors", "pytorch_model.bin")]
        offline = any(isinstance(path, str) for path in weights)
        if config.family == "dinov2":
            if config.pooling not in ("cls", "gem"):
                raise ValueError("DINO pooling must be cls or gem")
            self.model = AutoModel.from_pretrained(config.model_id, revision=config.revision, local_files_only=offline)
            self.processor = None
        elif config.family == "clip":
            self.model = CLIPVisionModelWithProjection.from_pretrained(config.model_id, revision=config.revision, local_files_only=offline)
            self.processor = CLIPImageProcessor.from_pretrained(config.model_id, revision=config.revision)
        else:
            raise ValueError(config.family)
        self.model.eval().requires_grad_(False).to(self.device)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    def preprocess(self, image):
        if self.processor is not None:
            return self.processor(images=image, return_tensors="pt")["pixel_values"]
        w, h = image.size
        scale = self.config.resolution / max(w, h)
        # Keep entire ROI, approximate aspect ratio, dimensions divisible by patch size.
        nw, nh = max(14, round(w * scale / 14) * 14), max(14, round(h * scale / 14) * 14)
        if self.config.resize == "square":
            nw = nh = max(14, round(self.config.resolution / 14) * 14)
        elif self.config.resize != "long_side":
            raise ValueError(f"Unknown DINO resize: {self.config.resize}")
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        x = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        return ((x - mean) / std).unsqueeze(0)

    @torch.inference_mode()
    def encode(self, image):
        x = self.preprocess(image).to(self.device)
        # float32 avoids device-dependent mixed-precision changes in the baseline.
        output = self.model(pixel_values=x)
        if self.config.family == "clip":
            feature = output.image_embeds
        else:
            feature = pool_dino_tokens(output.last_hidden_state, self.config.pooling, self.config.gem_p)
        return normalize(feature.float().cpu().numpy())[0]

    def extract(self, dataset, cache_dir):
        from pathlib import Path
        config = asdict(self.config)
        config["resolved_revision"] = self.resolved_revision
        key = fingerprint({"config": config, "gnd": dataset.signature, "version": 2, "implementation": implementation_id("deep.py", "data.py", "utils.py")})
        path = Path(cache_dir) / dataset.name / f"{self.config.name}_{key}.npz"
        if path.exists():
            with np.load(path) as z:
                return z["database"], z["queries"], path
        arrays = []
        timing = {}
        for query in (False, True):
            start = time.perf_counter()
            features = [self.encode(im) for im in tqdm(dataset.records(query), total=len(dataset.qimlist) if query else len(dataset.imlist),
                                                       desc=f"{dataset.name} {self.config.name} {'query' if query else 'db'}")]
            arrays.append(np.stack(features))
            timing["query_encoding_ms_mean" if query else "database_encoding_ms_mean"] = (time.perf_counter() - start) * 1000 / len(features)
        save_npz(path, database=arrays[0], queries=arrays[1])
        write_json(path.with_suffix(".json"), {"config": config, "dataset": dataset.name,
                   "annotation_sha256": dataset.signature, "query_crop": True, **timing})
        return *arrays, path


def pool_dino_tokens(tokens, pooling="cls", p=3.0, eps=1e-6):
    """P3: aggregate DINOv2 tokens [B,1+T,D] into features [B,D].
    CLS selects the class token. GeM uses ONLY patch tokens.
    For this exercise use positive GeM: clamp activations to eps before pooling.
    The result retains input device/dtype and is not L2-normalized here.
    Reject unknown pooling modes and nonpositive p with ValueError.
    Models and preprocessing are provided; this helper must work on CPU tensors.
    """
    # TODO(P3): implement CLS and GeM aggregation.
    raise NotImplementedError("P3: pool_dino_tokens")
