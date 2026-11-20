"""Official image order and bounding-box query crops."""
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

DATASETS = ("roxford5k", "rparis6k")

@dataclass
class RevisitedDataset:
    root: Path
    name: str

    def __post_init__(self):
        if self.name not in DATASETS:
            raise ValueError(f"Unknown dataset: {self.name}")
        self.root = Path(self.root)
        self.directory = self.root / self.name
        annotation = self.directory / f"gnd_{self.name}.pkl"
        # Only use trusted official ground-truth pickle files.
        with annotation.open("rb") as f:
            cfg = pickle.load(f)
        self.imlist = list(cfg["imlist"])
        self.qimlist = list(cfg["qimlist"])
        self.gnd = cfg["gnd"]
        self.signature = hashlib.sha256(annotation.read_bytes()).hexdigest()
        self.paths = [self.directory / "jpg" / f"{name}.jpg" for name in self.imlist]
        self.query_paths = [self.directory / "jpg" / f"{name}.jpg" for name in self.qimlist]
        missing = [str(p) for p in set(self.paths + self.query_paths) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing images; first: {missing[0]}")
        if len(self.gnd) != len(self.qimlist):
            raise ValueError("Query annotation count mismatch")

    def image(self, index, query=False):
        path = self.query_paths[index] if query else self.paths[index]
        with Image.open(path) as im:
            image = im.convert("RGB")
        if query:
            # Same PIL crop convention as revisitop/example_process_images.py.
            image = image.crop(tuple(self.gnd[index]["bbx"]))
        return image

    def records(self, query=False):
        count = len(self.qimlist) if query else len(self.imlist)
        for i in range(count):
            yield self.image(i, query=query)
