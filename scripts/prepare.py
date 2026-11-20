"""Download official data and vendor the evaluator byte-for-byte."""
import concurrent.futures
import hashlib
import json
import pickle
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
URLS = {
    "roxford5k": ["https://thor.robots.ox.ac.uk/oxbuildings/oxbuild_images-v1.tgz"],
    "rparis6k": ["https://thor.robots.ox.ac.uk/parisbuildings/paris_1-v1.tgz",
                 "https://thor.robots.ox.ac.uk/parisbuildings/paris_2-v1.tgz"],
}

def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl", "-4", "-fsSL", "--retry", "3", "--connect-timeout", "20", "--speed-time", "30", "--speed-limit", "1024",
                        "--max-time", "3600", "-C", "-", url, "-o", str(partial)], check=True)
        partial.replace(target)

def prepare_dataset(name):
    dest = DATA / name
    annotation = dest / f"gnd_{name}.pkl"
    download(f"https://cmp.felk.cvut.cz/revisitop/data/datasets/{name}/gnd_{name}.pkl", annotation)
    with annotation.open("rb") as f:
        cfg = pickle.load(f)
    required = set(cfg["imlist"] + cfg["qimlist"])
    if all((dest / "jpg" / f"{stem}.jpg").is_file() for stem in required):
        print(name, "already complete", flush=True)
        return
    for url in URLS[name]:
        archive = DATA / "archives" / url.rsplit("/", 1)[1]
        download(url, archive)
        jpg = dest / "jpg"
        jpg.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            # Stream ordinary JPEG members only; flatten paths, never extract links.
            for member in tar:
                if member.isfile() and member.name.lower().endswith(".jpg"):
                    target = jpg / Path(member.name).name
                    if not target.exists():
                        with tar.extractfile(member) as source, target.open("wb") as out:
                            shutil.copyfileobj(source, out)
    missing = [stem for stem in required if not (dest / "jpg" / f"{stem}.jpg").is_file()]
    if missing:
        raise RuntimeError(f"{name}: missing {len(missing)} images after extraction")
    print(name, "prepared", flush=True)

def vendor():
    target = ROOT / "instance_retrieval" / "vendor"
    target.mkdir(parents=True, exist_ok=True)
    source = "https://raw.githubusercontent.com/filipradenovic/revisitop/master/python/evaluate.py"
    download(source, target / "evaluate.py")
    (target / "provenance.json").write_text(json.dumps({
        "repository": "https://github.com/filipradenovic/revisitop",
        "source": source,
        "sha256": hashlib.sha256((target / "evaluate.py").read_bytes()).hexdigest(),
        "modified": False,
        "note": "Original upstream evaluator; numpy<2 preserves np.in1d compatibility."
    }, indent=2))
    download("https://raw.githubusercontent.com/filipradenovic/revisitop/master/README.md",
             target / "UPSTREAM_README.md")

if __name__ == "__main__":
    vendor()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(prepare_dataset, URLS))

