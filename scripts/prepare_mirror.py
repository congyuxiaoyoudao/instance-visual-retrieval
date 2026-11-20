"""Mirror fallback: restore JPEG bytes, then verify names and query labels against official pkl."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import pickle
import subprocess
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ac376d0b8dd52898b75b301170ee7f5f0c76c1b5"
BASE = f"https://hf-mirror.com/datasets/vikhyatk/revisitop/resolve/{REVISION}"

def shard(args):
    name, filename = args
    target = ROOT / "data" / "mirror" / name / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        part = target.with_suffix(".part")
        subprocess.run(["curl", "-4", "-fsSL", "--retry", "3", "--connect-timeout", "20",
                        "--max-time", "900", "--speed-time", "30", "--speed-limit", "1024",
                        "-C", "-", f"{BASE}/{name}/{filename}", "-o", str(part)], check=True)
        part.replace(target)
    rows = pq.read_table(target).to_pylist()
    dest = ROOT / "data" / name / "jpg"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest.parent / f"gnd_{name}.pkl").open("rb") as f:
        official = pickle.load(f)
    for row in rows:
        basename = Path(row["filename"]).stem
        if basename not in set(official["imlist"] + official["qimlist"]):
            raise ValueError(f"Unknown image {basename}")
        if filename.startswith("qimlist"):
            q = official["qimlist"].index(basename)
            for key in ("bbx", "easy", "hard", "junk"):
                if list(row[key]) != list(official["gnd"][q][key]):
                    raise ValueError(f"Mirror label mismatch: {basename}, {key}")
        image = row["image"]["bytes"]
        (dest / f"{basename}.jpg").write_bytes(image)
    print(name, filename, len(rows), "verified", flush=True)

if __name__ == "__main__":
    jobs = []
    for name, count in (("roxford5k", 20), ("rparis6k", 26)):
        jobs.extend((name, f"imlist-{i:05d}-of-{count:05d}.parquet") for i in range(count))
        jobs.append((name, "qimlist-00000-of-00001.parquet"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(shard, jobs))
    (ROOT / "data" / "mirror_provenance.json").write_text(json.dumps({
        "repository": "vikhyatk/revisitop", "revision": REVISION,
        "image_handling": "original embedded JPEG bytes, no resize/recompression",
        "verification": "every filename and query bbx/easy/hard/junk matched against official ground truth"
    }, indent=2))
