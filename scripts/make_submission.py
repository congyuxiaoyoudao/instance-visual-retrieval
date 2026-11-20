"""Package only current coursework; never bundle historical answers or caches."""
import argparse
from pathlib import Path
import zipfile
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument("--starter",action="store_true",help="distribute an unsolved starter, without results")
args=parser.parse_args()
files=[root/p for p in ("README.md","writeup.ipynb","pyproject.toml","requirements-lock.txt",
                       "run_all.sh","run_eval.sh","test_and_make_submission.sh",".gitignore")]
for name in ("instance_retrieval","tests"):
    files.extend(p for p in (root/name).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
files.extend(root/"scripts"/name for name in (
    "prepare.py","prepare_mirror.py","execute_notebook.py","make_submission.py"))
if not args.starter:
    files.extend(p for p in (root/"results/coursework").rglob("*") if p.is_file())
    files.extend(p for p in (root/"logs/coursework").rglob("*") if p.is_file())
with zipfile.ZipFile(root/"submission_code.zip","w",zipfile.ZIP_DEFLATED) as z:
    for path in files:
        if path.is_file():
            z.write(path,path.relative_to(root))
    z.writestr("PACKAGE_STATUS.txt",
               "UNSOLVED COURSEWORK STARTER; no completed results.\n" if args.starter
               else "Coursework submission; see writeup.ipynb and logs/coursework.\n")
print("Packaged",root/"submission_code.zip")
