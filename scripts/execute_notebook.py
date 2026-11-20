"""Execute the single deliverable notebook using the project interpreter."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
path = root / "writeup.ipynb"
notebook = nbformat.read(path, as_version=4)
# Full-dataset experiments can legitimately take longer than ten minutes per cell.
client = NotebookClient(notebook, timeout=-1, kernel_name="ivr",
                        resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.write(notebook, path)
print(f"Executed {sum(c.cell_type == 'code' for c in notebook.cells)} code cells: {path}")
