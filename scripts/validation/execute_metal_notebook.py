"""Execute the saved metal-benchmark analysis with this Python environment."""

import json
import os
import time
from pathlib import Path

import nbformat
from execute_cmos_notebook import LocalKernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[2]
    path = root / "examples/notebooks/metal_solver_benchmarks.ipynb"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    notebook = nbformat.read(path, 4)
    start = time.perf_counter()
    NotebookClient(
        notebook,
        timeout=180,
        kernel_manager_class=LocalKernelManager,
        resources={"metadata": {"path": str(root)}},
    ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    report = dict(
        notebook=str(path.relative_to(root)),
        seconds=time.perf_counter() - start,
        code_cells_executed=sum(
            c.cell_type == "code" and c.execution_count is not None
            for c in notebook.cells
        ),
        errors=sum(
            o.output_type == "error"
            for c in notebook.cells
            if c.cell_type == "code"
            for o in c.outputs
        ),
    )
    (root / "docs/reviews/metal-benchmarks/analysis_execution.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
