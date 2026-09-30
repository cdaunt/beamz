"""Execute the saved-evidence CMOS analysis; no new electromagnetic runs."""

import json
import os
import time

import nbformat
from execute_cmos_notebook import ROOT, LocalKernelManager
from nbclient import NotebookClient


def main():
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    path = ROOT / "examples/notebooks/cmos_cause_isolation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    start = time.perf_counter()
    NotebookClient(
        notebook,
        timeout=180,
        kernel_manager_class=LocalKernelManager,
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    report = dict(
        notebook=str(path.relative_to(ROOT)),
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
    (ROOT / "docs/reviews/cmos-parity-followup/analysis_execution.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
