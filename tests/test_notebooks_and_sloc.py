import json
from pathlib import Path
import pytest

from common.count_sloc import count_all_implementations

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "nb_relpath,expected_script,expected_json",
    [
        (
            "notebooks/01_benchmark_pytorch.ipynb",
            "implementations/01_pytorch/benchmark.py",
            "results_01_pytorch.json",
        ),
        (
            "notebooks/02_benchmark_triton.ipynb",
            "implementations/02_triton/benchmark.py",
            "results_02_triton.json",
        ),
        (
            "notebooks/03_benchmark_cuda_cpp.ipynb",
            "implementations/03_cuda_cpp/benchmark.py",
            "results_03_cuda_cpp.json",
        ),
    ],
)
def test_colab_notebook_structure_and_contents(
    nb_relpath: str, expected_script: str, expected_json: str
) -> None:
    nb_path = REPO_ROOT / nb_relpath
    assert nb_path.exists(), f"Notebook missing: {nb_relpath}"

    nb_data = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb_data.get("nbformat") == 4, f"Expected nbformat=4 in {nb_relpath}"
    assert "cells" in nb_data and isinstance(nb_data["cells"], list)

    code_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "code"]
    md_cells = [c for c in nb_data["cells"] if c.get("cell_type") == "markdown"]
    assert len(code_cells) >= 4, (
        f"Expected at least 4 code cells in {nb_relpath}, got {len(code_cells)}"
    )
    assert len(md_cells) >= 1, f"Expected markdown cells in {nb_relpath}"

    def _cell_source(cell: dict) -> str:
        src = cell.get("source", [])
        return "".join(src) if isinstance(src, list) else str(src)

    all_code = "\n".join(_cell_source(c) for c in code_cells)

    # Cell 1 requirements: GPU check and GitHub clone config
    assert "nvidia-smi" in _cell_source(code_cells[0])
    assert "REPO_URL" in _cell_source(code_cells[0])

    # Cell 2 requirements: golden generation with seed 998244353
    assert "common.generate_golden" in _cell_source(code_cells[1])
    assert "998244353" in _cell_source(code_cells[1])
    assert "/tmp/golden_gpt" in _cell_source(code_cells[1])

    # Cell 3 requirements: full benchmark invocation
    assert expected_script in _cell_source(code_cells[2])
    assert expected_json in _cell_source(code_cells[2])
    assert "--max-iters 5000" in _cell_source(code_cells[2])
    assert "--warmup-iters 50" in _cell_source(code_cells[2])
    assert "--eval-interval 500" in _cell_source(code_cells[2])
    assert "--eval-iters 50" in _cell_source(code_cells[2])
    assert "--gen-tokens 1000" in _cell_source(code_cells[2])

    # Cell 4 requirements: inspect and download JSON
    assert expected_json in _cell_source(code_cells[3])
    assert "files.download" in _cell_source(code_cells[3])
    assert "ImportError" in _cell_source(code_cells[3])

    assert expected_script in all_code
    assert "998244353" in all_code


def test_sloc_ordering_across_implementations() -> None:
    report = count_all_implementations(str(REPO_ROOT / "implementations"))
    assert "01_pytorch" in report
    assert "02_triton" in report
    assert "03_cuda_cpp" in report

    sloc_pt = report["01_pytorch"]["sloc_core"]
    sloc_tr = report["02_triton"]["sloc_core"]
    sloc_cu = report["03_cuda_cpp"]["sloc_core"]

    assert sloc_pt > 0
    assert sloc_pt < sloc_tr < sloc_cu, (
        f"Expected 01_pytorch ({sloc_pt}) < 02_triton ({sloc_tr}) < 03_cuda_cpp ({sloc_cu})"
    )


def test_readme_exists_and_documents_sloc_and_workflow() -> None:
    readme_path = REPO_ROOT / "README.md"
    assert readme_path.exists(), "README.md does not exist"

    content = readme_path.read_text(encoding="utf-8")
    report = count_all_implementations(str(REPO_ROOT / "implementations"))

    # Check architecture & seed constants
    assert "998244353" in content
    assert "10,788,929" in content or "10.788.929" in content or "10788929" in content
    assert "65" in content

    # Check implementations and notebooks
    assert "01_pytorch" in content
    assert "02_triton" in content
    assert "03_cuda_cpp" in content
    assert "01_benchmark_pytorch.ipynb" in content
    assert "02_benchmark_triton.ipynb" in content
    assert "03_benchmark_cuda_cpp.ipynb" in content

    # Check exact SLOC numbers from count_sloc.py are documented in README
    for impl_key in ("01_pytorch", "02_triton", "03_cuda_cpp"):
        assert str(report[impl_key]["sloc_core"]) in content
        assert str(report[impl_key]["sloc_harness"]) in content
        assert str(report[impl_key]["sloc_total"]) in content

    # Check aggregation command instruction
    assert (
        "python3 -m common.aggregate_results --results-dir results --output-dir results/summary"
        in content
    )
