import importlib
import os
from pathlib import Path
import shutil
import subprocess

from common.count_sloc import HARNESS_MARKER, count_implementation_sloc


REPO_ROOT = Path(__file__).resolve().parent.parent
IMPL_DIR = REPO_ROOT / "implementations" / "03_cuda_cpp"
CU_FILE = IMPL_DIR / "gpt_cuda.cu"


def _find_nvcc() -> str:
    if os.path.exists("/opt/cuda/bin/nvcc"):
        return "/opt/cuda/bin/nvcc"
    found = shutil.which("nvcc")
    assert found is not None, "nvcc compiler not found"
    return found


def test_cuda_source_and_sloc_split():
    assert CU_FILE.exists(), f"Missing {CU_FILE}"
    content = CU_FILE.read_text(encoding="utf-8")
    assert HARNESS_MARKER in content, f"Missing '{HARNESS_MARKER}' in {CU_FILE}"

    for kernel_name in (
        "embedding_forward_kernel",
        "embedding_backward_kernel",
        "layernorm_forward_kernel",
        "layernorm_backward_kernel",
        "qkv_split_transpose_kernel",
        "qkv_backward_transpose_kernel",
        "causal_softmax_forward_kernel",
        "causal_softmax_backward_kernel",
        "attn_concat_transpose_kernel",
        "bias_add_kernel",
        "bias_backward_kernel",
        "residual_add_kernel",
        "relu_forward_kernel",
        "relu_backward_kernel",
        "cross_entropy_forward_backward_kernel",
        "adamw_update_kernel",
    ):
        assert kernel_name in content, f"Missing CUDA kernel {kernel_name}"

    sloc = count_implementation_sloc(str(IMPL_DIR))
    assert sloc["sloc_core"] > 300, f"Expected sloc_core > 300, got {sloc['sloc_core']}"
    assert sloc["sloc_harness"] > 50, f"Expected sloc_harness > 50, got {sloc['sloc_harness']}"
    assert sloc["sloc_total"] == sloc["sloc_core"] + sloc["sloc_harness"]
    assert "gpt_cuda.cu (core)" in sloc["files"]
    assert "gpt_cuda.cu (harness)" in sloc["files"]


def test_nvcc_compiles_gpt_cuda(tmp_path):
    nvcc = _find_nvcc()
    out_bin = tmp_path / "gpt_cuda"
    cmd = [
        nvcc,
        "-O3",
        str(CU_FILE),
        "-lcublas",
        "-o",
        str(out_bin),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"nvcc compilation failed (rc={proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    assert out_bin.exists()


def test_benchmark_wrapper_and_makefile_exist():
    makefile = IMPL_DIR / "Makefile"
    assert makefile.exists(), "Missing Makefile in implementations/03_cuda_cpp"
    mk_text = makefile.read_text(encoding="utf-8")
    assert "-lcublas" in mk_text
    assert "gpt_cuda.cu" in mk_text

    bench_mod = importlib.import_module("implementations.03_cuda_cpp.benchmark")
    assert hasattr(bench_mod, "compile_cuda_binary")
    assert hasattr(bench_mod, "run_benchmark")
