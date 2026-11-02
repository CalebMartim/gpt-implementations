import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.count_sloc import count_implementation_sloc
from common.generate_golden import generate_golden_artifacts

IMPL_DIR = Path(__file__).resolve().parent
CU_SOURCE = IMPL_DIR / "gpt_cuda.cu"
DEFAULT_BINARY = IMPL_DIR / "gpt_cuda"


def _find_nvcc() -> str:
    if os.path.exists("/opt/cuda/bin/nvcc"):
        return "/opt/cuda/bin/nvcc"
    found = shutil.which("nvcc")
    if found is None:
        raise RuntimeError("nvcc compiler not found in /opt/cuda/bin/nvcc or PATH")
    return found


def compile_cuda_binary(
    binary_path: str | Path = DEFAULT_BINARY,
    force: bool = False,
) -> Path:
    bin_path = Path(binary_path)
    if (
        not force
        and bin_path.exists()
        and bin_path.stat().st_mtime >= CU_SOURCE.stat().st_mtime
    ):
        return bin_path

    nvcc = _find_nvcc()
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        nvcc,
        "-O3",
        str(CU_SOURCE),
        "-lcublas",
        "-o",
        str(bin_path),
    ]
    subprocess.run(cmd, check=True)
    return bin_path


def run_benchmark(
    golden_dir: str = "/tmp/golden",
    output_json: str = "results/results_03_cuda_cpp.json",
    max_iters: int = 5000,
    warmup_iters: int = 50,
    eval_interval: int = 500,
    eval_iters: int = 50,
    gen_tokens: int = 1000,
    binary_path: str | Path = DEFAULT_BINARY,
) -> dict:
    golden_path = Path(golden_dir)
    meta_file = golden_path / "golden_meta.json"
    if not meta_file.exists():
        generate_golden_artifacts(
            str(golden_path), num_steps=max_iters, eval_iters=eval_iters
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    bin_path = compile_cuda_binary(binary_path=binary_path)

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required to execute the native gpt_cuda binary."
        )

    cmd = [
        str(bin_path),
        str(golden_path),
        str(out_file),
        str(max_iters),
        str(warmup_iters),
        str(eval_interval),
        str(eval_iters),
        str(gen_tokens),
        f"{float(meta['loss_step_0']):.8f}",
        f"{float(meta['loss_step_1']):.8f}",
        str(int(meta["num_steps"])),
        str(int(meta["eval_iters"])),
    ]
    subprocess.run(cmd, check=True)

    report = json.loads(out_file.read_text(encoding="utf-8"))
    sloc_info = count_implementation_sloc(str(IMPL_DIR))
    report["sloc_core"] = sloc_info["sloc_core"]
    report["sloc_harness"] = sloc_info["sloc_harness"]
    report["sloc_total"] = sloc_info["sloc_total"]

    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile and run Native C++/CUDA + cuBLAS GPT benchmark."
    )
    parser.add_argument(
        "--golden-dir",
        type=str,
        default="/tmp/golden",
        help="Directory containing golden weights and batches.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/results_03_cuda_cpp.json",
        help="Path to write JSON benchmark report.",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--gen-tokens", type=int, default=1000)
    args = parser.parse_args()

    res = run_benchmark(
        golden_dir=args.golden_dir,
        output_json=args.output_json,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        gen_tokens=args.gen_tokens,
    )
    print(
        f"[03_cuda_cpp] train_tps={res['train_tokens_per_sec']:,.1f} tok/s | "
        f"infer_b1={res['infer_b1_tokens_per_sec']:,.1f} tok/s | "
        f"infer_b64={res['infer_b64_tokens_per_sec']:,.1f} tok/s | "
        f"val_loss={res['final_val_loss']:.4f}"
    )
