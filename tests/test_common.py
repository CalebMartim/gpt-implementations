import json
import os
import numpy as np
import pytest

from common.generate_golden import generate_golden_artifacts, PARAM_KEYS
from common.count_sloc import (
    count_file_sloc,
    count_implementation_sloc,
    count_all_implementations,
)
from common.aggregate_results import aggregate_all


def test_generate_golden_artifacts(tmp_path):
    meta = generate_golden_artifacts(
        str(tmp_path), num_steps=10, eval_iters=4, seed=998244353
    )
    assert meta["seed"] == 998244353
    assert meta["num_parameters"] == 10788929
    assert meta["vocab_size"] == 65
    assert meta["batch_size"] == 64
    assert meta["block_size"] == 256
    assert meta["n_embd"] == 384
    assert meta["n_head"] == 6
    assert meta["n_layer"] == 6
    assert 4.0 < meta["loss_step_0"] < 4.5
    assert meta["loss_step_1"] < meta["loss_step_0"]
    assert os.path.exists(tmp_path / "golden_weights.npz")
    assert os.path.exists(tmp_path / "golden_weights.bin")
    assert os.path.exists(tmp_path / "golden_batches.npz")
    assert os.path.exists(tmp_path / "golden_batches.bin")
    assert os.path.exists(tmp_path / "golden_meta.json")

    weights = np.load(tmp_path / "golden_weights.npz")
    total_floats = sum(weights[k].size for k in PARAM_KEYS)
    assert total_floats == 10788929
    bin_floats = os.path.getsize(tmp_path / "golden_weights.bin") // 4
    assert bin_floats == total_floats

    # Verify binary weights match npz weights in exact PARAM_KEYS order
    raw_weights = np.fromfile(tmp_path / "golden_weights.bin", dtype=np.float32)
    concat_weights = np.concatenate([weights[k].reshape(-1) for k in PARAM_KEYS])
    np.testing.assert_array_equal(raw_weights, concat_weights)

    # Verify batches shapes and binary order
    batches = np.load(tmp_path / "golden_batches.npz")
    assert batches["train_x"].shape == (10, 64, 256)
    assert batches["train_y"].shape == (10, 64, 256)
    assert batches["eval_train_x"].shape == (4, 64, 256)
    assert batches["eval_train_y"].shape == (4, 64, 256)
    assert batches["val_x"].shape == (4, 64, 256)
    assert batches["val_y"].shape == (4, 64, 256)
    assert batches["infer_prompt_b1"].shape == (1, 1)
    assert batches["infer_prompt_b64"].shape == (64, 1)

    raw_batches = np.fromfile(tmp_path / "golden_batches.bin", dtype=np.int32)
    expected_batches = np.concatenate(
        [
            batches["train_x"].reshape(-1),
            batches["train_y"].reshape(-1),
            batches["eval_train_x"].reshape(-1),
            batches["eval_train_y"].reshape(-1),
            batches["val_x"].reshape(-1),
            batches["val_y"].reshape(-1),
        ]
    )
    np.testing.assert_array_equal(raw_batches, expected_batches)


def test_count_file_sloc(tmp_path):
    sample_py = tmp_path / "sample.py"
    sample_py.write_text(
        '''"""Docstring
multiline"""
# Comment
x = 1
y = 2 # inline

'''
    )
    assert count_file_sloc(str(sample_py)) == 2

    sample_cu = tmp_path / "sample.cu"
    sample_cu.write_text(
        """// Single line comment
/* Block comment
   spanning lines */
__global__ void kernel(float* x) {
    int i = threadIdx.x; // inline comment
    x[i] = 0.0f;
}
"""
    )
    assert count_file_sloc(str(sample_cu)) == 4


def test_count_implementation_sloc(tmp_path):
    impl_dir = tmp_path / "implementations"
    py_dir = impl_dir / "01_pytorch"
    py_dir.mkdir(parents=True)
    (py_dir / "model.py").write_text("# model\nclass GPT:\n    pass\n")
    (py_dir / "benchmark.py").write_text("# bench\ndef run():\n    return 1\n")

    res_py = count_implementation_sloc(str(py_dir))
    assert res_py["sloc_core"] == 2
    assert res_py["sloc_harness"] == 2
    assert res_py["sloc_total"] == 4

    cu_dir = impl_dir / "03_cuda_cpp"
    cu_dir.mkdir(parents=True)
    (cu_dir / "gpt_cuda.cu").write_text(
        """// Core kernel
__global__ void k() {}
void forward() {}
// === BENCHMARK HARNESS ===
int main() {
    return 0;
}
"""
    )
    res_cu = count_implementation_sloc(str(cu_dir))
    assert res_cu["sloc_core"] == 2
    assert res_cu["sloc_harness"] == 3
    assert res_cu["sloc_total"] == 5

    all_res = count_all_implementations(str(impl_dir))
    assert "01_pytorch" in all_res
    assert "03_cuda_cpp" in all_res


def test_aggregate_all(tmp_path):
    results_dir = tmp_path / "results"
    output_dir = tmp_path / "out"
    results_dir.mkdir()

    sample_data = [
        {
            "implementation": "01_pytorch",
            "gpu_name": "NVIDIA L4",
            "train_time_s": 120.0,
            "train_tokens_per_sec": 682666.7,
            "infer_b1_time_s": 2.5,
            "infer_b1_tokens_per_sec": 400.0,
            "infer_b64_time_s": 4.0,
            "infer_b64_tokens_per_sec": 16000.0,
            "peak_vram_mb": 2400.0,
            "warmup_time_s": 1.2,
            "final_train_loss": 1.45,
            "final_val_loss": 1.65,
            "loss_history": [
                {"step": 0, "train_loss": 4.17, "val_loss": 4.17},
                {"step": 500, "train_loss": 1.45, "val_loss": 1.65},
            ],
        },
        {
            "implementation": "02_triton",
            "gpu_name": "NVIDIA L4",
            "train_time_s": 90.0,
            "train_tokens_per_sec": 910222.2,
            "infer_b1_time_s": 1.8,
            "infer_b1_tokens_per_sec": 555.6,
            "infer_b64_time_s": 2.8,
            "infer_b64_tokens_per_sec": 22857.1,
            "peak_vram_mb": 1600.0,
            "warmup_time_s": 3.5,
            "final_train_loss": 1.45,
            "final_val_loss": 1.65,
            "loss_history": [
                {"step": 0, "train_loss": 4.17, "val_loss": 4.17},
                {"step": 500, "train_loss": 1.45, "val_loss": 1.65},
            ],
        },
    ]

    for item in sample_data:
        fname = results_dir / f"results_{item['implementation']}.json"
        fname.write_text(json.dumps(item))

    summary = aggregate_all(str(results_dir), str(output_dir))
    assert "implementations" in summary
    assert len(summary["implementations"]) == 2
    assert os.path.exists(output_dir / "summary_table.md")
    assert os.path.exists(output_dir / "summary_table.tex")
    assert os.path.exists(output_dir / "throughput_comparison.png")
    assert os.path.exists(output_dir / "pareto_sloc_vs_throughput.png")
    assert os.path.exists(output_dir / "loss_convergence.png")
