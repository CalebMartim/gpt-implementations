import argparse
import importlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.count_sloc import count_implementation_sloc
from common.generate_golden import generate_golden_artifacts

_model_mod = importlib.import_module("implementations.02_triton.model")
GPTConfig = _model_mod.GPTConfig
TritonGPT = _model_mod.TritonGPT


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def _evaluate_split(
    model: torch.nn.Module,
    x_batches: np.ndarray,
    y_batches: np.ndarray,
    eval_iters: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    num_avail = len(x_batches)
    for k in range(eval_iters):
        idx = k % num_avail
        xb = torch.from_numpy(x_batches[idx]).long().to(device)
        yb = torch.from_numpy(y_batches[idx]).long().to(device)
        _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses))


def run_benchmark(
    golden_dir: str = "/tmp/golden",
    output_json: str = "results/results_02_triton.json",
    max_iters: int = 5000,
    warmup_iters: int = 50,
    eval_interval: int = 500,
    eval_iters: int = 50,
    gen_tokens: int = 1000,
    device: str | None = None,
) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    golden_path = Path(golden_dir)
    meta_file = golden_path / "golden_meta.json"
    if not meta_file.exists():
        generate_golden_artifacts(
            str(golden_path), num_steps=max_iters, eval_iters=eval_iters
        )

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    weights_path = str(golden_path / "golden_weights.npz")
    batches = np.load(golden_path / "golden_batches.npz")

    train_x = batches["train_x"]
    train_y = batches["train_y"]
    eval_train_x = batches["eval_train_x"]
    eval_train_y = batches["eval_train_y"]
    val_x = batches["val_x"]
    val_y = batches["val_y"]
    infer_prompt_b1 = batches["infer_prompt_b1"]
    infer_prompt_b64 = batches["infer_prompt_b64"]

    cfg = GPTConfig(
        vocab_size=int(meta["vocab_size"]),
        block_size=int(meta["block_size"]),
        n_embd=int(meta["n_embd"]),
        n_head=int(meta["n_head"]),
        n_layer=int(meta["n_layer"]),
        dropout=0.0,
    )
    model = TritonGPT(cfg).to(dev)

    def _make_optimizer() -> torch.optim.AdamW:
        return torch.optim.AdamW(
            model.parameters(),
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

    # 1. Sanity Check Gate
    model.load_golden_weights(weights_path)
    model.train()
    optimizer = _make_optimizer()

    x0 = torch.from_numpy(train_x[0]).long().to(dev)
    y0 = torch.from_numpy(train_y[0]).long().to(dev)
    _, loss0_tensor = model(x0, y0)
    loss_0 = float(loss0_tensor.item())

    optimizer.zero_grad(set_to_none=True)
    loss0_tensor.backward()
    optimizer.step()

    idx_next = 1 if len(train_x) > 1 else 0
    x1 = torch.from_numpy(train_x[idx_next]).long().to(dev)
    y1 = torch.from_numpy(train_y[idx_next]).long().to(dev)
    with torch.no_grad():
        _, loss1_tensor = model(x1, y1)
    loss_1 = float(loss1_tensor.item())

    diff_0 = abs(loss_0 - float(meta["loss_step_0"]))
    diff_1 = abs(loss_1 - float(meta["loss_step_1"]))
    assert diff_0 < 1e-4, (
        f"Sanity check failed at step 0: got {loss_0:.6f}, "
        f"expected {meta['loss_step_0']:.6f} (diff={diff_0:.6e})"
    )
    assert diff_1 < 1e-4, (
        f"Sanity check failed at step 1: got {loss_1:.6f}, "
        f"expected {meta['loss_step_1']:.6f} (diff={diff_1:.6e})"
    )

    # 2. Warm-up Phase
    model.load_golden_weights(weights_path)
    model.train()
    optimizer = _make_optimizer()
    _sync(dev)
    t_warm_start = time.perf_counter()
    for i in range(warmup_iters):
        idx = i % len(train_x)
        xb = torch.from_numpy(train_x[idx]).long().to(dev)
        yb = torch.from_numpy(train_y[idx]).long().to(dev)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    _sync(dev)
    warmup_time_s = time.perf_counter() - t_warm_start

    # 3. Timed Training Phase (starts cleanly from theta_0)
    model.load_golden_weights(weights_path)
    model.train()
    optimizer = _make_optimizer()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)

    loss_history: list[dict] = []
    train_wall_time_s = 0.0
    eval_wall_time_s = 0.0

    # Evaluate initial theta_0 at step 0 outside training timer
    t_eval_0 = time.perf_counter()
    tr_loss_0 = _evaluate_split(model, eval_train_x, eval_train_y, eval_iters, dev)
    va_loss_0 = _evaluate_split(model, val_x, val_y, eval_iters, dev)
    _sync(dev)
    eval_wall_time_s += time.perf_counter() - t_eval_0
    loss_history.append(
        {"step": 0, "train_loss": tr_loss_0, "val_loss": va_loss_0}
    )

    t_train_segment_start = time.perf_counter()
    for step in range(max_iters):
        idx = step % len(train_x)
        xb = torch.from_numpy(train_x[idx]).long().to(dev)
        yb = torch.from_numpy(train_y[idx]).long().to(dev)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        is_eval_step = (
            (step > 0 and step % eval_interval == 0) or step == max_iters - 1
        )
        if is_eval_step:
            _sync(dev)
            train_wall_time_s += time.perf_counter() - t_train_segment_start

            t_eval_start = time.perf_counter()
            tr_loss = _evaluate_split(
                model, eval_train_x, eval_train_y, eval_iters, dev
            )
            va_loss = _evaluate_split(model, val_x, val_y, eval_iters, dev)
            _sync(dev)
            eval_wall_time_s += time.perf_counter() - t_eval_start

            loss_history.append(
                {"step": step, "train_loss": tr_loss, "val_loss": va_loss}
            )
            t_train_segment_start = time.perf_counter()

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024.0 * 1024.0)
        if dev.type == "cuda"
        else 0.0
    )

    batch_size = int(meta["batch_size"])
    block_size = int(meta["block_size"])
    total_train_tokens = max_iters * batch_size * block_size
    train_tps = total_train_tokens / max(train_wall_time_s, 1e-9)

    # 4. Timed Inference Phase
    model.eval()

    # B = 1
    prompt_b1 = torch.from_numpy(infer_prompt_b1).long().to(dev)
    _ = model.generate(prompt_b1, max_new_tokens=5, deterministic=True)
    _sync(dev)
    t_b1_start = time.perf_counter()
    _ = model.generate(prompt_b1, max_new_tokens=gen_tokens, deterministic=True)
    _sync(dev)
    b1_time_s = time.perf_counter() - t_b1_start
    b1_tps = gen_tokens / max(b1_time_s, 1e-9)
    b1_latency_ms = (b1_time_s * 1000.0) / max(gen_tokens, 1)

    # B = 64
    prompt_b64 = torch.from_numpy(infer_prompt_b64).long().to(dev)
    _ = model.generate(prompt_b64, max_new_tokens=5, deterministic=True)
    _sync(dev)
    t_b64_start = time.perf_counter()
    _ = model.generate(
        prompt_b64, max_new_tokens=gen_tokens, deterministic=True
    )
    _sync(dev)
    b64_time_s = time.perf_counter() - t_b64_start
    b64_total_tokens = prompt_b64.size(0) * gen_tokens
    b64_tps = b64_total_tokens / max(b64_time_s, 1e-9)

    impl_dir = REPO_ROOT / "implementations" / "02_triton"
    sloc_info = count_implementation_sloc(str(impl_dir))

    gpu_name = (
        torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    )
    final_train_loss = loss_history[-1]["train_loss"]
    final_val_loss = loss_history[-1]["val_loss"]

    report = {
        "implementation": "02_triton",
        "language_paradigm": "PyTorch + Custom Triton Kernels (@triton.jit)",
        "gpu_name": gpu_name,
        "device": str(dev),
        "train_time_s": train_wall_time_s,
        "train_tokens_per_sec": train_tps,
        "eval_time_s": eval_wall_time_s,
        "warmup_time_s": warmup_time_s,
        "infer_b1_time_s": b1_time_s,
        "infer_b1_tokens_per_sec": b1_tps,
        "infer_b1_latency_ms_per_token": b1_latency_ms,
        "infer_b64_time_s": b64_time_s,
        "infer_b64_tokens_per_sec": b64_tps,
        "peak_vram_mb": peak_vram_mb,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "sloc_core": sloc_info["sloc_core"],
        "sloc_harness": sloc_info["sloc_harness"],
        "sloc_total": sloc_info["sloc_total"],
        "sanity_check": {
            "loss_step_0": loss_0,
            "golden_loss_step_0": float(meta["loss_step_0"]),
            "loss_step_0_diff": diff_0,
            "loss_step_1": loss_1,
            "golden_loss_step_1": float(meta["loss_step_1"]),
            "loss_step_1_diff": diff_1,
            "passed": True,
        },
        "training": {
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "warmup_time_s": warmup_time_s,
            "wall_time_s": train_wall_time_s,
            "eval_wall_time_s": eval_wall_time_s,
            "total_tokens": total_train_tokens,
            "throughput_tok_s": train_tps,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
        },
        "inference": {
            "gen_tokens": gen_tokens,
            "batch_1": {
                "wall_time_s": b1_time_s,
                "throughput_tok_s": b1_tps,
                "latency_ms_per_token": b1_latency_ms,
            },
            "batch_64": {
                "wall_time_s": b64_time_s,
                "throughput_tok_s": b64_tps,
            },
        },
        "memory": {
            "peak_vram_mb": peak_vram_mb,
        },
        "loss_history": loss_history,
    }

    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PyTorch + Custom Triton Kernels GPT benchmark."
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
        default="results/results_02_triton.json",
        help="Path to write JSON benchmark report.",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--gen-tokens", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    res = run_benchmark(
        golden_dir=args.golden_dir,
        output_json=args.output_json,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        gen_tokens=args.gen_tokens,
        device=args.device,
    )
    print(
        f"[02_triton] train_tps={res['train_tokens_per_sec']:,.1f} tok/s | "
        f"infer_b1={res['infer_b1_tokens_per_sec']:,.1f} tok/s | "
        f"infer_b64={res['infer_b64_tokens_per_sec']:,.1f} tok/s | "
        f"val_loss={res['final_val_loss']:.4f}"
    )
