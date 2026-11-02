import importlib
import json
import os
import numpy as np
import torch
from common.generate_golden import generate_golden_artifacts


def test_pytorch_eager_golden_parity(tmp_path):
    meta = generate_golden_artifacts(
        str(tmp_path), num_steps=3, eval_iters=2, seed=998244353
    )
    batches = np.load(tmp_path / "golden_batches.npz")
    mod = importlib.import_module("implementations.01_pytorch.model")
    cfg = mod.GPTConfig()
    model = mod.PyTorchGPT(cfg)
    model.load_golden_weights(str(tmp_path / "golden_weights.npz"))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    x0 = torch.from_numpy(batches["train_x"][0]).long()
    y0 = torch.from_numpy(batches["train_y"][0]).long()
    _, loss0 = model(x0, y0)
    assert abs(loss0.item() - meta["loss_step_0"]) < 1e-5

    optimizer.zero_grad(set_to_none=True)
    loss0.backward()
    optimizer.step()

    x1 = torch.from_numpy(batches["train_x"][1]).long()
    y1 = torch.from_numpy(batches["train_y"][1]).long()
    with torch.no_grad():
        _, loss1 = model(x1, y1)
    assert abs(loss1.item() - meta["loss_step_1"]) < 1e-5

    # Test deterministic generation
    prompt = torch.zeros((2, 1), dtype=torch.long)
    out = model.generate(prompt, max_new_tokens=5, deterministic=True)
    assert out.shape == (2, 6)
    assert torch.equal(out[0], out[1])


def test_pytorch_benchmark_end_to_end(tmp_path):
    golden_dir = tmp_path / "golden"
    generate_golden_artifacts(
        str(golden_dir), num_steps=4, eval_iters=2, seed=998244353
    )
    output_json = tmp_path / "results_01_pytorch.json"

    bench_mod = importlib.import_module("implementations.01_pytorch.benchmark")
    report = bench_mod.run_benchmark(
        golden_dir=str(golden_dir),
        output_json=str(output_json),
        max_iters=4,
        warmup_iters=2,
        eval_interval=2,
        eval_iters=2,
        gen_tokens=8,
        device="cpu",
    )

    assert os.path.exists(output_json)
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["implementation"] == "01_pytorch"
    assert loaded["train_time_s"] > 0.0
    assert loaded["train_tokens_per_sec"] > 0.0
    assert loaded["infer_b1_time_s"] > 0.0
    assert loaded["infer_b1_tokens_per_sec"] > 0.0
    assert loaded["infer_b64_time_s"] > 0.0
    assert loaded["infer_b64_tokens_per_sec"] > 0.0
    assert loaded["sloc_core"] > 0
    assert loaded["sloc_harness"] > 0
    assert loaded["sloc_total"] == loaded["sloc_core"] + loaded["sloc_harness"]
    assert "training" in loaded
    assert "inference" in loaded
    assert "batch_1" in loaded["inference"]
    assert "batch_64" in loaded["inference"]
    assert len(loaded["loss_history"]) >= 2
    assert abs(loaded["sanity_check"]["loss_step_0_diff"]) < 1e-5
    assert abs(loaded["sanity_check"]["loss_step_1_diff"]) < 1e-5
    assert report == loaded
