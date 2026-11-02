import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.count_sloc import count_all_implementations


def _extract_metric(record: dict, key: str, default: float = 0.0) -> float:
    if key in record and record[key] is not None:
        return float(record[key])
    # Check nested sections if present
    for section in ("training", "inference", "memory", "metrics"):
        if isinstance(record.get(section), dict) and key in record[section]:
            return float(record[section][key])
    return default


def aggregate_all(
    results_dir: str,
    output_dir: str,
    implementations_dir: str = "implementations",
) -> dict:
    """Aggregate benchmark JSON results with SLOC metrics, generating tables and plots."""
    res_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sloc_map = count_all_implementations(implementations_dir)

    records: list[dict] = []
    if res_path.exists():
        for json_file in sorted(res_path.glob("results_*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            impl_name = data.get(
                "implementation",
                json_file.stem.replace("results_", ""),
            )
            sloc_info = sloc_map.get(
                impl_name,
                {
                    "sloc_core": data.get("sloc_core", 0),
                    "sloc_harness": data.get("sloc_harness", 0),
                    "sloc_total": data.get("sloc_total", 0),
                },
            )
            merged = {
                **data,
                "implementation": impl_name,
                "sloc_core": sloc_info.get("sloc_core", data.get("sloc_core", 0)),
                "sloc_harness": sloc_info.get(
                    "sloc_harness", data.get("sloc_harness", 0)
                ),
                "sloc_total": sloc_info.get("sloc_total", data.get("sloc_total", 0)),
            }
            records.append(merged)

    # Generate Markdown table
    md_lines = [
        "# Benchmark Summary Table",
        "",
        "| Implementation | GPU | Core SLOC | Total SLOC | Train Time (s) | Train Tok/s | Infer B=1 Tok/s | Infer B=64 Tok/s | Peak VRAM (MB) | Final Val Loss |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in records:
        impl = r["implementation"]
        gpu = r.get("gpu_name", "GPU")
        s_core = r["sloc_core"]
        s_tot = r["sloc_total"]
        t_time = _extract_metric(r, "train_time_s")
        t_tps = _extract_metric(r, "train_tokens_per_sec")
        ib1_tps = _extract_metric(r, "infer_b1_tokens_per_sec")
        ib64_tps = _extract_metric(r, "infer_b64_tokens_per_sec")
        vram = _extract_metric(r, "peak_vram_mb")
        val_loss = _extract_metric(r, "final_val_loss")
        md_lines.append(
            f"| `{impl}` | {gpu} | {s_core} | {s_tot} | {t_time:.2f} | {t_tps:,.1f} | {ib1_tps:,.1f} | {ib64_tps:,.1f} | {vram:.1f} | {val_loss:.4f} |"
        )

    (out_path / "summary_table.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    # Generate LaTeX table
    tex_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Comparison of GPU Programming Paradigms for Transformer Training and Inference}",
        "\\label{tab:benchmark_summary}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\hline",
        "Implementation & Core SLOC & Total SLOC & Train (s) & Train (tok/s) & Infer B=1 (tok/s) & Infer B=64 (tok/s) & VRAM (MB) & Val Loss \\\\",
        "\\hline",
    ]
    for r in records:
        impl_tex = r["implementation"].replace("_", "\\_")
        s_core = r["sloc_core"]
        s_tot = r["sloc_total"]
        t_time = _extract_metric(r, "train_time_s")
        t_tps = _extract_metric(r, "train_tokens_per_sec")
        ib1_tps = _extract_metric(r, "infer_b1_tokens_per_sec")
        ib64_tps = _extract_metric(r, "infer_b64_tokens_per_sec")
        vram = _extract_metric(r, "peak_vram_mb")
        val_loss = _extract_metric(r, "final_val_loss")
        tex_lines.append(
            f"{impl_tex} & {s_core} & {s_tot} & {t_time:.2f} & {t_tps:.1f} & {ib1_tps:.1f} & {ib64_tps:.1f} & {vram:.1f} & {val_loss:.4f} \\\\"
        )
    tex_lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    (out_path / "summary_table.tex").write_text(
        "\n".join(tex_lines) + "\n", encoding="utf-8"
    )

    # Plot 1: Throughput comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = [r["implementation"] for r in records] or ["none"]
    train_tps = [_extract_metric(r, "train_tokens_per_sec") for r in records] or [0.0]
    b1_tps = [_extract_metric(r, "infer_b1_tokens_per_sec") for r in records] or [0.0]
    b64_tps = [_extract_metric(r, "infer_b64_tokens_per_sec") for r in records] or [0.0]
    colors = ["#ff6b8b", "#4ecdc4", "#45b7d1"][: len(names)]

    axes[0].bar(names, train_tps, color=colors)
    axes[0].set_title("Training Throughput (tokens/s)")
    axes[0].set_ylabel("Tokens / sec")
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    axes[1].bar(names, b1_tps, color=colors)
    axes[1].set_title("Inference B=1 Throughput (tokens/s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    axes[2].bar(names, b64_tps, color=colors)
    axes[2].set_title("Inference B=64 Throughput (tokens/s)")
    axes[2].grid(axis="y", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_path / "throughput_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 2: Pareto SLOC vs Training Throughput
    fig, ax = plt.subplots(figsize=(7, 5))
    for idx, r in enumerate(records):
        x_val = r["sloc_core"]
        y_val = _extract_metric(r, "train_tokens_per_sec")
        ax.scatter(x_val, y_val, s=120, label=r["implementation"])
        ax.annotate(
            r["implementation"],
            (x_val, y_val),
            textcoords="offset points",
            xytext=(8, 6),
        )
    ax.set_xlabel("Core Implementation Complexity (SLOC)")
    ax.set_ylabel("Training Throughput (tokens/s)")
    ax.set_title("Complexity (SLOC) vs. Training Throughput")
    ax.grid(True, linestyle="--", alpha=0.5)
    if records:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "pareto_sloc_vs_throughput.png", dpi=150)
    plt.close(fig)

    # Plot 3: Loss Convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in records:
        history = r.get("loss_history", [])
        if history:
            steps = [pt["step"] for pt in history]
            val_losses = [pt.get("val_loss", pt.get("train_loss", 0.0)) for pt in history]
            ax.plot(steps, val_losses, marker="o", label=f"{r['implementation']} (val)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Validation Loss Convergence Parity")
    ax.grid(True, linestyle="--", alpha=0.5)
    if records:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "loss_convergence.png", dpi=150)
    plt.close(fig)

    summary = {
        "implementations": records,
        "sloc": sloc_map,
        "artifacts": {
            "markdown_table": str(out_path / "summary_table.md"),
            "latex_table": str(out_path / "summary_table.tex"),
            "throughput_chart": str(out_path / "throughput_comparison.png"),
            "pareto_chart": str(out_path / "pareto_sloc_vs_throughput.png"),
            "loss_chart": str(out_path / "loss_convergence.png"),
        },
    }
    (out_path / "aggregated_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark JSON outputs and SLOC counts into tables and plots."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing results_*.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/summary",
        help="Directory to write tables and figures.",
    )
    args = parser.parse_args()
    summary = aggregate_all(args.results_dir, args.output_dir)
    print(f"Aggregated {len(summary['implementations'])} implementation results into {args.output_dir}")
