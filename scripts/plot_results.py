"""
scripts/plot_results.py
-----------------------
Multi-panel benchmark dashboard.

Usage:
    python scripts/plot_results.py
    python scripts/plot_results.py --file results/benchmark_20260504_123456.json
    python scripts/plot_results.py --save results/dashboard.png
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.grid.axis":     "x",
    "grid.color":         "#e0e0e0",
    "grid.linewidth":     0.8,
    "figure.facecolor":   "#fafafa",
    "axes.facecolor":     "#fafafa",
})

MODEL_ORDER  = ["arima", "xgboost", "lightgbm", "lstm", "gru", "heatwave"]
MODEL_COLORS = {
    "arima":     "#9E9E9E",
    "xgboost":   "#43A047",
    "lightgbm":  "#7CB342",
    "lstm":      "#1E88E5",
    "gru":       "#039BE5",
    "heatwave":  "#E53935",
}
MODEL_LABELS = {k: k.upper() for k in MODEL_COLORS}
MODEL_LABELS["heatwave"] = "HeatWave"


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_latest(results_dir: str) -> list[dict]:
    files = sorted(Path(results_dir).glob("benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"No benchmark_*.json files in {results_dir}")
    path = files[-1]
    print(f"Loading: {path}")
    with open(path) as f:
        return json.load(f)


def sort_results(results: list[dict]) -> list[dict]:
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(results, key=lambda r: order.get(r["model"], 99))


def normalize(values: list[float], higher_is_better: bool = False) -> list[float]:
    """Min-max normalize to [0, 1]; flip if higher is better."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5] * len(values)
    normed = [(v - lo) / (hi - lo) for v in values]
    return [1 - n for n in normed] if not higher_is_better else normed


# ── Panel drawing helpers ──────────────────────────────────────────────────────

def _hbar(ax, labels, values, colors, title, unit="", ref_line=None, ref_label=""):
    bars = ax.barh(labels, values, color=colors, height=0.55, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}{unit}", va="center", ha="left", fontsize=8.5, fontweight="bold")
    if ref_line is not None:
        ax.axvline(ref_line, color="#e53935", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=ref_label)
        ax.legend(fontsize=7.5, framealpha=0.5)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlim(0, max(values) * 1.22)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_xlabel(unit if unit else "", fontsize=8)
    ax.invert_yaxis()


# ── Individual panels ──────────────────────────────────────────────────────────

def panel_accuracy_bars(axes, ok: list[dict]):
    """Three horizontal bar charts: MAE, RMSE, MAPE."""
    labels = [MODEL_LABELS[r["model"]] for r in ok]
    colors = [MODEL_COLORS[r["model"]] for r in ok]
    metrics = [
        ("mae",  "MAE  (lower is better)",  ""),
        ("rmse", "RMSE  (lower is better)", ""),
        ("mape", "MAPE  (lower is better)", "%"),
    ]
    for ax, (key, title, unit) in zip(axes, metrics):
        vals = [r.get(key, 0.0) for r in ok]
        _hbar(ax, labels, vals, colors, title, unit=unit)


def panel_directional_accuracy(ax, ok: list[dict]):
    labels = [MODEL_LABELS[r["model"]] for r in ok]
    colors = [MODEL_COLORS[r["model"]] for r in ok]
    vals   = [r.get("dir_accuracy", 0.0) for r in ok]
    _hbar(ax, labels, vals, colors,
          "Directional Accuracy  (higher is better)",
          unit="%", ref_line=50, ref_label="random (50%)")
    ax.set_xlim(0, 100)


def panel_latency(ax, ok: list[dict]):
    labels = [MODEL_LABELS[r["model"]] for r in ok]
    colors = [MODEL_COLORS[r["model"]] for r in ok]
    vals   = [r.get("mean_latency_ms", 0.0) for r in ok]
    _hbar(ax, labels, vals, colors,
          "Inference Latency  (lower is better)", unit=" ms")


def panel_scatter(ax, ok: list[dict]):
    """Accuracy vs latency — the core tradeoff plot."""
    ax.set_title("Accuracy vs Latency Tradeoff", fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Mean Inference Latency (ms)", fontsize=9)
    ax.set_ylabel("MAE  (lower is better)", fontsize=9)
    ax.grid(True, alpha=0.4)

    for r in ok:
        x   = r.get("mean_latency_ms", 0)
        y   = r.get("mae", 0)
        col = MODEL_COLORS[r["model"]]
        lbl = MODEL_LABELS[r["model"]]
        ax.scatter(x, y, color=col, s=120, zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(7, 3),
                    fontsize=8.5, color=col, fontweight="bold")

    # Ideal direction arrow
    ax.annotate("", xy=(0.08, 0.08), xycoords="axes fraction",
                xytext=(0.3, 0.3),
                arrowprops=dict(arrowstyle="->", color="#999", lw=1.2))
    ax.text(0.05, 0.04, "ideal", transform=ax.transAxes,
            fontsize=7.5, color="#999", style="italic")


def panel_heatmap(ax, ok: list[dict]):
    """Normalised performance heatmap — all metrics, all models at a glance."""
    metrics = [
        ("mae",           False, "MAE"),
        ("rmse",          False, "RMSE"),
        ("mape",          False, "MAPE %"),
        ("dir_accuracy",  True,  "Dir. Acc %"),
        ("mean_latency_ms", False, "Latency ms"),
    ]
    labels  = [MODEL_LABELS[r["model"]] for r in ok]
    metric_labels = [m[2] for m in metrics]

    matrix = []
    for key, higher, _ in metrics:
        vals  = [r.get(key, 0.0) for r in ok]
        matrix.append(normalize(vals, higher_is_better=higher))

    data = np.array(matrix)   # shape: (n_metrics, n_models)

    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=9)
    ax.set_title("Normalized Performance\n(green = best per metric)", fontsize=10,
                 fontweight="bold", pad=8)
    ax.grid(False)
    ax.spines[:].set_visible(False)

    for i in range(len(metric_labels)):
        for j in range(len(labels)):
            val = data[i, j]
            txt_color = "white" if val < 0.25 or val > 0.75 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=txt_color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="0 = worst  |  1 = best")


# ── Main dashboard ─────────────────────────────────────────────────────────────

def print_table(results: list[dict]) -> None:
    ok = sort_results([r for r in results if "error" not in r])
    if not ok:
        return
    col_w = 13
    headers = ["Model", "MAE", "RMSE", "MAPE %", "Dir. Acc %", "Latency ms"]
    sep = "─" * (col_w * len(headers))
    print(f"\n{sep}")
    print("".join(h.ljust(col_w) for h in headers))
    print(sep)
    for r in sorted(ok, key=lambda x: x.get("mae", float("inf"))):
        row = [
            MODEL_LABELS[r["model"]],
            f"{r.get('mae', 0):.4f}",
            f"{r.get('rmse', 0):.4f}",
            f"{r.get('mape', 0):.2f}",
            f"{r.get('dir_accuracy', 0):.1f}",
            f"{r.get('mean_latency_ms', 0):.2f}",
        ]
        print("".join(v.ljust(col_w) for v in row))
    print(f"{sep}\n")


def dashboard(results: list[dict], output_path: Optional[str] = None) -> None:
    ok = sort_results([r for r in results if "error" not in r])
    errored = [r for r in results if "error" in r]
    if errored:
        print("Skipped (errors):", [r["model"] for r in errored])
    if not ok:
        print("No successful results to plot.")
        return

    single = len(ok) == 1

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("ts-heatwave-bench  ·  Model Benchmark Dashboard",
                 fontsize=15, fontweight="bold", y=0.98)

    # Layout: top row = 3 accuracy bars | bottom row = dir acc, latency, scatter, heatmap
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.52, wspace=0.38,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)

    ax_mae  = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_mape = fig.add_subplot(gs[0, 2])
    ax_norm = fig.add_subplot(gs[0, 3])

    ax_dir  = fig.add_subplot(gs[1, 0])
    ax_lat  = fig.add_subplot(gs[1, 1])
    ax_sct  = fig.add_subplot(gs[1, 2])
    ax_leg  = fig.add_subplot(gs[1, 3])

    panel_accuracy_bars([ax_mae, ax_rmse, ax_mape], ok)
    panel_heatmap(ax_norm, ok)
    panel_directional_accuracy(ax_dir, ok)
    panel_latency(ax_lat, ok)

    if not single:
        panel_scatter(ax_sct, ok)
    else:
        ax_sct.set_visible(False)

    # Legend / model key panel
    ax_leg.axis("off")
    patches = [mpatches.Patch(color=MODEL_COLORS[r["model"]],
                               label=MODEL_LABELS[r["model"]]) for r in ok]
    if errored:
        patches += [mpatches.Patch(color="#cccccc", label=f"{r['model'].upper()} (error)")
                    for r in errored]
    ax_leg.legend(handles=patches, title="Models", title_fontsize=10,
                  fontsize=9.5, loc="center", framealpha=0.5,
                  edgecolor="#cccccc", borderpad=1.2, labelspacing=0.8)
    if single:
        ax_leg.text(0.5, 0.15,
                    "Run the full benchmark\nto populate all panels.",
                    transform=ax_leg.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {output_path}")
    else:
        plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=None,     help="Specific results JSON")
    parser.add_argument("--dir",  default="results", help="Results directory")
    parser.add_argument("--save", default=None,     help="Save to file instead of showing")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            results = json.load(f)
    else:
        results = load_latest(args.dir)

    print_table(results)
    dashboard(results, output_path=args.save)


if __name__ == "__main__":
    main()
