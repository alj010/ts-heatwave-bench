"""
scripts/plot_training.py
------------------------
Live loss monitor for LSTM/GRU training.

Usage:
    python scripts/plot_training.py              # watches both lstm and gru
    python scripts/plot_training.py --model lstm
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation


def read_losses(path: Path) -> list[tuple[int, float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
            rows.append((d["epoch"], d["loss"]))
        except (json.JSONDecodeError, KeyError):
            continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "gru", "both"], default="both")
    parser.add_argument("--interval", type=int, default=2000, help="Refresh interval ms")
    parser.add_argument("--save", default=None, help="Save to file instead of showing (disables live mode)")
    args = parser.parse_args()

    models = ["lstm", "gru"] if args.model == "both" else [args.model]
    paths  = {m: Path("results") / f"{m}_loss.jsonl" for m in models}

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Training Loss (live)", fontsize=13)
    colors = {"lstm": "#2196F3", "gru": "#FF9800"}
    lines  = {m: ax.plot([], [], label=m.upper(), color=colors[m], linewidth=2)[0] for m in models}
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_xlim(1, 50)
    ax.set_ylim(0, 0.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    waiting = ax.text(0.5, 0.5, "Waiting for benchmark to start…",
                      transform=ax.transAxes, ha="center", va="center",
                      fontsize=12, color="gray", style="italic")

    def update(_frame):
        any_data   = False
        any_file   = any(p.exists() for p in paths.values())

        for m, line in lines.items():
            data = read_losses(paths[m])
            if data:
                any_data = True
                epochs, losses = zip(*data)
                line.set_data(epochs, losses)

        if any_data:
            waiting.set_visible(False)
            ax.relim()
            ax.autoscale_view()
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=0)
        elif any_file:
            waiting.set_text("Training started — waiting for first epoch…")
            waiting.set_visible(True)
        else:
            waiting.set_text("Waiting for benchmark to start…")
            waiting.set_visible(True)

        return list(lines.values()) + [waiting]

    if args.save:
        update(None)
        plt.tight_layout()
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
    else:
        ani = animation.FuncAnimation(fig, update, interval=args.interval, blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
