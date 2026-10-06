"""Training curves for the RoboTwin flow-map study (runs/flowmap_fulljoint/robotwin_*).

1. <out>_lsd_loss.png: LSD grid vs curriculum per optimizer update: unweighted loss,
   off-diagonal (flow-map) residual and diagonal (flow-matching) loss, each as the mean
   over that branch's examples (the logged per-update means include the other branch's
   zeros; the 75/25 split is exact, so they are rescaled by 1/0.25 and 1/0.75).
2. <out>_previews.png: the trainer's fixed-clip previews every 250 updates (4 training
   clips, one per rank, raw weights) at NFE 1/2/4 for all four runs: video PSNR of the
   rollout against the VAE decode of the ground truth, DINO feature MSE, action L1.
   These are TRAINING clips (no held-out split): a generation-quality trace, not validation.

    python scripts/plot_training_curves.py [OUT_PREFIX]   # needs matplotlib (e.g. diff_env)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "runs/flowmap_fulljoint"
RUNS = {"LSD grid": "robotwin_lsd_grid_s42_20260926", "LSD curriculum": "robotwin_lsd_curriculum_s42_20260926",
        "LMD grid": "robotwin_lmd_grid_s42_20260926", "LMD curriculum": "robotwin_lmd_curriculum_s42_20260926"}
STREAMS = ("video", "dino", "pointmap", "action")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # validated categorical slots 1-3


def rows(run, name):
    path = ROOT / run / name
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def ema(values, alpha=0.05):
    out, m = [], None
    for v in values:
        m = v if m is None else (1 - alpha) * m + alpha * v
        out.append(m)
    return out


def style(ax, title, ylabel=None, xlabel="optimizer update"):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9.5)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9.5)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.set_facecolor(SURFACE)


def lsd_loss(out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    panels = [
        ("Unweighted loss (all examples)", lambda m: m["loss_unweighted"]["mean"]),
        ("Off-diagonal flow-map residual", lambda m: sum(m[f"loss_flowmap_{s}"]["mean"] for s in STREAMS) / 0.25),
        ("Diagonal flow-matching loss", lambda m: sum(m[f"loss_diagonal_{s}"]["mean"] for s in STREAMS) / 0.75),
    ]
    for ax, (title, get) in zip(axes, panels):
        for label, color in (("LSD grid", BLUE), ("LSD curriculum", ORANGE)):
            data = rows(RUNS[label], "train_update_metrics.jsonl")
            steps = [r["step"] for r in data]
            values = [get(r["metrics"]) for r in data]
            ax.plot(steps, values, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(steps, ema(values), color=color, linewidth=2, label=label)
            ax.text(steps[-1] + 15, ema(values)[-1], f"{ema(values)[-1]:.3f}", color=INK2, fontsize=8.5,
                    va="center")
        style(ax, title, "loss (sum over streams)" if ax is axes[0] else None)
        ax.set_xlim(0, 2150)
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK)
    fig.text(0.012, 0.015, "RoboTwin LSD, 2,000 updates of 192. Thin: per update; bold: EMA (α = 0.05). "
             "Branch panels are means over that branch's examples (75% diagonal / 25% off-diagonal).\n"
             "The curriculum's off-diagonal jumps widen until update 1000, so its residual is not "
             "comparable to the grid run's early on.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.22, wspace=0.18)
    fig.savefig(f"{out}_lsd_loss.png", dpi=200, facecolor=SURFACE)


def previews(out):
    metrics = [("psnr_rd", "Video PSNR vs VAE decode (dB) ↑"), ("dino_mse", "DINO feature MSE ↓"),
               ("action_l1", "Action L1 ↓")]
    fig, axes = plt.subplots(len(metrics), len(RUNS), figsize=(16, 9.5), facecolor=SURFACE, sharex=True)
    for col, (label, run) in enumerate(RUNS.items()):
        data = rows(run, "preview_metrics.jsonl")
        for row, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            for nfe, color in ((1, BLUE), (2, ORANGE), (4, AQUA)):
                pts = sorted((r["step"], r[key]) for r in data if r["nfe"] == nfe and key in r)
                if pts:
                    ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=4.5, label=f"NFE {nfe}")
            style(ax, label if row == 0 else "", title if col == 0 else None,
                  "optimizer update" if row == len(metrics) - 1 else "")
            ax.set_xlim(-50, 2100)
        # share each metric's y-range across runs so the columns compare directly
    for row in range(len(metrics)):
        lo = min(a.get_ylim()[0] for a in axes[row]); hi = max(a.get_ylim()[1] for a in axes[row])
        for a in axes[row]:
            a.set_ylim(lo, hi)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=INK)
    fig.text(0.012, 0.012, "Trainer previews every 250 updates: 4 fixed RoboTwin training clips (one per task "
             "shown to each rank), raw weights, full joint generation at NFE 1/2/4. Training data, not a "
             "held-out set. LMD runs are still training; their extra point before 1000 is the preview\n"
             "taken when the first 24 h job paused (update 963/964). With 4 clips, single-checkpoint dips are mostly noise.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.95, bottom=0.09, wspace=0.12, hspace=0.18)
    fig.savefig(f"{out}_previews.png", dpi=200, facecolor=SURFACE)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/robotwin"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    lsd_loss(out)
    previews(out)
    print("wrote", f"{out}_lsd_loss.png", f"{out}_previews.png")


if __name__ == "__main__":
    main()
