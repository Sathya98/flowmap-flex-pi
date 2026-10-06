"""RoboTwin simulator success rates: models x NFE (evaluate_results/robotwin/...).

1. <out>_mean.png: success over all tasks and episodes per model and NFE (grouped bars,
   95% Wilson intervals over the pooled episodes).
2. <out>_tasks.png: one panel per NFE; per task and model a violin of the posterior of
   that task's success rate given k successes in n episodes (Beta(k+1, n-k+1), uniform
   prior) with the observed rate as a dot. With 10 binary trials per task, a violin of
   the raw outcomes is only two points; the posterior shows how much each rate can move.

    python scripts/plot_robotwin_eval.py [--set s500] [--out docs/figures/robotwin_eval_s500]
Add a model set to SETS as more checkpoints finish (names are the eval output labels).
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "evaluate_results/robotwin"
TASKS = ("adjust_bottle", "click_bell", "hanging_mug", "move_playingcard_away", "place_a2b_left",
         "place_can_basket", "place_fan", "place_phone_stand", "rotate_qrcode", "stack_blocks_two")
FM = ("FM release (baseline)", "step_048060/fm_release_s48060")
RUN = "flowmap_fulljoint_robotwin_{}_s42_20260926/{}"
SETS = {
    "s500": [FM, ("LSD grid, step 500", RUN.format("lsd_grid", "lsdgrid_s500_ema995")),
             ("LSD curriculum, step 500", RUN.format("lsd_curriculum", "lsdcurr_s500_ema995"))],
    # Final checkpoints (evals queued 2026-09-30: jobs 27397058-60; NFE 1, 2, 4).
    "final": [FM, ("LSD grid, step 2000", RUN.format("lsd_grid", "lsdgrid_s2000_ema995")),
              ("LSD curriculum, step 2000", RUN.format("lsd_curriculum", "lsdcurr_s2000_ema995")),
              ("LMD grid, step 1631", RUN.format("lmd_grid", "lmdgrid_s1631_ema995")),
              ("LMD curriculum, step 1636", RUN.format("lmd_curriculum", "lmdcurr_s1636_ema995"))],
    # Slides: best of each method family at one NFE (use with --grid).
    "slides": [FM, ("LSD curriculum, step 500", RUN.format("lsd_curriculum", "lsdcurr_s500_ema995")),
               ("LSD curriculum, step 2000", RUN.format("lsd_curriculum", "lsdcurr_s2000_ema995")),
               ("LMD grid, step 500", RUN.format("lmd_grid", "lmdgrid_s500_ema995")),
               ("LMD grid, step 1631", RUN.format("lmd_grid", "lmdgrid_s1631_ema995"))],
}
EPISODES = 10
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")  # validated categorical slots 1-5 (adjacent pairs)


def rate(base, nfe, task):
    path = ROOT / f"{base}_nfe{nfe}" / task / "_result_clean.txt"
    return float(path.read_text().split()[-1]) if path.exists() else None


def wilson(k, n, z=1.96):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def beta_pdf(x, a, b):
    return np.exp((a - 1) * np.log(x) + (b - 1) * np.log1p(-x) - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))


def style(ax):
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def mean_figure(models, nfes, out):
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor=SURFACE)
    width = 0.8 / len(models)
    for m, (label, base) in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for i, nfe in enumerate(nfes):
            rates = [rate(base, nfe, t) for t in TASKS]
            if None in rates:
                continue
            k = round(sum(rates) * EPISODES)
            n = EPISODES * len(TASKS)
            l, h = wilson(k, n)
            xs.append(i + (m - (len(models) - 1) / 2) * width)
            ys.append(k / n); lo.append(k / n - l); hi.append(h - k / n)
        bars = ax.bar(xs, ys, width=width * 0.86, color=COLORS[m], label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.2, capsize=3, zorder=3)
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(nfes)), [f"NFE {n}" for n in nfes], fontsize=10.5, color=INK)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("success rate", color=INK2, fontsize=10)
    style(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=len(models),
              labelcolor=INK)
    ax.set_title("RoboTwin success over 10 tasks × 10 episodes (clean scenes)", loc="left", fontsize=13,
                 fontweight="bold", color=INK, pad=12)
    fig.text(0.012, 0.012, "Error bars: 95% Wilson intervals over the 100 pooled episodes. Same seeds for every model;\n"
             "full joint generation, eager BF16; the baseline uses its released sampling schedule.",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.9, bottom=0.22)
    fig.savefig(f"{out}_mean.png", dpi=200, facecolor=SURFACE)


def task_figure(models, nfes, out):
    fig, axes = plt.subplots(len(nfes), 1, figsize=(16, 3.1 * len(nfes) + 1), facecolor=SURFACE, sharex=True)
    grid = np.linspace(1e-4, 1 - 1e-4, 400)
    width = 0.8 / len(models)
    for ax, nfe in zip(np.atleast_1d(axes), nfes):
        for m, (label, base) in enumerate(models):
            for i, task in enumerate(TASKS):
                r = rate(base, nfe, task)
                if r is None:
                    continue
                k = round(r * EPISODES)
                dens = beta_pdf(grid, k + 1, EPISODES - k + 1)
                dens = dens / dens.max() * width * 0.45
                x = i + (m - (len(models) - 1) / 2) * width
                ax.fill_betweenx(grid, x - dens, x + dens, color=COLORS[m], alpha=0.35, linewidth=0)
                ax.plot(x, k / EPISODES, "o", color=COLORS[m], markersize=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, zorder=3, label=label if (i == 0 and ax is np.atleast_1d(axes)[0]) else None)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(f"NFE {nfe}\nsuccess", color=INK, fontsize=10)
        style(ax)
        for i in range(1, len(TASKS)):
            ax.axvline(i - 0.5, color=GRID, linewidth=0.8)
    last = np.atleast_1d(axes)[-1]
    last.set_xticks(range(len(TASKS)), [t.replace("_", " ") for t in TASKS], fontsize=9.5, color=INK, rotation=0)
    last.set_xlim(-0.5, len(TASKS) - 0.5)
    first = np.atleast_1d(axes)[0]
    first.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(models),
                 labelcolor=INK)
    fig.suptitle("Per-task success by NFE", x=0.012, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.995)
    fig.text(0.012, 0.006, "Dot: observed success in 10 episodes. Violin: posterior of the task's success rate given "
             "that count (Beta(k+1, 11−k)); wider = less certain. Clean scenes, same seeds for every model.",
             fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.06, hspace=0.12)
    fig.savefig(f"{out}_tasks.png", dpi=200, facecolor=SURFACE)


def grid_figure(models, nfe, out):
    """One NFE, one panel per task (2 x 5), one violin per model; value under each violin."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6), facecolor=SURFACE, sharey=True)
    grid = np.linspace(1e-4, 1 - 1e-4, 400)
    for ax, task in zip(axes.flat, TASKS):
        labels = []
        for m, (label, base) in enumerate(models):
            r = rate(base, nfe, task)
            labels.append("–" if r is None else f"{r:.1f}")
            if r is None:
                continue
            k = round(r * EPISODES)
            dens = beta_pdf(grid, k + 1, EPISODES - k + 1)
            dens = dens / dens.max() * 0.4
            ax.fill_betweenx(grid, m - dens, m + dens, color=COLORS[m], alpha=0.35, linewidth=0)
            ax.plot(m, k / EPISODES, "o", color=COLORS[m], markersize=7.5, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3, label=label if task == TASKS[0] else None)
        ax.set_xticks(range(len(models)), labels, fontsize=9.5, color=INK)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(-0.03, 1.03)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(task.replace("_", " "), loc="left", fontsize=11, color=INK, pad=6)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate", color=INK2, fontsize=10)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), frameon=False, fontsize=10, loc="upper center",
               bbox_to_anchor=(0.5, 0.935), ncol=len(models), labelcolor=INK)
    fig.suptitle(f"RoboTwin per-task success, NFE {nfe}", x=0.012, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.985)
    fig.text(0.012, 0.012, f"Dot and number: observed success in {EPISODES} episodes. Violin: posterior of the "
             f"task's success rate given that count (Beta(k+1, {EPISODES + 1}−k)); wider = less certain. "
             "Clean scenes, the same episodes for every model.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.08, wspace=0.08, hspace=0.32)
    fig.savefig(f"{out}_nfe{nfe}_grid.png", dpi=200, facecolor=SURFACE)
    return f"{out}_nfe{nfe}_grid.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="s500", choices=sorted(SETS))
    parser.add_argument("--nfes", default="1,2,3,4")
    parser.add_argument("--out", default=None)
    parser.add_argument("--grid", action="store_true", help="per-task panels (2 x 5), one figure per NFE")
    args = parser.parse_args()
    out = args.out or f"docs/figures/robotwin_eval_{args.set}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    nfes = [int(n) for n in args.nfes.split(",")]
    if args.grid:
        for nfe in nfes:
            print("wrote", grid_figure(SETS[args.set], nfe, out))
        return
    mean_figure(SETS[args.set], nfes, out)
    task_figure(SETS[args.set], nfes, out)
    print("wrote", f"{out}_mean.png", f"{out}_tasks.png")


if __name__ == "__main__":
    main()
