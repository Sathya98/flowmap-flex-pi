"""RoboTwin per-task success vs NFE: one panel per task (2 x 5), one line per checkpoint.

Each point is the task's success over 10 episodes at that NFE, with a 95% Wilson
interval; lines are dodged sideways a little so overlapping points (many tasks sit at
1.0) stay visible. Checkpoints come from a SETS entry in plot_robotwin_eval.py.

    python scripts/plot_robotwin_nfe_lines.py [--set slides] [--nfes 1,2,4] [--out docs/figures/robotwin_nfe_lines]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_robotwin_eval import COLORS, EPISODES, INK, INK2, SETS, SURFACE, TASKS, rate, style, wilson


def figure(models, nfes, out):
    fig, axes = plt.subplots(2, 5, figsize=(17, 7.4), facecolor=SURFACE, sharex=True, sharey=True)
    dodge = 0.1
    for ax, task in zip(axes.flat, TASKS):
        for m, (label, base) in enumerate(models):
            xs, ys, lo, hi = [], [], [], []
            for i, nfe in enumerate(nfes):
                r = rate(base, nfe, task)
                if r is None:
                    continue
                k = round(r * EPISODES)
                l, h = wilson(k, EPISODES)
                xs.append(i + (m - (len(models) - 1) / 2) * dodge)
                ys.append(k / EPISODES); lo.append(k / EPISODES - l); hi.append(h - k / EPISODES)
            ax.errorbar(xs, ys, yerr=[lo, hi], color=COLORS[m], linewidth=2, elinewidth=1.1, capsize=2.5,
                        marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                        label=label if task == TASKS[0] else None, zorder=3)
        ax.set_title(task.replace("_", " "), loc="left", fontsize=11, color=INK, pad=6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_yticks([0, 0.5, 1])
        style(ax)
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("success rate", color=INK2, fontsize=10)
    for ax in axes[1]:
        ax.set_xticks(range(len(nfes)), [f"NFE {n}" for n in nfes], fontsize=9.5, color=INK)
        ax.set_xlim(-0.4, len(nfes) - 0.6)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), frameon=False, fontsize=10, loc="center left",
               bbox_to_anchor=(0.835, 0.55), labelcolor=INK, handlelength=2.2, labelspacing=1.1)
    fig.suptitle("RoboTwin per-task success vs NFE", x=0.012, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.985)
    fig.text(0.012, 0.012, f"Point: success in {EPISODES} episodes at that NFE; bars: 95% Wilson interval. "
             "Lines are shifted sideways slightly so overlapping points stay visible. "
             "Clean scenes, the same episodes for every model.", fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.045, right=0.825, top=0.9, bottom=0.09, wspace=0.08, hspace=0.3)
    fig.savefig(f"{out}.png", dpi=200, facecolor=SURFACE)
    return f"{out}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="slides", choices=sorted(SETS))
    parser.add_argument("--nfes", default="1,2,4")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or f"docs/figures/robotwin_nfe_lines_{args.set}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    print("wrote", figure(SETS[args.set], [int(n) for n in args.nfes.split(",")], out))


if __name__ == "__main__":
    main()
