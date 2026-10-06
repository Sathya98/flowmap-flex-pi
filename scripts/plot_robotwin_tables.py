"""RoboTwin success tables for slides: one PNG per NFE, tasks x checkpoints.

Cells are the task's success rate over 10 episodes (clean scenes, seed 42, same episodes
for every model), tinted by the difference from the FM baseline on that task (blue
better, red worse; no tint when equal). The bottom row is the mean over the 10 tasks
± its standard error over the 100 pooled episodes, sqrt(p(1-p)/100). A dash is an eval
that has not run yet; a mean over fewer than 10 tasks is shown muted with the task count.

    python scripts/plot_robotwin_tables.py [--nfes 1,2,4] [--out docs/figures/robotwin_table]
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_robotwin_eval import EPISODES, FM, RUN, TASKS, rate

GROUPS = [
    ("FM baseline", [("released", FM[1])]),
    ("LSD grid", [(f"step {s}", RUN.format("lsd_grid", f"lsdgrid_s{s}_ema995")) for s in (500, 1000, 2000)]),
    ("LSD curriculum", [(f"step {s}", RUN.format("lsd_curriculum", f"lsdcurr_s{s}_ema995")) for s in (500, 1000, 2000)]),
    ("LMD grid", [(f"step {s}", RUN.format("lmd_grid", f"lmdgrid_s{s}_ema995")) for s in (500, 1000, 1631)]),
    ("LMD curriculum", [(f"step {s}", RUN.format("lmd_curriculum", f"lmdcurr_s{s}_ema995")) for s in (500, 1000, 1636)]),
]
SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#d6d5d0"
BETTER = ("#e3eefc", "#b7d3f6")      # blue ramp, light tints (diff < 0.3, >= 0.3)
WORSE = ("#fbe3e2", "#f4bcbb")       # red, matching tints


def tint(diff):
    if diff is None or abs(diff) < 1e-9:
        return None
    return (BETTER if diff > 0 else WORSE)[abs(diff) >= 0.3 - 1e-9]


def table(nfe, out):
    cols = [(g, label, base) for g, members in GROUPS for label, base in members]
    fm = {t: rate(FM[1], nfe, t) for t in TASKS}
    task_w, col_w, row_h, head_h = 2.3, 0.86, 0.36, 0.95
    width = task_w + col_w * len(cols)
    height = head_h + row_h * (len(TASKS) + 1) + 0.9
    fig = plt.figure(figsize=(width + 0.4, height + 0.55), facecolor=SURFACE)
    ax = fig.add_axes([0.2 / (width + 0.4), 0.02, width / (width + 0.4), (height + 0.35) / (height + 0.55)])
    ax.set_xlim(0, width)
    ax.set_ylim(height, -0.35)
    ax.axis("off")
    ax.text(0, -0.2, f"RoboTwin success, NFE {nfe}", fontsize=15, fontweight="bold", color=INK, va="center")
    # group headers with a rule under each span, then the per-checkpoint subheaders
    x = task_w
    for g, members in GROUPS:
        span = col_w * len(members)
        ax.text(x + span / 2, 0.38, g, ha="center", va="center", fontsize=10.5, fontweight="bold", color=INK)
        ax.plot([x + 0.06, x + span - 0.06], [0.56, 0.56], color=INK2, linewidth=1)
        for i, (label, _) in enumerate(members):
            ax.text(x + col_w * (i + 0.5), 0.78, label, ha="center", va="center", fontsize=8.5, color=INK2)
        x += span
    ax.text(0.05, 0.78, "task", va="center", fontsize=8.5, color=INK2)
    y0 = head_h
    ax.plot([0, width], [y0, y0], color=INK2, linewidth=1)
    for r, task in enumerate(TASKS):
        y = y0 + r * row_h
        ax.text(0.05, y + row_h / 2, task.replace("_", " "), va="center", fontsize=9.5, color=INK)
        for c, (_, _, base) in enumerate(cols):
            v = rate(base, nfe, task)
            cx = task_w + c * col_w
            shade = tint(None if v is None or c == 0 or fm[task] is None else v - fm[task])
            if shade:
                ax.add_patch(Rectangle((cx + 0.03, y + 0.03), col_w - 0.06, row_h - 0.06, facecolor=shade,
                                       edgecolor="none"))
            ax.text(cx + col_w / 2, y + row_h / 2, "–" if v is None else f"{v:.1f}", ha="center", va="center",
                    fontsize=10, color=MUTED if v is None else INK)
        if r < len(TASKS) - 1:
            ax.plot([0, width], [y + row_h, y + row_h], color=RULE, linewidth=0.5)
    # mean row: mean over tasks ± standard error over the pooled episodes
    y = y0 + len(TASKS) * row_h
    ax.plot([0, width], [y, y], color=INK2, linewidth=1)
    ax.text(0.05, y + row_h / 2 + 0.04, "mean ± s.e.", va="center", fontsize=9.5, fontweight="bold", color=INK)
    for c, (_, _, base) in enumerate(cols):
        vals = [v for v in (rate(base, nfe, t) for t in TASKS) if v is not None]
        cx = task_w + c * col_w + col_w / 2
        if not vals:
            ax.text(cx, y + row_h / 2 + 0.04, "–", ha="center", va="center", fontsize=10, color=MUTED)
            continue
        p, n = sum(vals) / len(vals), len(vals) * EPISODES
        se = math.sqrt(p * (1 - p) / n)
        full = len(vals) == len(TASKS)
        ax.text(cx, y + row_h / 2 - 0.03, f"{p:.2f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold" if full else "normal", color=INK if full else MUTED)
        ax.text(cx, y + row_h / 2 + 0.2, f"± {se:.2f}" if full else f"{len(vals)}/10 tasks", ha="center",
                va="center", fontsize=7.5, color=INK2 if full else MUTED)
    # group separators through header and body
    x = task_w
    for _, members in GROUPS[:-1]:
        x += col_w * len(members)
        ax.plot([x, x], [0.62, y + row_h + 0.1], color=RULE, linewidth=0.8)
    ax.plot([task_w, task_w], [0.62, y + row_h + 0.1], color=RULE, linewidth=0.8)
    note = (f"Success rate over {EPISODES} episodes per task (clean scenes, unseen instructions; the same episodes "
            "for every model), full joint generation, EMA 0.995 weights.\n"
            "Tint: better / worse than the FM baseline on that task (darker: by 0.3 or more). One episode is 0.1. "
            "s.e. = sqrt(p(1−p)/100) over the 100 pooled episodes. Dash: not evaluated yet.")
    ax.text(0, y + row_h + 0.42, note, va="top", fontsize=7.5, color=INK2, linespacing=1.5)
    fig.savefig(f"{out}_nfe{nfe}.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return f"{out}_nfe{nfe}.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nfes", default="1,2,4")
    parser.add_argument("--out", default="docs/figures/robotwin_table")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    for nfe in (int(n) for n in args.nfes.split(",")):
        print("wrote", table(nfe, args.out))


if __name__ == "__main__":
    main()
