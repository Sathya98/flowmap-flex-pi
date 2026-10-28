"""RoboTwin 50-task evaluation (50 clean + 50 randomized episodes per task): FM release against
LMD grid 1631 and LMD curriculum 1636 (EMA 0.995) at NFE 1, 2 and 4.

1. <out>_table.png: success per task, clean and randomized side by side, the overall rates on top.
   A cell is the success rate over the task's finished episodes; star: fewer than 40 finished,
   dash: none yet. Flow-map cells are tinted by their difference from FM at the same NFE and setting.
2. <out>_nfe_lines.png: overall success against NFE, one panel each for all episodes, clean and
   randomized, with 95% intervals from a bootstrap over tasks.

Reads the per-episode files (<run>/<task>/_episodes_{clean,random}/NNN.json), so unfinished runs
are included as far as they got.

    python scripts/plot_robotwin_t50.py [--out docs/figures/robotwin_t50]
"""
import argparse
import json
import os
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from plot_robotwin_eval import COLORS, INK, INK2, ROOT, SURFACE, style

MODELS = [("FM", "FM release (step 48060)", "step_048060/fm_release_s48060_t50e50"),
          ("Grid", "LMD grid (step 1631)", "flowmap_fulljoint_robotwin_lmd_grid_s42_20260926/lmdgrid_s1631_ema995_t50e50"),
          ("Curr.", "LMD curriculum (step 1636)",
           "flowmap_fulljoint_robotwin_lmd_curriculum_s42_20260926/lmdcurr_s1636_ema995_t50e50")]
NFES = (1, 2, 4)
PHASES = (("clean", "Clean scenes"), ("random", "Randomized scenes"))
EPISODES = 50
STAR_BELOW = 40
MUTED, RULE = "#8a8985", "#d6d5d0"
BETTER = ("#e3eefc", "#b7d3f6")      # flow map ahead of FM by >= 10 / >= 30 points
WORSE = ("#fbe3e2", "#f4bcbb")


def load():
    """{(model, nfe, phase): {task: [success, ...]}} over the finished episodes."""
    runs, tasks = {}, set()
    for m, _, base in MODELS:
        for nfe in NFES:
            run = ROOT / f"{base}_nfe{nfe}"
            for phase, _ in PHASES:
                per = runs.setdefault((m, nfe, phase), {})
                for task in (os.listdir(run) if run.is_dir() else []):
                    ep = run / task / f"_episodes_{phase}"
                    if not ep.is_dir():
                        continue
                    tasks.add(task)
                    per[task] = [int(json.loads((ep / f).read_text())["success"])
                                 for f in sorted(os.listdir(ep)) if f.endswith(".json")]
    return runs, sorted(tasks)


def counts(runs, m, nfe, phases, tasks):
    """Successes and finished episodes per task, summed over the given phases."""
    k = np.array([sum(sum(runs[m, nfe, p].get(t, [])) for p in phases) for t in tasks], float)
    n = np.array([sum(len(runs[m, nfe, p].get(t, [])) for p in phases) for t in tasks], float)
    return k, n


def overall(runs, m, nfe, phases, tasks, draws=2000):
    """Pooled success (%), its 95% bootstrap interval over tasks, and the episode count."""
    k, n = counts(runs, m, nfe, phases, tasks)
    if n.sum() == 0:
        return None
    pick = np.random.default_rng(0).integers(0, len(tasks), (draws, len(tasks)))
    boot = k[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100 * k.sum() / n.sum(), 100 * lo, 100 * hi, int(n.sum())


def tint(diff):
    if diff is None or abs(diff) < 10 - 1e-9:
        return None
    return (BETTER if diff > 0 else WORSE)[abs(diff) >= 30 - 1e-9]


def table(runs, tasks, out):
    cols = [(phase, nfe, m) for phase, _ in PHASES for nfe in NFES for m, _, _ in MODELS]
    task_w, col_w, gap, row_h = 2.35, 0.56, 0.22, 0.3
    group_w = col_w * len(NFES) * len(MODELS)
    width = task_w + 2 * group_w + gap
    head_h, over_h = 1.25, 0.95
    body0 = head_h + over_h
    height = body0 + row_h * len(tasks) + 1.0
    fig = plt.figure(figsize=(width + 0.4, height + 0.55), facecolor=SURFACE)
    ax = fig.add_axes([0.2 / (width + 0.4), 0.01, width / (width + 0.4), (height + 0.35) / (height + 0.55)])
    ax.set_xlim(0, width)
    ax.set_ylim(height, -0.35)
    ax.axis("off")

    def col_x(c):
        return task_w + c * col_w + (gap if c >= len(cols) // 2 else 0)

    ax.text(0, -0.18, "RoboTwin 2.0, 50 tasks: success rate (%)", fontsize=15, fontweight="bold", color=INK,
            va="center")
    ax.text(width, -0.18, f"as of {date.today():%Y-%m-%d}", fontsize=9, color=INK2, va="center", ha="right")
    for g, (_, label) in enumerate(PHASES):
        x0 = col_x(g * len(cols) // 2)
        ax.text(x0 + group_w / 2, 0.3, label, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)
        ax.plot([x0 + 0.06, x0 + group_w - 0.06], [0.48, 0.48], color=INK2, linewidth=1)
        for i, nfe in enumerate(NFES):
            xs = x0 + i * col_w * len(MODELS)
            ax.text(xs + col_w * len(MODELS) / 2, 0.68, f"NFE {nfe}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=INK)
            ax.plot([xs + 0.08, xs + col_w * len(MODELS) - 0.08], [0.82, 0.82], color=RULE, linewidth=1)
    for c, (_, _, m) in enumerate(cols):
        ax.text(col_x(c) + col_w / 2, 1.03, m, ha="center", va="center", fontsize=8.5, color=INK2)
    ax.text(0.05, 1.03, "task", va="center", fontsize=8.5, color=INK2)
    ax.plot([0, width], [head_h, head_h], color=INK2, linewidth=1)

    # overall rows: pooled success with the half-width of its 95% interval, then the episode count
    ax.text(0.05, head_h + 0.3, "All 50 tasks", va="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(0.05, head_h + 0.52, "± 95% interval (half-width)", va="center", fontsize=7, color=INK2)
    ax.text(0.05, head_h + 0.77, "episodes finished", va="center", fontsize=8, color=INK2)
    for c, (phase, nfe, m) in enumerate(cols):
        cx = col_x(c) + col_w / 2
        r = overall(runs, m, nfe, (phase,), tasks)
        if r is None:
            ax.text(cx, head_h + 0.32, "–", ha="center", va="center", fontsize=10, color=MUTED)
            continue
        p, lo, hi, n = r
        star = n < STAR_BELOW * len(tasks)
        ax.text(cx, head_h + 0.3, f"{p:.1f}" + ("*" if star else ""), ha="center", va="center", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(cx, head_h + 0.52, f"± {(hi - lo) / 2:.1f}", ha="center", va="center", fontsize=7, color=INK2)
        ax.text(cx, head_h + 0.77, f"{n:,}", ha="center", va="center", fontsize=7.5,
                color=INK2 if n >= EPISODES * len(tasks) - 5 else "#b0493a")
    ax.plot([0, width], [body0, body0], color=INK2, linewidth=1)

    for r, task in enumerate(tasks):
        y = body0 + r * row_h
        ax.text(0.05, y + row_h / 2, task.replace("_", " "), va="center", fontsize=9, color=INK)
        for c, (phase, nfe, m) in enumerate(cols):
            ep = runs[m, nfe, phase].get(task, [])
            cx = col_x(c)
            if not ep:
                ax.text(cx + col_w / 2, y + row_h / 2, "–", ha="center", va="center", fontsize=9, color=MUTED)
                continue
            v = 100 * sum(ep) / len(ep)
            fm = runs["FM", nfe, phase].get(task, [])
            shade = tint(None if m == "FM" or not fm else v - 100 * sum(fm) / len(fm))
            if shade:
                ax.add_patch(Rectangle((cx + 0.025, y + 0.025), col_w - 0.05, row_h - 0.05, facecolor=shade,
                                       edgecolor="none"))
            ax.text(cx + col_w / 2, y + row_h / 2, f"{v:.0f}" + ("*" if len(ep) < STAR_BELOW else ""),
                    ha="center", va="center", fontsize=9, color=INK)
        if r < len(tasks) - 1:
            ax.plot([0, width], [y + row_h, y + row_h], color=RULE, linewidth=0.5)
    end = body0 + len(tasks) * row_h
    ax.plot([0, width], [end, end], color=INK2, linewidth=1)
    # separators between NFE groups (thin) and in front of each setting (stronger)
    for g in range(2):
        x0 = col_x(g * len(cols) // 2)
        ax.plot([x0 - 0.01, x0 - 0.01], [0.55, end], color=INK2 if g else RULE, linewidth=0.9 if g else 0.8)
        for i in (1, 2):
            xs = x0 + i * col_w * len(MODELS)
            ax.plot([xs, xs], [0.88, end], color=RULE, linewidth=0.8)
    names = "; ".join(f"{m} = {label}" for m, label, _ in MODELS)
    note = (f"Success rate over each task's finished episodes (of {EPISODES} per setting; unseen instructions, full "
            "joint generation, flow maps at EMA 0.995). The same seeds for every model (shared expert cache).\n"
            f"{names}. Star: fewer than {STAR_BELOW} episodes finished "
            f"(overall: fewer than {STAR_BELOW * len(tasks):,}); dash: none. One episode is 2 points.\n"
            "Tint: flow map better (blue) / worse (red) than FM on that task at the same NFE and setting, by 10 points "
            "or more (darker: 30 or more). Interval: bootstrap over tasks, 2,000 resamples.\n"
            "Episode counts in red: run not finished. Three randomized episodes (place shoe 30, click bell 32, place burger "
            "fries 1) use a replacement seed, the same for every model: their original scenes hung SAPIEN's renderer.")
    ax.text(0, end + 0.22, note, va="top", fontsize=7.5, color=INK2, linespacing=1.55)
    path = f"{out}_table.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def lines(runs, tasks, out):
    panels = [(("clean", "random"), "Clean + randomized"), (("clean",), "Clean scenes"),
              (("random",), "Randomized scenes")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), facecolor=SURFACE, sharey=True)
    dodge = 0.07
    partial = []
    for ax, (phases, title) in zip(axes, panels):
        for i, (m, label, _) in enumerate(MODELS):
            pts = [(j, overall(runs, m, nfe, phases, tasks)) for j, nfe in enumerate(NFES)]
            pts = [(j, r) for j, r in pts if r is not None]
            xs = [j + (i - 1) * dodge for j, _ in pts]
            ys = [r[0] for _, r in pts]
            err = [[r[0] - r[1] for _, r in pts], [r[2] - r[0] for _, r in pts]]
            ax.errorbar(xs, ys, yerr=err, color=COLORS[i], linewidth=2, elinewidth=1.2, capsize=3, marker="none",
                        zorder=3)
            full = EPISODES * len(phases) * len(tasks)
            for x, (_, r) in zip(xs, pts):
                done = r[3] >= full - 10
                ax.plot(x, r[0], "o", markersize=7, color=COLORS[i] if done else SURFACE,
                        markeredgecolor=COLORS[i] if not done else SURFACE, markeredgewidth=1.6, zorder=4)
                if not done and phases == ("clean", "random"):
                    partial.append(f"{label}, NFE {NFES[int(round(x))]}: {r[3]:,} of {full:,} episodes")
        ax.set_title(f"{title} ({EPISODES * len(phases)} episodes per task)", loc="left", fontsize=11.5,
                     color=INK, pad=8)
        ax.set_xticks(range(len(NFES)), [f"NFE {n}" for n in NFES], fontsize=10, color=INK)
        ax.set_xlim(-0.3, len(NFES) - 0.7)
        ax.set_ylim(55, 100)
        style(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("success (%)", color=INK2, fontsize=10.5)
    handles = [Line2D([], [], color=COLORS[i], linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                      markeredgewidth=1.2) for i in range(len(MODELS))]
    fig.legend(handles, [label for _, label, _ in MODELS], frameon=False, fontsize=10.5, loc="upper right",
               ncol=3, bbox_to_anchor=(0.995, 0.985), labelcolor=INK, handlelength=2.2, columnspacing=1.8)
    fig.suptitle("RoboTwin 2.0, 50 tasks: success vs NFE", x=0.012, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.97)
    note = ("Point: success over all finished episodes; bars: 95% interval from a bootstrap over the 50 tasks "
            "(2,000 resamples). Same seeds for every model.")
    if partial:
        note += " Hollow: run not finished (" + "; ".join(partial) + ")."
    fig.text(0.012, 0.025, note, fontsize=8.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.8, bottom=0.14, wspace=0.08)
    path = f"{out}_nfe_lines.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="docs/figures/robotwin_t50")
    args = parser.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    runs, tasks = load()
    for m, _, _ in MODELS:
        for nfe in NFES:
            r = [overall(runs, m, nfe, ph, tasks) for ph in (("clean", "random"), ("clean",), ("random",))]
            print(f"{m:5s} NFE {nfe}: " + "  ".join("–" if x is None else f"{x[0]:.1f} [{x[1]:.1f}, {x[2]:.1f}] n={x[3]}"
                                                    for x in r))
    print("wrote", table(runs, tasks, args.out))
    print("wrote", lines(runs, tasks, args.out))


if __name__ == "__main__":
    main()
