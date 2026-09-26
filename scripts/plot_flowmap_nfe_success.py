"""Plot LIBERO success vs NFE for the FM teacher and a flow-map student.

Reads the validated comparison summary written for the LMD pilot
(`comparison/step_50_summary.json`) and writes a PNG + PDF beside it.
Error bars are Wilson 95% intervals; the footnote gives exact McNemar
p-values from the paired (same initial state / seed) episode outcomes.

    python scripts/plot_flowmap_nfe_success.py \
        runs/flowmap_fulljoint/libero_lmd_100_s42_20260918/comparison/step_50_summary.json
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
STYLE = {  # model -> (label, color, marker, x-dodge factor)
    "fm": ("FM teacher (released)", "#2a78d6", "o", 0.94),
    "lmd": ("LMD student", "#eb6834", "s", 1.06),
}
SUITES = [
    ("libero_spatial", "Spatial"),
    ("libero_object", "Object"),
    ("libero_goal", "Goal"),
    ("libero_10", "Long"),
]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (mid - half), 100 * (mid + half)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def draw(ax, rows, get_kn, nfes, ylim, title, direct_labels=False):
    ax.set_facecolor(SURFACE)
    for model, (label, color, marker, dodge) in STYLE.items():
        pts = sorted((r["nfe"], *get_kn(r)) for r in rows if r["model"] == model)
        xs = [n * dodge for n, _, _ in pts]
        ys = [100 * k / tot for _, k, tot in pts]
        lo, hi = zip(*(wilson(k, tot) for _, k, tot in pts))
        ax.errorbar(
            xs, ys, yerr=[[max(0, y - l) for y, l in zip(ys, lo)], [max(0, h - y) for y, h in zip(ys, hi)]],
            color=color, lw=2, marker=marker, ms=8, mec=SURFACE, mew=2,
            elinewidth=1.2, capsize=0, label=label, zorder=3,
        )
        if direct_labels:
            ax.annotate(
                label.split(" (")[0], (xs[-1], ys[-1]), xytext=(10, 0),
                textcoords="offset points", va="center", fontsize=9, color=INK_2,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(nfes, [str(n) for n in nfes])
    ax.minorticks_off()
    ax.set_xlim(nfes[0] / 1.35, nfes[-1] * (2.2 if direct_labels else 1.35))
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=10 if not direct_labels else 11, color=INK, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--out-stem", default=None)
    args = ap.parse_args()
    d = json.loads(args.summary.read_text())
    rows = d["results"]
    nfes = sorted({r["nfe"] for r in rows})

    fig = plt.figure(figsize=(12, 4.8), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.6, 0.08, 1, 1], hspace=0.55, wspace=0.3)
    ax = fig.add_subplot(gs[:, 0])
    draw(ax, rows, lambda r: (r["successes"], r["episodes"]), nfes, (94, 100.4),
         "All 4 suites (400 episodes / point)", direct_labels=True)
    ax.set_xlabel("Denoising steps (NFE)", color=INK_2, fontsize=9)
    ax.set_ylabel("Task success (%)", color=INK_2, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK)

    for i, (key, name) in enumerate(SUITES):
        sax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        draw(sax, rows, lambda r, k=key: (r["suites"][k]["success"], r["suites"][k]["total"]),
             nfes, (86, 100.8), f"{name} (100 / point)")
        if i >= 2:
            sax.set_xlabel("NFE", color=INK_2, fontsize=8.5)

    pvals = {
        (p["fm_nfe"], p["lmd_nfe"]): (p["lmd_only_success"], p["fm_only_success"],
                                       mcnemar_exact(p["lmd_only_success"], p["fm_only_success"]))
        for p in d["paired_comparisons"]
    }
    ptxt = "; ".join(
        f"LMD@{l} vs FM@{f} {b}/{c}, p={p:.2f}"
        for (f, l), (b, c, p) in sorted(pvals.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    )
    fig.suptitle(
        f"LIBERO: FM teacher vs LMD flow-map student (step {d['checkpoint_step']}, "
        f"{d['student_weights']} weights), full-joint generation",
        x=0.06, ha="left", fontsize=12.5, color=INK, y=0.99,
    )
    fig.text(
        0.06, 0.015,
        f"{d['trials_per_task']} trials × {d['tasks']} tasks per point, seed 42, same initial states. "
        "Bars: Wilson 95% CI.\nPaired exact McNemar on discordant episodes (LMD-only / FM-only successes): " + ptxt + ".",
        fontsize=7.5, color=INK_2, va="bottom", linespacing=1.5,
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.2)

    stem = args.out_stem or str(args.summary.with_name(args.summary.stem.replace("summary", "nfe_success")))
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=180, facecolor=SURFACE)
    print(f"wrote {stem}.png / .pdf")


if __name__ == "__main__":
    main()
