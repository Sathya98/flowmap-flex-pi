"""Presentation figure: time per optimizer update (192 examples, 4xH100) as each efficiency
change landed, for LMD and LSD side by side. Numbers from .claude/context/07-efficiency-notes.md
§12 (sources per row below). Estimated rows (derived from measured parts) are hatched.

    python scripts/plot_update_timeline.py [OUT_PREFIX]   # needs matplotlib (e.g. diff_env)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, seconds per update, estimated?)
LMD = [
    ("Baseline: raw video decode + frozen encoders, DeepSpeed 0.18.5", 785, False),  # job 26887233
    ("+ latent cache of encoder outputs", 774, True),        # 16.0 s/microstep, job 27176766
    ("+ fix DeepSpeed ZeRO-2 hook bug", 153, False),          # 3.00 s/microstep, job 27180109
    ("+ fused attention-JVP kernel, microbatch 2", 84, False),  # 3.11 s per 2, job 27198562
    ("+ background EMA", 72, False),                          # 71.7-73.6 s, job 27202122
]
LSD = [
    ("Baseline: raw video decode + frozen encoders, DeepSpeed 0.18.5", 732, True),  # cost model §5
    ("+ latent cache, fix ZeRO-2 hook bug", 110, True),       # branch times, job 27183027
    ("+ stratified diagonal/off-diagonal mask", 67, False),   # 1.21 s mean, job 27183027
    ("+ fused attention-JVP kernel, batched loss, microbatch 2", 39, False),  # job 27199500
    ("+ background EMA", 30, False),                          # 29.7-30.3 s, job 27205496
]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"        # validated categorical slots 1-2 (dataviz palette)


def fmt(seconds):
    return f"{seconds / 60:.1f} min" if seconds >= 120 else f"{seconds:.0f} s"


def panel(ax, rows, color, title):
    base = rows[0][1]
    n = len(rows)
    xmax = 1100       # bars end by ~800; the multiplier column sits at the right edge
    for i, (label, sec, est) in enumerate(rows):
        y = n - 1 - i
        ax.barh(y, sec, height=0.42, color=color if not est else "none", alpha=1.0,
                edgecolor=color, hatch="////" if est else None, linewidth=1.2 if est else 0)
        ax.text(0, y + 0.36, label, ha="left", va="bottom", fontsize=10.5, color=INK2)
        mult = base / sec
        mult_txt = "1×" if i == 0 else f"{mult:.1f}×" if mult < 10 else f"{mult:.0f}×"
        ax.text(sec + 12, y, f"{fmt(sec)}{' (est.)' if est else ''}", ha="left", va="center",
                fontsize=10.5, color=INK2)
        ax.text(xmax - 5, y, mult_txt, ha="right", va="center", fontsize=13,
                fontweight="bold", color=INK)
    final = base / rows[-1][1]
    ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK, va="bottom")
    ax.text(0, 1.025, f"{fmt(base)}  →  {fmt(rows[-1][1])} per update:  {final:.0f}× faster",
            transform=ax.transAxes, fontsize=11.5, color=INK2, va="bottom")
    ax.text(xmax - 5, n - 0.35, "vs. baseline", ha="right", va="bottom", fontsize=9, color=INK2)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.5, n - 0.1)
    ax.set_yticks([])
    ax.set_xticks(range(0, 801, 200))
    ax.set_xlabel("seconds per optimizer update (192 examples, 4×H100)", color=INK2, fontsize=10)
    ax.tick_params(axis="x", colors=INK2, labelsize=9.5, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_facecolor(SURFACE)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures/update_time_reduction")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 1.0})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=SURFACE)
    panel(axes[0], LMD, BLUE, "LMD  (distillation from the released model)")
    panel(axes[1], LSD, ORANGE, "LSD  (self-distillation)")
    fig.text(0.012, 0.015, "Each row adds one change to the row above. Hatched bars are estimated from "
             "measured components; all others are measured. Full-joint flow-map training on LIBERO, "
             "multipliers vs. the baseline row.", fontsize=9, color=INK2)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.83, bottom=0.14, wspace=0.08)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(str(out) + suffix, dpi=220, facecolor=SURFACE)
    print("wrote", out.with_suffix(".png"), "+ .pdf, .svg")


if __name__ == "__main__":
    main()
