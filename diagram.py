"""Architecture figure for the Medium post: a decision model in front of the LLM orchestrator."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path.home() / "projects/pers/Medium/medium-jev-assets"
SURFACE, INK, INK2, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834"
BLUE_BG, ORANGE_BG, GRAY_BG = "#e8f1fb", "#fdeee7", "#f0efec"
plt.rcParams.update({"font.family": "Helvetica Neue", "figure.facecolor": SURFACE})


def box(ax, x, y, w, h, edge, fill, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                lw=1.6, ec=edge, fc=fill))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=13, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=11, color=INK2, linespacing=1.55)


def arrow(ax, a, b, label=None, lx=0.0, ly=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK2,
                                 connectionstyle="arc3,rad=0"))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, ha=ha, va="center",
                fontsize=11, color=INK, fontstyle="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    ax.text(0.2, 3.75, "request", ha="left", va="center", fontsize=12, color=INK2)
    arrow(ax, (1.15, 3.75), (2.4, 3.75))
    box(ax, 2.4, 2.75, 5.2, 1.95, BLUE, BLUE_BG, "Decision model (one pass, typed questions)",
        "Which queue, agent or tool?\nPolicy violation? Prompt injection?\nUrgent? Done? Needs a human?")
    arrow(ax, (4.0, 2.75), (2.3, 1.4), "confident", lx=-0.3, ha="right")
    arrow(ax, (6.0, 2.75), (7.7, 1.4), "unsure (p < threshold)", lx=0.3, ha="left")
    box(ax, 0.3, 0.15, 4.0, 1.25, INK2, GRAY_BG, "Act directly", "route, block, tag, score")
    box(ax, 5.7, 0.15, 4.0, 1.25, ORANGE, ORANGE_BG, "LLM orchestrator or human", "reason, plan, explain, write")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig0-where-jev-fits.png", facecolor=SURFACE)


if __name__ == "__main__":
    main()
