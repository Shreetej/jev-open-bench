"""Medium figures for the Jev post, computed straight from the per-item result logs.

All numbers are the 4,000-item paired sample (seed 20261006) vs Jev 1.13.0's recorded answers.
Writes PNGs to ~/projects/pers/Medium/medium-jev-assets/.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analyze import ece
from analyze_full import jev_probs

R = Path(__file__).resolve().parent / "results"
OUT = Path.home() / "projects/pers/Medium/medium-jev-assets"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill), light mode; validated: blue/orange pass all six checks.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, NEUTRAL, RED = "#2a78d6", "#eb6834", "#f0efec", "#e34948"
plt.rcParams.update({"font.family": "Helvetica Neue", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

TASKS = ["banking77", "boolq", "ag_news", "emotion"]
TASK_LABEL = {"banking77": "Banking77\n(77 intents)", "boolq": "BoolQ\n(yes/no)", "ag_news": "AG News\n(topic)",
              "emotion": "Emotion\n(6-way)"}
ARMS = [  # arm id, display name, group
    ("clef-flash-8bit", "Clef-flash 9B (Cloudflare)", "decision"),
    ("decider-0.8b", "decider-0.8b", "decision"),
    ("von-1.3", "Von 1.3 (CPU)", "decision"),
    ("nimble-9b-v3", "Nimble-9B v3 (Bespoke Labs)", "decision"),
    ("strands-E4B-gemma4", "Strands Decider E4B (Amazon)", "decision"),
    ("nimble-9b-v1", "Nimble-9B v1 (Bespoke Labs)", "decision"),
    ("kev-4b", "Kev-4B", "decision"),
    ("strands-2B-qwen3.5", "Strands Decider 2B (Amazon)", "decision"),
    ("qwen3-4b+lora1000", "Qwen3-4B + my LoRA (1,000 labels/task)", "plain"),
    ("qwen3.5-9b", "Qwen3.5-9B, no decision training", "plain"),
    ("qwen3-4b-instruct", "Qwen3-4B, no decision training", "plain"),
]


def load():
    items = {it["id"]: it for it in json.loads((R / "full_items.json").read_text())}
    sample = [it["id"] for it in json.loads((R / "sample4k_items.json").read_text())]
    jev = {k: jev_probs(v) for k, v in json.loads((R / "bonn_jev_by_item.json").read_text()).items()}
    arms: dict = defaultdict(dict)
    for f in sorted(R.glob("full_calls*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if not r.get("error") and r.get("probs"):
                arms[r["arm"]][r["item"]] = r["probs"]
    return items, sample, jev, arms


def stats(items, sample, jev, res):
    ok = lambda p, i: max(p, key=p.get) == items[i]["gold"]  # noqa: E731
    ids = [i for i in sample if i in res]
    a = [ok(res[i], i) for i in ids]
    j = [ok(jev[i], i) for i in ids]
    rng = random.Random(0)
    d = sorted(sum(a[k] - j[k] for k in s) / len(s) * 100
               for s in ([rng.randrange(len(ids)) for _ in ids] for _ in range(1000)))
    per = {}
    for t in TASKS:
        tid = [k for k, i in enumerate(ids) if items[i]["task"] == t]
        per[t] = (sum(a[k] for k in tid) - sum(j[k] for k in tid)) / len(tid) * 100
    return dict(n=len(ids), diff=(sum(a) - sum(j)) / len(ids) * 100, lo=d[25], hi=d[974],
                ece=ece([(max(res[i].values()), x) for i, x in zip(ids, a)]), per=per)


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def forest(S):
    rows = [(name, grp, S[a]) for a, name, grp in ARMS]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    for y, (name, grp, s) in enumerate(reversed(rows)):
        c = BLUE if grp == "decision" else ORANGE
        ax.plot([s["lo"], s["hi"]], [y, y], color=c, lw=2, solid_capstyle="round")
        ax.scatter([s["diff"]], [y], s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(s["hi"] + 0.3, y, f"{s['diff']:+.1f}", va="center", fontsize=10, color=INK2)
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.text(0.1, -0.85, "Jev 1.13 = 0", color=INK2, fontsize=9.5, va="center")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    ax.set_xlabel("Accuracy difference vs Jev, percentage points (dot = estimate, line = 95% CI)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(-9, 6.2)
    style(ax)
    ax.scatter([], [], color=BLUE, s=50, label="Open decision models")
    ax.scatter([], [], color=ORANGE, s=50, label="Plain open models (± my fine-tune)")
    ax.set_ylim(-1.3, len(rows) - 0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    fig.suptitle("Open decision models match or beat Jev; plain open models don't",
                 x=0.02, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.02, 0.905, "4,000 items, each paired with Jev's recorded answer (Banking77, BoolQ, AG News, Emotion)",
             fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig1-accuracy-vs-jev.png", facecolor=SURFACE)


def heatmap(S):
    rows = [(name, S[a]) for a, name, grp in ARMS if grp == "decision"]
    cmap = LinearSegmentedColormap.from_list("div", [RED, NEUTRAL, BLUE])
    data = [[s["per"][t] for t in TASKS] for _, s in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=200)
    ax.imshow(data, cmap=cmap, vmin=-15, vmax=15, aspect="auto")
    for y, row in enumerate(data):
        for x, v in enumerate(row):
            ax.text(x, y, f"{v:+.1f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(v) >= 9 else INK, fontweight="bold" if abs(v) >= 5 else "normal")
    ax.set_xticks(range(len(TASKS)), [TASK_LABEL[t] for t in TASKS])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.xaxis.tick_top()
    ax.set_xticks([x - 0.5 for x in range(1, len(TASKS))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("The wins are task-specific, and they track training data",
                 x=0.02, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.02, 0.905, "Accuracy difference vs Jev by task, percentage points. Blue = beats Jev, red = trails; "
             "colour capped at ±15.", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "fig2-per-task.png", facecolor=SURFACE)


def calibration(S, jev_ece):
    rows = [(name, S[a]["ece"], False) for a, name, _ in ARMS] + [("Jev 1.13", jev_ece, True)]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    for y, (name, v, is_jev) in enumerate(reversed(rows)):
        ax.barh(y, v, height=0.62, color=ORANGE if is_jev else BLUE)
        ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "Jev 1.13":
            lab.set_fontweight("bold")
    ax.set_xlabel("Expected calibration error (lower = confidence matches accuracy better)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    style(ax)
    fig.suptitle("Most open decision models state their confidence more honestly than Jev",
                 x=0.02, ha="left", fontsize=14, color=INK, fontweight="bold")
    fig.text(0.02, 0.905, "Same 4,000 items; Jev in orange", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "fig3-calibration.png", facecolor=SURFACE)


def main() -> None:
    items, sample, jev, arms = load()
    S = {a: stats(items, sample, jev, arms[a]) for a, _, _ in ARMS}
    ok = lambda p, i: max(p, key=p.get) == items[i]["gold"]  # noqa: E731
    jev_ece = ece([(max(jev[i].values()), ok(jev[i], i)) for i in sample])
    for a, name, _ in ARMS:
        s = S[a]
        print(f"{name:42} n={s['n']} diff={s['diff']:+.1f} [{s['lo']:+.1f},{s['hi']:+.1f}] ece={s['ece']:.3f} "
              + " ".join(f"{t}={s['per'][t]:+.1f}" for t in TASKS))
    print("jev ece", round(jev_ece, 3))
    forest(S)
    heatmap(S)
    calibration(S, jev_ece)
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
