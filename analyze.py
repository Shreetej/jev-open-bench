"""Score every arm against Jev 1.13 on identical items.

Jev and Laya answers come from jev-frontier-bench's published logs (calls.jsonl,
ext_calls.jsonl); everything else from results/open_calls.jsonl. Confidence = top probability
(as in jev-frontier-bench). Paired accuracy differences vs Jev use a bootstrap over items.
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict

from open_bench import JFB, LOG, load_items

TASKS = ["route", "yesno", "rate", "ambig", "agnews", "emotion"]
NAMES = {"route": "Banking77 (78-way)", "yesno": "BoolQ", "rate": "Yelp 1-5★", "ambig": "ChaosNLI",
         "agnews": "AG News", "emotion": "Emotion"}


def read(path, arms=None) -> dict:
    out: dict = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r.get("error") or not r.get("probs"):
            continue
        if arms and r["arm"] not in arms:
            continue
        out[r["arm"]][r["item"]] = r
    return out


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    tot = len(pairs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(c, ok) for c, ok in pairs if (lo < c <= hi) or (b == 0 and c == 0)]
        if sel:
            e += len(sel) / tot * abs(sum(c for c, _ in sel) / len(sel) - sum(ok for _, ok in sel) / len(sel))
    return e


def main() -> None:
    items = {it["id"]: it for it in load_items()}
    arms = read(JFB / "calls.jsonl", {"jev", "laya", "laya-ml", "laya-td"})
    for k, v in read(JFB / "ext_calls.jsonl", {"jev", "laya", "laya-ml", "laya-td"}).items():
        arms[k].update(v)
    for k, v in read(LOG).items():
        arms[k] = v
    jev = arms["jev"]
    only = sys.argv[1:] or list(arms)

    def correct(r, iid):
        p = r["probs"]
        return max(p, key=p.get) == items[iid]["gold"]

    rows = []
    for arm in only:
        a = arms[arm]
        common = [i for i in items if i in a and i in jev]
        if not common:
            continue
        per = {}
        for t in TASKS:
            ids = [i for i in common if items[i]["task"] == t]
            per[t] = (sum(correct(a[i], i) for i in ids) / len(ids) * 100) if ids else None
        acc = sum(correct(a[i], i) for i in common) / len(common) * 100
        jacc = sum(correct(jev[i], i) for i in common) / len(common) * 100
        rng = random.Random(0)
        diffs = []
        for _ in range(2000):
            s = [rng.choice(common) for _ in common]
            diffs.append(sum(correct(a[i], i) - correct(jev[i], i) for i in s) / len(s) * 100)
        diffs.sort()
        cal = ece([(max(a[i]["probs"].values()), correct(a[i], i)) for i in common])
        lat = st.median(a[i]["wall_s"] for i in common if a[i].get("wall_s") is not None) \
            if any(a[i].get("wall_s") is not None for i in common) else None
        rows.append((arm, len(common), acc, acc - jacc, diffs[50], diffs[1949], cal, lat, per))

    rows.sort(key=lambda r: -r[2])
    hdr = f"{'arm':28} {'n':>4} {'acc':>6} {'vs Jev':>7} {'95% CI':>15} {'ECE':>6} {'p50 s':>6} " + \
          " ".join(f"{t:>7}" for t in TASKS)
    print(hdr)
    for arm, n, acc, d, lo, hi, cal, lat, per in rows:
        print(f"{arm:28} {n:4d} {acc:6.1f} {d:+7.1f} {f'[{lo:+.1f},{hi:+.1f}]':>15} {cal:6.3f} "
              f"{(f'{lat:.2f}' if lat is not None else '-'):>6} "
              + " ".join(f"{(f'{per[t]:.0f}' if per[t] is not None else '-'):>7}" for t in TASKS))
    json.dump([dict(arm=r[0], n=r[1], acc=r[2], vs_jev=r[3], ci=[r[4], r[5]], ece=r[6],
                    p50_s=r[7], per_task=r[8]) for r in rows],
              open(LOG.parent / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
