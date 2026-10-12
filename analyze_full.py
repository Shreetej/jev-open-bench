"""Large-sample check: open models vs Jev 1.13.0, paired per item.

Jev's per-item responses come from the Bonn/Fraunhofer release (arXiv 2609.37647, Zenodo
10.5281/zenodo.23039006, responses.db), matched to results/full_items.json by the SHA-256 request
key their harness uses; all 15,946 items matched, so the prompts are byte-identical.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from analyze import ece

R = Path(__file__).resolve().parent / "results"
TASKS = ["banking77", "boolq", "ag_news", "emotion"]


def jev_probs(resp: dict) -> dict:
    a = resp["answers"]["answer"]
    if a["type"] == "noul":
        return {"yes": a["noul"], "no": 1 - a["noul"]}
    return a["probabilities"]


def main() -> None:
    items = {it["id"]: it for it in json.loads((R / "full_items.json").read_text())}
    jev = {k: jev_probs(v) for k, v in json.loads((R / "bonn_jev_by_item.json").read_text()).items()}
    arms: dict = defaultdict(dict)
    for f in sorted(R.glob("full_calls*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            if not r.get("error") and r.get("probs"):
                arms[r["arm"]][r["item"]] = r["probs"]

    def ok(p, iid):
        return max(p, key=p.get) == items[iid]["gold"]

    print(f"{'system':22} {'task':10} {'n':>5} {'open':>6} {'jev':>6} {'diff':>6} {'95% CI':>14} "
          f"{'ECE open':>8} {'ECE jev':>8}")
    out = []
    for arm, res in sorted(arms.items()):
        for t in TASKS + ["ALL"]:
            ids = [i for i in res if i in jev and (t == "ALL" or items[i]["task"] == t)]
            if not ids:
                continue
            a = [ok(res[i], i) for i in ids]
            j = [ok(jev[i], i) for i in ids]
            rng = random.Random(0)
            d = sorted(sum(a[k] - j[k] for k in s) / len(s) * 100
                       for s in ([rng.randrange(len(ids)) for _ in ids] for _ in range(1000)))
            ea = ece([(max(res[i].values()), x) for i, x in zip(ids, a)])
            ej = ece([(max(jev[i].values()), x) for i, x in zip(ids, j)])
            acc, jacc = sum(a) / len(a) * 100, sum(j) / len(j) * 100
            print(f"{arm:22} {t:10} {len(ids):5d} {acc:6.1f} {jacc:6.1f} {acc - jacc:+6.1f} "
                  f"{f'[{d[25]:+.1f},{d[974]:+.1f}]':>14} {ea:8.3f} {ej:8.3f}")
            out.append(dict(arm=arm, task=t, n=len(ids), acc=acc, jev=jacc, ci=[d[25], d[974]],
                            ece=ea, ece_jev=ej))
    (R / "summary_full.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
