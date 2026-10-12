"""mlx_lm completions files from results/train_items.json: data/lora{100,1000}/{train,valid}.jsonl.

Same prompt as the benchmark readout (llm_readout.user_prompt); the completion is the 1-based option
number. 5% of each set (at least 20 rows) is held out for validation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from llm_readout import option_keys, user_prompt

ROOT = Path(__file__).resolve().parent


def main() -> None:
    items = json.loads((ROOT / "results/train_items.json").read_text())
    by: dict[str, list[dict]] = {}
    for it in items:
        by.setdefault(it["task"], []).append(it)
    for n in (100, 1000):
        d = ROOT / f"data/lora{n}"
        d.mkdir(parents=True, exist_ok=True)
        rows = []
        for its in by.values():
            for it in its[:n]:
                num = option_keys(it).index(it["gold"]) + 1
                rows.append({"prompt": user_prompt(it), "completion": f"Answer: {num}"})
        random.Random(1).shuffle(rows)
        nv = max(20, len(rows) // 20)
        (d / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:nv]))
        (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[nv:]))
        print(n, len(rows) - nv, nv)


if __name__ == "__main__":
    main()
