"""Fill the redacted Yelp reviews back into training-data/train_items.json.

The Yelp dataset licence doesn't allow redistribution, so the published file keeps only each review's
row index in Yelp/yelp_review_full's train split. This fetches those rows from the HF datasets-server
(about 30 pages of 100), checks each label against the stored gold rating, and writes
results/train_items.json, which build_lora_data.py turns into the exact LoRA files (see SHA256SUMS).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ROWS = "https://datasets-server.huggingface.co/rows"


def page(offset: int) -> list[dict]:
    q = urllib.parse.urlencode({"dataset": "Yelp/yelp_review_full", "config": "yelp_review_full",
                                "split": "train", "offset": offset, "length": 100})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(f"{ROWS}?{q}", timeout=120) as r:
                return [x["row"] for x in json.loads(r.read())["rows"]]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"datasets-server kept failing at offset {offset}")


def main() -> None:
    items = json.loads((ROOT / "training-data/train_items.json").read_text())
    todo = [it for it in items if it["source"]["dataset"] == "Yelp/yelp_review_full"]
    pages: dict[int, list[dict]] = {}
    for it in todo:
        row = it["source"]["row"]
        off = row - row % 100
        if off not in pages:
            pages[off] = page(off)
            time.sleep(1.5)
        r = pages[off][row - off]
        if str(int(r["label"]) + 1) != it["gold"]:
            raise ValueError(f"row {row}: label {r['label']} doesn't match gold {it['gold']}")
        it["state"] = {"review": r["text"]}
        it.pop("redacted", None)
    out = ROOT / "results/train_items.json"
    out.write_text(json.dumps(items))
    print(f"restored {len(todo)} Yelp reviews from {len(pages)} pages -> {out}")


if __name__ == "__main__":
    main()
