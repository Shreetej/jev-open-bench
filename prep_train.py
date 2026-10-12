"""Labelled training items for the six tasks, from each dataset's TRAIN split (test items come
from test/validation splits, so the two never overlap). Same state/instructions/options as the
benchmark items. Writes results/train_items.json with 1,000 items per task (seed 7); the
100-label runs use the first 100 of each task. Each item records its source dataset, split and
row index (Banking77: row in the PolyAI train.csv).
"""
from __future__ import annotations

import csv
import io
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request

from open_bench import OUT, load_items

N = 1000
ROWS = "https://datasets-server.huggingface.co/rows"


def hf_rows(dataset: str, config: str, split: str, n: int, rng: random.Random,
            total: int, keep=lambda r: True) -> list[dict]:
    """Random pages of 100 rows from the HF datasets-server (avoids downloading whole splits)."""
    out: list[dict] = []
    offsets = rng.sample(range(0, total - 100, 100), k=min(30, total // 100 - 1))
    for off in offsets:
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": off, "length": 100})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(f"{ROWS}?{q}", timeout=120) as r:
                    rows = [x["row"] for x in json.loads(r.read())["rows"]]
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504):
                    raise
                time.sleep(5 * 2 ** attempt)
        time.sleep(1.5)
        out += [dict(r, _row=off + k) for k, r in enumerate(rows) if keep(r)]
        if len(out) >= n * 2:
            break
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    rng = random.Random(7)
    tmpl = {}
    for it in load_items():
        tmpl.setdefault(it["task"], it)
    test_text = {json.dumps(it["state"], sort_keys=True) for it in load_items()}
    items: list[dict] = []

    def add(task: str, state: dict, gold: str, src: tuple[str, str, int],
            instructions: str | None = None) -> None:
        if json.dumps(state, sort_keys=True) in test_text:
            return
        t = tmpl[task]
        items.append({"task": task, "type": t["type"], "state": state,
                      "instructions": instructions or t["instructions"],
                      "options": t["options"], "gold": gold,
                      "source": {"dataset": src[0], "split": src[1], "row": src[2]}})

    # route: PolyAI BANKING77 train CSV
    url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    with urllib.request.urlopen(url, timeout=60) as r:
        rows = [dict(x, _row=i) for i, x in enumerate(csv.DictReader(io.StringIO(r.read().decode())))]
    for r in rng.sample(rows, N):
        add("route", {"message": r["text"]}, r["category"].rstrip("?"), ("PolyAI/banking77", "train", r["_row"]))

    # yesno: BoolQ train
    for r in hf_rows("google/boolq", "default", "train", N, rng, 9427):
        q = r["question"].strip().rstrip("?") + "?"
        add("yesno", {"passage": r["passage"]}, "yes" if r["answer"] else "no", ("google/boolq", "train", r["_row"]),
            f"According to `passage`, is the answer to this question yes? {q}")

    # rate: Yelp full train, reviews <= 1500 chars (same filter as the test items)
    for r in hf_rows("Yelp/yelp_review_full", "yelp_review_full", "train", N, rng, 650000,
                     keep=lambda r: len(r["text"]) <= 1500):
        add("rate", {"review": r["text"]}, str(int(r["label"]) + 1), ("Yelp/yelp_review_full", "train", r["_row"]))

    # ambig: MNLI train (ChaosNLI items are MNLI dev-matched)
    names = ["entailment", "neutral", "contradiction"]
    for r in hf_rows("nyu-mll/multi_nli", "default", "train", N, rng, 392702,
                     keep=lambda r: r["label"] in (0, 1, 2)):
        add("ambig", {"premise": r["premise"], "hypothesis": r["hypothesis"]}, names[r["label"]],
            ("nyu-mll/multi_nli", "train", r["_row"]))

    # agnews / emotion train
    ag = ["World", "Sports", "Business", "Sci/Tech"]
    for r in hf_rows("fancyzhx/ag_news", "default", "train", N, rng, 120000):
        add("agnews", {"article": r["text"]}, ag[int(r["label"])], ("fancyzhx/ag_news", "train", r["_row"]))
    emo = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    for r in hf_rows("dair-ai/emotion", "split", "train", N, rng, 16000):
        add("emotion", {"text": r["text"]}, emo[int(r["label"])], ("dair-ai/emotion", "train", r["_row"]))

    from collections import Counter
    print(Counter(i["task"] for i in items))
    (OUT / "train_items.json").write_text(json.dumps(items))


if __name__ == "__main__":
    main()
