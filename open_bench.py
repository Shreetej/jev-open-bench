"""Run open decision models on the exact items Jev 1.13 was scored on (jev-frontier-bench).

Every arm receives the same typed question Jev received (bench.run_jev's request shape), via a
local TypeSafe-compatible /v1/systemone endpoint or an in-process predictor. One JSON line per
(item, arm) in results/open_calls.jsonl; runs resume.

  python open_bench.py run --arm kev4b --url http://127.0.0.1:8009
  python open_bench.py run --arm decider4b --inproc decider
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

JFB = Path(__file__).resolve().parent.parent / "jev-frontier-bench" / "results"
OUT = Path(__file__).resolve().parent / "results"
LOG = OUT / "open_calls.jsonl"


def load_items() -> list[dict]:
    items = json.loads((JFB / "items.json").read_text())["items"]
    ext = json.loads((JFB / "ext_items.json").read_text())["items"]
    # ext_bench sends empty descriptions as the option name; mirror that exactly.
    ext = [dict(it, options={k: (v or k) for k, v in it["options"].items()}) for it in ext]
    return items + ext


def question(item: dict) -> dict:
    """Identical to bench.run_jev's question construction."""
    if item["type"] == "noul":
        return {"type": "noul", "instructions": item["instructions"]}
    if item["type"] == "choice":
        return {"type": "choice", "instructions": item["instructions"], "criteria": item["options"]}
    return {"type": "score", "instructions": item["instructions"],
            "criteria": list(item["options"].values())}


def to_probs(item: dict, a: dict) -> dict:
    if a["type"] == "noul":
        return {"yes": a["noul"], "no": 1 - a["noul"]}
    if a["type"] == "choice":
        return a["probabilities"]
    keys = list(item["options"])
    return {keys[int(k)]: v for k, v in a["probabilities"].items()}


def call_http(url: str, item: dict, model: str, null_desc_as_name: bool = False) -> tuple[dict, float]:
    q = question(item)
    if null_desc_as_name and isinstance(q.get("criteria"), dict):
        # Some servers (Strands Decider 0.1.0) reject the spec's null option descriptions.
        q = dict(q, criteria={k: (v if v is not None else k) for k, v in q["criteria"].items()})
    body = json.dumps({"state": item["state"], "model": model, "questions": {"q": q}}).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", data=body, method="POST",
                                 headers={"content-type": "application/json",
                                          "authorization": "Bearer local"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    return resp, time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--url")
    ap.add_argument("--model", default="kev-latest")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--items", help="JSON list of items (default: the jev-frontier-bench items)")
    ap.add_argument("--log", help="output JSONL (default: results/open_calls.jsonl)")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--null-desc-as-name", action="store_true")
    args = ap.parse_args()
    log = Path(args.log) if args.log else LOG
    source = json.loads(Path(args.items).read_text()) if args.items else load_items()

    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] == args.arm and not r.get("error"):
                done.add(r["item"])
    items = [it for it in source if it["id"] not in done
             and (not args.tasks or it["task"] in args.tasks.split(","))]
    if args.limit:
        items = items[: args.limit]
    print(f"{args.arm}: {len(items)} items to run", flush=True)
    with log.open("a") as f:
        for n, it in enumerate(items, 1):
            rec = {"item": it["id"], "task": it["task"], "arm": args.arm,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                resp, wall = call_http(args.url, it, args.model, args.null_desc_as_name)
                a = resp["answers"]["q"]
                probs = to_probs(it, a)
                rec.update(probs=probs, wall_s=wall, answer=max(probs, key=probs.get),
                           confidence=a.get("confidence"))
            except Exception as e:  # logged, retried on the next run
                rec["error"] = repr(e)[:500]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
