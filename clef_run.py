"""Cloudflare Clef-flash (MLX conversion) on the same typed questions Jev received.

Uses the bundled clef_mlx.py loader (backbone + joint schema head) and its Jev-compatible
`systemone` call. Run with an interpreter that has mlx-vlm 0.7.x.

  python clef_run.py --arm clef-flash-8bit --repo mlx-community/clef-flash-8bit [--items F --log F]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from open_bench import LOG, load_items, question, to_probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--items")
    ap.add_argument("--log")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    log = Path(args.log) if args.log else LOG
    source = json.loads(Path(args.items).read_text()) if args.items else load_items()

    path = snapshot_download(args.repo)
    sys.path.insert(0, path)
    import clef_mlx  # noqa: E402
    import mlx.core as mx

    # MLX's buffer cache otherwise grows with every new input shape until a 36 GB Mac swaps.
    mx.set_cache_limit(2 * 1024**3)
    model = clef_mlx.load(path)
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] == args.arm and not r.get("error"):
                done.add(r["item"])
    items = [it for it in source if it["id"] not in done]
    if args.limit:
        items = items[: args.limit]
    print(f"{args.arm}: {len(items)} items", flush=True)
    with log.open("a") as f:
        for n, it in enumerate(items, 1):
            rec = {"item": it["id"], "task": it["task"], "arm": args.arm,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                t0 = time.perf_counter()
                resp = model.systemone({"model": args.arm, "state": it["state"],
                                        "questions": {"q": question(it)}})
                wall = time.perf_counter() - t0
                a = resp["answers"]["q"]
                probs = to_probs(it, a)
                rec.update(probs=probs, wall_s=wall, answer=max(probs, key=probs.get),
                           confidence=a.get("confidence"))
            except Exception as e:  # logged and retried on the next run
                rec["error"] = repr(e)[:500]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            mx.clear_cache()
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
