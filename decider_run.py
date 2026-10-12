"""decider-0.8b in-process on MPS, same typed questions as Jev; logs to results/open_calls.jsonl."""
import glob, json, sys, time
D = glob.glob(f"{__import__('os').path.expanduser('~')}/.cache/huggingface/hub/models--Mapika--decider-0.8b/snapshots/*/")[0]
sys.path.insert(0, D)
from decider.infer import Decider
from pathlib import Path
from open_bench import LOG, load_items, question, to_probs
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--items"); ap.add_argument("--log"); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
LOG = Path(args.log) if args.log else LOG
SOURCE = json.loads(Path(args.items).read_text()) if args.items else None

arm = "decider-0.8b"
d = Decider(D, device="mps", use_graphs=False)
done = set() if not LOG.exists() else {json.loads(l)["item"] for l in open(LOG) if json.loads(l)["arm"] == arm and not json.loads(l).get("error")}
items = [it for it in (SOURCE or load_items()) if it["id"] not in done]
limit = args.limit
items = items[:limit] if limit else items
print(f"{arm}: {len(items)} items", flush=True)
with open(LOG, "a") as f:
    for n, it in enumerate(items, 1):
        rec = {"item": it["id"], "task": it["task"], "arm": arm, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            t0 = time.perf_counter()
            resp = d.system_one(it["state"], {"q": question(it)})
            wall = time.perf_counter() - t0
            a = resp["answers"]["q"]
            probs = to_probs(it, a)
            rec.update(probs=probs, wall_s=wall, answer=max(probs, key=probs.get), confidence=a.get("confidence"))
        except Exception as e:
            rec["error"] = repr(e)[:500]
        f.write(json.dumps(rec) + "\n"); f.flush()
        if n % 50 == 0: print(f"  {n}/{len(items)}", flush=True)
