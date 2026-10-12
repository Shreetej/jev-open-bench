"""Zero-shot (or LoRA-tuned) open LLM as a decision model: read option probabilities from logits.

Options are numbered 1..K. The model sees the same state, instructions and options Jev saw and
is asked for the option number. No text is generated: P(option) is read from the next-token
distribution over digits (two steps for K > 9: first digit, then second digit or end), so every
answer is a full distribution over the declared options, like Jev's.

  python llm_readout.py --arm qwen3-4b --model <mlx path> [--adapter <dir>]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from open_bench import LOG, OUT, load_items

END = "\n"


def render_state(state: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in state.items())


def option_lines(item: dict) -> list[str]:
    if item["type"] == "noul":
        return ["yes", "no"]
    return [f"{k}: {v}" if v and v != k else k for k, v in item["options"].items()]


def option_keys(item: dict) -> list[str]:
    return ["yes", "no"] if item["type"] == "noul" else list(item["options"])


def user_prompt(item: dict) -> str:
    opts = "\n".join(f"{i}. {line}" for i, line in enumerate(option_lines(item), 1))
    return (f"Input:\n{render_state(item['state'])}\n\nQuestion: {item['instructions']}\n\n"
            f"Options:\n{opts}\n\nReply with the number of the correct option only.")


def build_prompt(tok, item: dict) -> str:
    msgs = [{"role": "user", "content": user_prompt(item)}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return p + "Answer: "


def digit_ids(tok) -> list[int]:
    ids = []
    for d in "0123456789":
        t = tok.encode(d, add_special_tokens=False)
        assert len(t) == 1, (d, t)
        ids.append(t[0])
    return ids


def logprobs(model, tokens: list[int], cache) -> mx.array:
    out = model(mx.array(tokens)[None], cache=cache)
    return out[0, -1].astype(mx.float32) - mx.logsumexp(out[0, -1].astype(mx.float32))


def option_probs(model, tok, item: dict, digits: list[int], stop_ids: list[int]) -> list[float]:
    k = len(option_keys(item))
    ids = tok.encode(build_prompt(tok, item), add_special_tokens=False)
    cache = make_prompt_cache(model)
    lp1 = logprobs(model, ids, cache)
    mx.eval(lp1)
    if k <= 9:
        sc = [lp1[digits[i]].item() for i in range(1, k + 1)]
    else:
        sc = [-math.inf] * k
        for first in range(1, 10):
            if first > k and first * 10 > k:
                continue
            c2 = copy.deepcopy(cache)
            lp2 = logprobs(model, [digits[first]], c2)
            mx.eval(lp2)
            base = lp1[digits[first]].item()
            if first <= k:  # single-digit label: first digit then a non-digit
                p_end = 1.0 - sum(math.exp(lp2[d].item()) for d in digits)
                sc[first - 1] = base + math.log(max(p_end, 1e-12))
            for second in range(10):
                n = first * 10 + second
                if 10 <= n <= k:
                    sc[n - 1] = base + lp2[digits[second]].item()
    m = max(sc)
    ex = [math.exp(s - m) for s in sc]
    z = sum(ex)
    return [e / z for e in ex]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--items", help="JSON list of items (default: the jev-frontier-bench items)")
    ap.add_argument("--log", help="output JSONL (default: results/open_calls.jsonl)")
    args = ap.parse_args()
    log = Path(args.log) if args.log else LOG
    source = json.loads(Path(args.items).read_text()) if args.items else load_items()

    model, tok = load(args.model, adapter_path=args.adapter)
    digits = digit_ids(tok)
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
    print(f"{args.arm}: {len(items)} items", flush=True)
    with log.open("a") as f:
        for n, it in enumerate(items, 1):
            rec = {"item": it["id"], "task": it["task"], "arm": args.arm,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            t0 = time.perf_counter()
            p = option_probs(model, tok, it, digits, [])
            keys = option_keys(it)
            probs = dict(zip(keys, p))
            rec.update(probs=probs, wall_s=time.perf_counter() - t0,
                       answer=max(probs, key=probs.get))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"  {n}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
