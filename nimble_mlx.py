"""Bespoke-Nimble-9B on Apple Silicon: the published adapter merged into Qwen3.5-9B, converted to MLX.

Nimble's own inference (CUDA-only) builds a prompt per question with its extended_schema module and
takes a softmax over the candidate-letter logits at the answer slot. This does the same in MLX, with
Nimble's prompt code (extended_schema.py, parallel_schema.py) and its typed-question -> schema
conversion copied unchanged from bespokelabs/bespoke-nimble-9b-v3-decision-index (code/nimble/evaluation).

  python nimble_mlx.py merge --adapter <dir> --out <merged dir>      # peft merge, CPU
  python -m mlx_lm convert --hf-path <merged dir> --mlx-path <mlx dir> -q --q-bits 8
  python nimble_mlx.py run --arm nimble-9b-v3 --mlx <mlx dir> --tok <merged dir> [--items F --log F]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "nimble"))


def native_question(question: dict) -> dict:
    if question["type"] != "noul":
        return question
    criteria = question.get("criteria") or {}
    if set(criteria) <= {"true", "false"}:
        return question
    if set(criteria) == {"yes", "no"}:
        return dict(question, criteria={"true": criteria["yes"], "false": criteria["no"]})
    raise ValueError(f"yes/no criteria {sorted(criteria)} have no true/false reading")


def native_schema(body: dict) -> tuple[str, dict]:
    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    schema = {}
    for name, q in body["questions"].items():
        field = {"description": text(q["instructions"])}
        criteria = q.get("criteria") or {}
        if q["type"] == "noul":
            field.update(type="boolean", choices=[False, True],
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        elif q["type"] == "choice":
            field.update(type="enum", choices=list(criteria),
                         choice_descriptions={k: text(v) for k, v in criteria.items() if v is not None})
        else:  # score
            field.update(type="enum", choices=[str(i) for i in range(len(criteria))],
                         choice_descriptions={str(i): text(v) for i, v in enumerate(criteria)})
        schema[name] = field
    return text(body["state"]), schema


def merge(adapter: str, out: str) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    base = Qwen3_5ForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-9B", dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.save_pretrained(out, max_shard_size="5GB")
    AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B").save_pretrained(out)  # text-only: no image processor needed


def run(args) -> None:
    import mlx.core as mx
    from mlx_lm import load
    from transformers import AutoTokenizer

    import extended_schema
    from open_bench import LOG, load_items, question

    tok = AutoTokenizer.from_pretrained(args.tok)
    mx.set_cache_limit(2 * 1024**3)
    model, _ = load(args.mlx)
    log = Path(args.log) if args.log else LOG
    source = json.loads(Path(args.items).read_text()) if args.items else load_items()
    done = set()
    if log.exists():
        for line in log.read_text().splitlines():
            r = json.loads(line)
            if r["arm"] == args.arm and not r.get("error"):
                done.add(r["item"])
    items = [it for it in source if it["id"] not in done][: args.limit or None]
    print(f"{args.arm}: {len(items)} items", flush=True)
    with log.open("a") as f:
        for n, it in enumerate(items, 1):
            rec = {"item": it["id"], "task": it["task"], "arm": args.arm,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                t0 = time.perf_counter()
                state = it["state"]
                if isinstance(state, str) and not state.strip():
                    state = json.dumps(state)
                body = {"state": state, "questions": {"q": native_question(question(it))}}
                context, schema = native_schema(body)
                p = extended_schema.prepare_prompts(tok, context, schema, 32768)
                logits = model(mx.array(p.full_ids[0])[None])[0, -1].astype(mx.float32)
                sel = logits[mx.array(p.candidate_ids[0])]
                probs_l = mx.softmax(sel).tolist()
                choices = [extended_schema.choice_key(c) for c in p.choices[0]]
                pr = dict(zip(choices, probs_l))
                if it["type"] == "noul":
                    probs = {"yes": pr["true"], "no": pr["false"]}
                elif it["type"] == "choice":
                    probs = pr
                else:
                    keys = list(it["options"])
                    probs = {keys[int(k)]: v for k, v in pr.items()}
                rec.update(probs=probs, wall_s=time.perf_counter() - t0,
                           answer=max(probs, key=probs.get))
            except Exception as e:
                rec["error"] = repr(e)[:500]
            f.write(json.dumps(rec) + "\n")
            f.flush()
            mx.clear_cache()
            if n % 100 == 0:
                print(f"  {n}/{len(items)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["merge", "run"])
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--arm")
    ap.add_argument("--mlx")
    ap.add_argument("--tok")
    ap.add_argument("--items")
    ap.add_argument("--log")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    merge(a.adapter, a.out) if a.cmd == "merge" else run(a)


if __name__ == "__main__":
    main()
