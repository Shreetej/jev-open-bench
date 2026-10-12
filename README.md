# Jev vs open models: benchmark artifacts (6 Oct 2026)

Backs `../../medium-post-jev.md`.

- Round 1 (400 items): `open_bench.py` (TypeSafe-API models: Kev, Von), `decider_run.py`, `llm_readout.py` (logit readout for plain/LoRA MLX models). Score with `analyze.py`.
  Items are rebuilt with github.com/manjunathshiva/jev-frontier-bench (`bench.py sample`, `ext_bench.py sample`);
  `jfb_bench_patched.py` is its bench.py with the ChaosNLI column fix (upstream renamed `label_counter` to `label_count`).
- Round 2 (4,000 paired items): items built with AppliedMachineLearning-Lab/jev-benchmarking's prompts; Jev per-item
  responses from Zenodo 10.5281/zenodo.23039006 (`responses.db`, research use only, not redistributed here), matched by
  request hash. Score with `analyze_full.py`.
- Fine-tuning: `prep_train.py` (train-split labels), `run_ft.sh` (mlx_lm LoRA), `run_sample.sh`.
- `results/`: raw per-item outputs (`open_calls.jsonl`, `full_calls*.jsonl`) and summaries. Dataset text and the
  Bonn-derived Jev answers are not included; rebuild them from the sources above.

- Added 7-12 Oct: `clef_run.py` (Cloudflare Clef-flash via the mlx-community 8-bit conversion and its bundled `clef_mlx.py`,
  run with mlx-vlm 0.7.x) and `nimble_mlx.py` (Bespoke-Nimble-9B v3/v1: peft merge into Qwen/Qwen3.5-9B, `mlx_lm convert`
  to 8-bit, scored with Nimble's own `extended_schema.py`). Both cap MLX's buffer cache; without that, the cache grew to
  25 GB and the 36 GB Mac swapped.
- `work/` holds the rebuilt environments, the six-task and full-set item files, Jev's per-item responses (research-use
  licence) and the converted Nimble models. It is several GB and can be deleted; everything in it is rebuildable.

- Added 12 Oct: Amazon Strands Decider 2B and E4B via `run_strands.sh` (`strands-decider serve` on MPS; E4B needs the GitHub build). `open_bench.py --null-desc-as-name` works around its server rejecting null option descriptions.

## Training data (LoRA fine-tune)

`training-data/train_items.json` is the labelled set behind the post's fine-tuning results: 5,994 items, 1,000 per task
from each dataset's train split (BoolQ has 994 after dropping overlaps with test items). Fields match the benchmark items
(`state`, `instructions`, `options`, `gold`); `source` gives the dataset, split and row index. The 100-label run uses the
first 100 items of each task.

| task | source | licence | text included |
|---|---|---|---|
| route | PolyAI BANKING77 `train.csv` | CC BY 4.0 | yes |
| yesno | google/boolq train | CC BY-SA 3.0 | yes |
| ambig | nyu-mll/multi_nli train | CC BY 3.0 / CC BY-SA 3.0 / MIT / OANC (varies by genre) | yes |
| agnews | fancyzhx/ag_news train | academic, non-commercial (no formal licence) | yes |
| emotion | dair-ai/emotion train | educational and research use only | yes |
| rate | Yelp/yelp_review_full train | Yelp Dataset Agreement (no redistribution) | **no**: row index only |

Rebuild the exact files used for training (checked against `training-data/SHA256SUMS`):

```sh
python restore_yelp.py      # fetches the 1,000 Yelp reviews by row index -> results/train_items.json
python build_lora_data.py   # -> data/lora100/, data/lora1000/ (mlx_lm completions format)
shasum -a 256 -c training-data/SHA256SUMS
```

`prep_train.py` regenerates `results/train_items.json` from scratch (seed 7); it needs the jev-frontier-bench item files to
skip test overlaps. Training itself is `run_ft.sh`: mlx_lm LoRA on Qwen3-4B-Instruct-2507 (4-bit), prompt-masked,
batch 4, lr 1e-4, about 1 epoch. The adapters aren't published.

The data is shared for research reference. Each source keeps its own licence and terms; cite the original datasets.
