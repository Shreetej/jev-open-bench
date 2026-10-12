set -x
.venv/bin/python llm_readout.py --arm qwen3.5-9b --model $HOME/.lmstudio/models/mlx-community/Qwen3.5-9B-OptiQ-4bit --items results/sample4k_items.json --log results/full_calls.jsonl
.venv/bin/python llm_readout.py --arm qwen3-4b+lora1000 --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --adapter adapters/q4-1000 --items results/sample4k_items.json --log results/full_calls.jsonl
.venv/bin/python llm_readout.py --arm qwen3-4b-instruct --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --items results/sample4k_items.json --log results/full_calls.jsonl
