set -x
while pgrep -f run_zeroshot.sh >/dev/null; do sleep 20; done
T0=$(date +%s)
.venv/bin/python -m mlx_lm lora --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --train --data data/lora100 --mask-prompt --batch-size 4 --iters 285 --learning-rate 1e-4 --max-seq-length 2048 --grad-checkpoint --adapter-path adapters/q4-100 --steps-per-eval 100 --val-batches 8 --seed 0
echo "TRAIN q4-100 seconds $(($(date +%s)-T0))"
.venv/bin/python llm_readout.py --arm qwen3-4b+lora100 --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --adapter adapters/q4-100
T0=$(date +%s)
.venv/bin/python -m mlx_lm lora --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --train --data data/lora1000 --mask-prompt --batch-size 4 --iters 1424 --learning-rate 1e-4 --max-seq-length 2048 --grad-checkpoint --adapter-path adapters/q4-1000 --steps-per-eval 400 --val-batches 8 --seed 0
echo "TRAIN q4-1000 seconds $(($(date +%s)-T0))"
.venv/bin/python llm_readout.py --arm qwen3-4b+lora1000 --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --adapter adapters/q4-1000
.venv/bin/python llm_readout.py --arm qwen3.5-9b --model $HOME/.lmstudio/models/mlx-community/Qwen3.5-9B-OptiQ-4bit --items results/full_items.json --log results/full_calls.jsonl
.venv/bin/python llm_readout.py --arm qwen3-4b+lora1000 --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510 --adapter adapters/q4-1000 --items results/full_items.json --log results/full_calls.jsonl
