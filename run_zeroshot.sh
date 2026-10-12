set -x
.venv/bin/python llm_readout.py --arm qwen3-4b-instruct --model $HOME/.lmstudio/models/mlx-community/Qwen3-4B-Instruct-2507-4bit-DWQ-2510
.venv/bin/python llm_readout.py --arm qwen3.5-9b --model $HOME/.lmstudio/models/mlx-community/Qwen3.5-9B-OptiQ-4bit
.venv/bin/python llm_readout.py --arm gemma4-12b --model $HOME/.lmstudio/models/mlx-community/gemma-4-12B-it-OptiQ-4bit
.venv/bin/python llm_readout.py --arm qwen3.6-35b-a3b --model $HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/38740b847e4cb78f352aba30aa41c76e08e6eb46
.venv/bin/python llm_readout.py --arm qwen3.8-27b --model $HOME/.lmstudio/models/mlx-community/Qwen3.8-27B-4bit
