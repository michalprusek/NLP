#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd /home/prusek/NLP
exec uv run python run_opro.py --model qwen --budget 200000 --minibatch-size 261 --num-candidates 8
