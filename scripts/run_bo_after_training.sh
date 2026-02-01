#!/bin/bash
# Wait for U-Net training to finish, then start BO

LOG_FILE=$(ls -t /home/prusek/NLP/study/results/unet_otcfm_10k_1000ep_*.log 2>/dev/null | head -1)
echo "Monitoring: $LOG_FILE"

while true; do
    # Check if training process is still running
    if ! pgrep -f "study.flow_matching.train.*unet" > /dev/null; then
        echo "Training finished or stopped!"
        break
    fi
    
    # Show latest progress
    tail -1 "$LOG_FILE" 2>/dev/null
    sleep 30
done

# Get final stats
echo ""
echo "=========================================="
echo "Training complete. Final checkpoint:"
python -c "
import torch
ckpt = torch.load('study/checkpoints/unet-otcfm-10k-none/best.pt', map_location='cpu')
print(f\"  Epoch: {ckpt['epoch']}\")
print(f\"  Best loss: {ckpt['best_loss']:.6f}\")
"
echo "=========================================="
echo ""

# Start BO
echo "Starting EcoFlow BO with Riemannian guidance..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/prusek/NLP uv run python -m ecoflow.run_bo_full \
    --warm-start-k 10 \
    --gp-kernel arccosine \
    --llm-budget 50000 \
    --eval-size 1319 \
    --model qwen \
    2>&1 | tee ecoflow/results/bo_unet_riemannian_$(date +%Y%m%d_%H%M%S).log
