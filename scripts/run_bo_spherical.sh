#!/bin/bash
# Wait for spherical-ot training to finish, then start BO with spherical flow model

LOG_FILE=$(ls -t /home/prusek/NLP/study/results/unet_spherical_ot_10k_2000ep_*.log 2>/dev/null | head -1)
echo "Monitoring: $LOG_FILE"

while true; do
    # Check if training process is still running
    if ! pgrep -f "study.flow_matching.train.*spherical" > /dev/null; then
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
echo "Spherical-OT Training complete. Final checkpoint:"
python -c "
import torch
ckpt = torch.load('study/checkpoints/unet-spherical-ot-10k-none/best.pt', map_location='cpu')
print(f\"  Epoch: {ckpt['epoch']}\")
print(f\"  Best loss: {ckpt['best_loss']:.6f}\")
"
echo "=========================================="
echo ""

# Start BO with spherical flow model
GPU=${GPU:-1}  # Default to GPU 1, configurable via environment
echo "Starting EcoFlow BO with Spherical-OT flow model + Riemannian guidance on GPU $GPU..."
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/prusek/NLP uv run python -m ecoflow.run_bo_full \
    --warm-start-k 10 \
    --gp-kernel arccosine \
    --llm-budget 50000 \
    --eval-size 1319 \
    --model qwen \
    --flow-checkpoint study/checkpoints/unet-spherical-ot-10k-none/best.pt \
    2>&1 | tee ecoflow/results/bo_spherical_ot_$(date +%Y%m%d_%H%M%S).log
