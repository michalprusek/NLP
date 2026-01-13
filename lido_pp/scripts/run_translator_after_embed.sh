#!/bin/bash
# Wait for embed precompute to finish, then start translator training

EMBED_PID=3817081
EMBED_FILE="lido_pp/data/combined_embeddings.pt"

echo "Waiting for embed precompute (PID $EMBED_PID) to finish..."

# Wait for process to finish
while kill -0 $EMBED_PID 2>/dev/null; do
    sleep 60
    echo "$(date): Still waiting... (embed precompute running)"
done

echo "$(date): Embed precompute finished!"

# Verify file exists
if [ -f "$EMBED_FILE" ]; then
    echo "Embeddings file found: $EMBED_FILE"
    ls -lh "$EMBED_FILE"
    
    echo ""
    echo "Starting translator training with 8 prefix tokens..."
    
    CUDA_VISIBLE_DEVICES=1 uv run python -m lido_pp.training.train_translator \
        --epochs 50 \
        --batch-size 32 \
        --gradient-accumulation 2 \
        --vae-lr 1e-5 \
        --proj-lr 1e-4 \
        --vae-beta 0.001 \
        --num-prefix-tokens 8 \
        --resume-vae lido_pp/checkpoints/vae_alpaca_final.pt \
        --precomputed-embeddings "$EMBED_FILE" \
        --checkpoint-interval 5 \
        --log-interval 1 \
        2>&1 | tee "lido_pp/results/translator_8tok_260k_$(date +%Y%m%d_%H%M%S).log"
else
    echo "ERROR: Embeddings file not found!"
    exit 1
fi
