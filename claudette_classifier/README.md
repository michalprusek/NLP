# Claudette Binary Classifier

Binary classification of Terms of Service (ToS) clauses as fair/unfair using Legal-BERT embeddings and deep residual MLP.

## Overview

This classifier addresses the highly imbalanced Claudette dataset (~11% unfair, ~89% fair) using:

- **Encoder**: `nlpaueb/legal-bert-base-uncased` - Pre-trained BERT model on legal documents
- **Classifier**: Deep residual MLP with skip connections
- **Class Imbalance Techniques**:
  - Focal Loss (down-weights easy examples, focuses on hard ones)
  - Weighted Random Sampling (oversamples minority class)
  - Class-weighted loss functions

## Dataset

- **Source**: `datasets/tos_converted/tos_dataset.json`
- **Total samples**: 9,414 clauses from 50 companies
- **Classes**:
  - Fair: 8,382 samples (89%)
  - Unfair: 1,032 samples (11%)
- **Split**: 80% train / 10% val / 10% test (stratified)

## Architecture

### 1. Legal-BERT Encoder
- Pre-trained BERT model specialized for legal text
- Extracts 768-dim embeddings from [CLS] token
- Fine-tuned during training with lower learning rate

### 2. Deep Residual MLP
```
Input (768) → Linear → BN → ReLU → Dropout
  ↓
[Residual Block × 3] at 512 dims
  ↓
Linear(512→256) → BN → ReLU → Dropout
  ↓
[Residual Block × 3] at 256 dims
  ↓
Linear(256→128) → BN → ReLU → Dropout
  ↓
[Residual Block × 3] at 128 dims
  ↓
Output Linear(128→1) → Sigmoid
```

**Residual Block**:
```
x → Linear → BN → ReLU → Dropout → Linear → BN → (+) → ReLU
↓_____________________________________________↑
```

### 3. Loss Functions

**Focal Loss** (default):
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```
- α = 0.25 (weight for positive class)
- γ = 2.0 (focusing parameter)
- Down-weights easy examples, focuses on hard misclassifications

**Weighted BCE Loss** (alternative):
- Uses class weights: `pos_weight = n_negative / n_positive ≈ 8.1`

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Usage

### Quick Start Scripts

```bash
# Single GPU training (default settings)
./run_claudette_classifier.sh

# Dual GPU training (L40S 48GB, DDP)
cd claudette_classifier
./run_dual_gpu.sh

# Hyperparameter tuning (Optuna)
cd claudette_classifier
./run_hyperparameter_tuning.sh
```

### Basic Training

```bash
# Train with default settings (Focal Loss + oversampling)
uv run python -m claudette_classifier.main

# Train for 30 epochs with batch size 64
uv run python -m claudette_classifier.main --epochs 30 --batch-size 64

# Fine-tune BERT encoder (unfrozen)
uv run python -m claudette_classifier.main --train-encoder --epochs 100
```

### Dual GPU Training (Distributed Data Parallel)

For training on 2× L40S GPUs (48GB VRAM each):

```bash
cd claudette_classifier

# Default configuration (batch size 256 per GPU, fine-tune BERT)
./run_dual_gpu.sh

# Custom configuration via environment variables
EPOCHS=50 BATCH_SIZE=128 ./run_dual_gpu.sh

# Disable BERT fine-tuning (faster, less memory)
TRAIN_ENCODER=false ./run_dual_gpu.sh

# Adjust model capacity
HIDDEN_DIMS="2048 1024 512" RESIDUAL_BLOCKS=5 DROPOUT=0.5 ./run_dual_gpu.sh
```

**Dual GPU default configuration:**
- Batch size: 256 per GPU (512 total effective batch size)
- Hidden dims: [1024, 512, 256] (larger capacity)
- Residual blocks: 4 per layer
- Dropout: 0.4 (higher regularization)
- BERT fine-tuning: Enabled by default
- Learning rates: 3e-5 (classifier), 1e-5 (encoder)
- Uses PyTorch DistributedDataParallel (DDP)

**Key benefits:**
- ~2x training speedup (near-linear scaling)
- Larger batch sizes (512 total) for more stable training
- Increased model capacity with more memory available

### Hyperparameter Tuning

Automated hyperparameter search using Optuna:

```bash
cd claudette_classifier

# Default: 50 trials
./run_hyperparameter_tuning.sh

# More trials for better optimization
N_TRIALS=100 ./run_hyperparameter_tuning.sh

# Resume existing study
STUDY_NAME="my_tuning" N_TRIALS=50 ./run_hyperparameter_tuning.sh
```

**Hyperparameters optimized:**
- Learning rate (classifier): 1e-5 to 1e-3 (log scale)
- Learning rate (encoder): 5e-6 to 5e-5 (log scale)
- Dropout: 0.2 to 0.6
- Batch size: 64, 128, 256, 512
- Hidden dimensions: Various combinations ([512-1024], [256-512], [128-256])
- Residual blocks: 2 to 5
- Focal loss alpha: 0.2 to 0.4
- Focal loss gamma: 1.5 to 2.5

**Results:**
- Best parameters and F1 score
- Hyperparameter importance ranking
- Saved to `results/claudette_classifier/hyperparameter_tuning_results.json`

### Advanced Options

```bash
# Use weighted BCE instead of Focal Loss
uv run python -m claudette_classifier.main --no-focal-loss

# Disable oversampling (balanced batches)
uv run python -m claudette_classifier.main --no-oversampling

# Freeze encoder (only train classifier)
uv run python -m claudette_classifier.main --freeze-encoder

# Custom architecture
uv run python -m claudette_classifier.main \
    --hidden-dims 768 384 192 \
    --num-residual-blocks 4 \
    --dropout 0.4

# Custom learning rates
uv run python -m claudette_classifier.main \
    --lr 5e-5 \
    --encoder-lr 1e-6

# Use specific device
uv run python -m claudette_classifier.main --device cuda
# or: --device mps (Apple Silicon), --device cpu
```

### All CLI Options

```
--batch-size INT              Batch size (default: 32)
--lr FLOAT                    Classifier learning rate (default: 2e-5)
--encoder-lr FLOAT            Encoder learning rate (default: 1e-5)
--epochs INT                  Number of epochs (default: 50)
--device {auto,cuda,mps,cpu}  Device to use (default: auto)
--no-focal-loss               Disable Focal Loss (use weighted BCE)
--no-oversampling             Disable minority class oversampling
--no-class-weights            Disable class weighting
--hidden-dims INT [INT ...]   MLP hidden dimensions (default: 512 256 128)
--num-residual-blocks INT     Residual blocks per layer (default: 3)
--dropout FLOAT               Dropout rate (default: 0.3)
--freeze-encoder              Freeze BERT weights (only train classifier)
```

## Results

Training outputs:
- **Best model**: `results/claudette_classifier/best_model.pt`
- **Training history**: `results/claudette_classifier/training_results.json`

Metrics tracked:
- Accuracy
- Precision (positive predictive value)
- Recall (sensitivity, true positive rate)
- F1 Score (harmonic mean of precision and recall)
- AUROC (Area Under ROC Curve)
- Confusion Matrix

## Implementation Details

### Class Imbalance Handling

1. **Focal Loss**:
   - Reduces loss for well-classified examples
   - Focuses training on hard-to-classify examples
   - Especially effective for minority class (unfair clauses)

2. **Oversampling**:
   - Weighted random sampler in training loader
   - Each sample weighted by inverse class frequency
   - Ensures balanced batches during training

3. **Class Weights**:
   - Alternative to Focal Loss
   - Directly weights loss by class frequency
   - `weight_unfair / weight_fair ≈ 8.1`

### Training Features

- **Early Stopping**: Patience of 10 epochs on validation F1
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor 0.5, patience 5)
- **Gradient Clipping**: Max norm 1.0
- **Separate Learning Rates**: Lower LR for pre-trained encoder
- **Best Model Saving**: Saved when validation F1 improves

### Evaluation

Validation during training:
```python
Val - Acc: 0.9245, P: 0.8123, R: 0.7456, F1: 0.7774, AUROC: 0.9102
```

Final test evaluation includes:
- Per-class metrics (precision, recall, F1)
- Confusion matrix
- Classification report

## File Structure

```
claudette_classifier/
├── __init__.py                   # Package exports
├── config.py                     # Configuration dataclass
├── data_loader.py                # Dataset loading and splitting
├── encoder.py                    # Legal-BERT encoder
├── model.py                      # Deep residual MLP
├── loss.py                       # Focal Loss and weighted BCE
├── train.py                      # Training loop with early stopping
├── evaluate.py                   # Evaluation metrics
├── inference.py                  # Inference utilities
├── main.py                       # CLI entry point
├── hyperparameter_tuning.py      # Optuna hyperparameter tuning
├── run_dual_gpu.sh               # Dual GPU training script (DDP)
├── run_hyperparameter_tuning.sh  # Hyperparameter tuning script
└── README.md                     # This file
```

## Example Workflow

```bash
# 1. Train with all imbalance techniques
uv run python -m claudette_classifier.main --epochs 50

# 2. Experiment with frozen encoder (faster training)
uv run python -m claudette_classifier.main \
    --freeze-encoder \
    --epochs 30 \
    --lr 1e-4

# 3. Ablation study: disable one technique at a time
uv run python -m claudette_classifier.main --no-focal-loss
uv run python -m claudette_classifier.main --no-oversampling
uv run python -m claudette_classifier.main --no-class-weights

# 4. Check results
cat results/claudette_classifier/training_results.json
```

## Distributed Training Details

The dual GPU script (`run_dual_gpu.sh`) uses **PyTorch DistributedDataParallel (DDP)**:

1. **Process Spawning**: `torchrun` launches 2 processes (one per GPU)
2. **Model Replication**: Each GPU maintains a full copy of the model
3. **Data Parallelism**: Each GPU processes different batches
4. **Gradient Synchronization**: Gradients are averaged across GPUs after backward pass
5. **Efficiency**: Near-linear scaling (~2x speedup with 2 GPUs)

**Implementation details:**
- `DistributedSampler` ensures each GPU sees different data
- Only rank 0 (main process) prints logs and saves checkpoints
- Batch size is per GPU (effective batch = batch_size × num_GPUs)
- Oversampling is disabled with DDP (incompatible with DistributedSampler)
- Set epoch for sampler to ensure proper shuffling across epochs

## Troubleshooting

### Model Predicting Only Majority Class

**Symptoms:**
- Validation F1 = 0.0
- Precision/Recall = 0.0
- Accuracy ~89% (just predicting all fair)

**Solutions:**
1. **Fine-tune BERT encoder**: Use `--train-encoder` or `TRAIN_ENCODER=true`
   - Frozen BERT may not capture task-specific features
2. **Adjust focal loss parameters**:
   ```bash
   # Increase focus on minority class
   # Edit config.py: focal_alpha=0.35, focal_gamma=2.5
   ```
3. **Increase model capacity**: Larger hidden dims with dual GPU
   ```bash
   cd claudette_classifier
   HIDDEN_DIMS="2048 1024 512" ./run_dual_gpu.sh
   ```
4. **Run hyperparameter tuning**: Find optimal configuration
   ```bash
   cd claudette_classifier
   N_TRIALS=100 ./run_hyperparameter_tuning.sh
   ```

### Out of Memory

**Solutions:**
1. Reduce batch size: `BATCH_SIZE=64 ./run_dual_gpu.sh`
2. Freeze BERT encoder: `TRAIN_ENCODER=false ./run_dual_gpu.sh`
3. Reduce model capacity: `HIDDEN_DIMS="512 256 128"`
4. Use gradient accumulation (TODO: not yet implemented)
5. Enable mixed precision training (TODO: not yet implemented)

### Slow Training

**Solutions:**
1. Use dual GPU script for 2x speedup: `cd claudette_classifier && ./run_dual_gpu.sh`
2. Only fine-tune BERT if necessary (frozen is faster)
3. Use larger batch sizes: `BATCH_SIZE=512 ./run_dual_gpu.sh`
4. Reduce number of workers: Edit `data_loader.py` (default: 4)

### Poor Validation Performance

**Solutions:**
1. Run hyperparameter tuning first
2. Increase training epochs: `EPOCHS=200 ./run_dual_gpu.sh`
3. Fine-tune BERT: `TRAIN_ENCODER=true`
4. Check for overfitting:
   - If train >> val: increase dropout, add regularization
   - If train ≈ val and both low: increase capacity, more epochs
5. Verify data splits are stratified (should be automatic)

## Performance Tips

1. **Fine-tune BERT**: Significantly improves performance but requires:
   - More memory (~2-3x)
   - Longer training time (~2x)
   - Lower learning rate for encoder (1e-5)

2. **Larger Batch Sizes**: Dual GPU allows 256-512 per GPU
   - More stable gradients
   - Better batch normalization statistics
   - Faster convergence

3. **Higher Dropout**: 0.4-0.5 helps prevent overfitting when fine-tuning
   - Start with 0.3 for frozen BERT
   - Increase to 0.4-0.5 for fine-tuned BERT

4. **Learning Rates**:
   - Encoder: 1e-5 to 5e-5 (preserve pre-training)
   - Classifier: 2e-5 to 1e-4 (can be higher)
   - Use learning rate warmup for large batch sizes (TODO)

5. **Early Stopping**: Monitor validation F1, patience=10-15 epochs
   - Prevents overfitting
   - Saves training time

6. **Hyperparameter Tuning**: Run 50-100 trials
   - Automated search finds optimal configuration
   - Much better than manual tuning
   - Use best parameters for final training

## Notes

- **Memory**:
  - Frozen BERT: ~500MB per GPU
  - Fine-tuned BERT: ~1.5GB per GPU
  - Dual GPU with large model: ~3-4GB per GPU
- **Training time** (single GPU):
  - Frozen BERT: ~5-10 min/epoch
  - Fine-tuned BERT: ~10-20 min/epoch
- **Training time** (dual GPU):
  - Fine-tuned BERT: ~5-10 min/epoch (~2x speedup)
- **Recommended**: Use dual GPU for fine-tuning, single GPU for frozen BERT
- **Best practices**:
  - Start with hyperparameter tuning
  - Monitor validation metrics (F1 and AUROC)
  - Check confusion matrix for bias toward majority class
  - Use early stopping to prevent overfitting
  - Fine-tune BERT for best results

## References

- **Legal-BERT**: Chalkidis et al. "LEGAL-BERT: The Muppets straight out of Law School" (2020)
- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- **Claudette Dataset**: Lippi et al. "CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service" (2019)
