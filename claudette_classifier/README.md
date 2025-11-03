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

### Basic Training

```bash
# Train with default settings (Focal Loss + oversampling)
uv run python -m claudette_classifier.main

# Train for 30 epochs with batch size 64
uv run python -m claudette_classifier.main --epochs 30 --batch-size 64
```

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
├── __init__.py           # Package exports
├── config.py             # Configuration dataclass
├── data_loader.py        # Dataset loading and splitting
├── encoder.py            # Legal-BERT encoder
├── model.py              # Deep residual MLP
├── loss.py               # Focal Loss and weighted BCE
├── train.py              # Training loop with early stopping
├── evaluate.py           # Evaluation metrics
├── main.py               # CLI entry point
└── README.md             # This file
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

## Notes

- **Memory**: Legal-BERT (110M params) + MLP (~1M params) ≈ 500MB GPU memory
- **Training time**: ~10-15 min/epoch on GPU, ~45-60 min/epoch on CPU
- **Recommended**: Use GPU (CUDA/MPS) for faster training
- **Best practices**:
  - Start with default settings
  - Monitor validation metrics (F1 and AUROC)
  - Check confusion matrix for bias toward majority class
  - If overfitting: increase dropout, add weight decay, freeze encoder
  - If underfitting: increase model capacity, decrease regularization

## References

- **Legal-BERT**: Chalkidis et al. "LEGAL-BERT: The Muppets straight out of Law School" (2020)
- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- **Claudette Dataset**: Lippi et al. "CLAUDETTE: an automated detector of potentially unfair clauses in online terms of service" (2019)
