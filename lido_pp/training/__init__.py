"""Training utilities for LID-O++."""

from lido_pp.training.data_prep import (
    InstructionDataset,
    FlowMatchingBatch,
    FlowMatchingDataLoader,
    LatentFlowDataLoader,
    load_ape_instructions,
    load_gsm8k_dataset,
    create_train_val_split,
)
from lido_pp.training.checkpointing import (
    CheckpointManager,
    MetricsLogger,
    save_training_results,
    load_training_results,
)
from lido_pp.training.trainer import (
    LIDOPPTrainer,
    SimplifiedTrainer,
    TrainingState,
)
from lido_pp.training.ddp_utils import (
    setup_ddp,
    cleanup_ddp,
    create_ddp_dataloader,
    wrap_model_ddp,
    save_checkpoint_ddp,
    load_checkpoint_ddp,
    reduce_tensor,
    gather_tensors,
    DDPTrainingContext,
    print_rank0,
)
from lido_pp.training.alpaca_dataset import (
    load_alpaca_dataset,
    load_ultrachat_dataset,
    AlpacaInstructionDataset,
    EmbeddingDataset,
    FlowMatchingDataset,
)

__all__ = [
    # Data
    "InstructionDataset",
    "FlowMatchingBatch",
    "FlowMatchingDataLoader",
    "LatentFlowDataLoader",
    "load_ape_instructions",
    "load_gsm8k_dataset",
    "create_train_val_split",
    # Alpaca/UltraChat
    "load_alpaca_dataset",
    "load_ultrachat_dataset",
    "AlpacaInstructionDataset",
    "EmbeddingDataset",
    "FlowMatchingDataset",
    # DDP utilities
    "setup_ddp",
    "cleanup_ddp",
    "create_ddp_dataloader",
    "wrap_model_ddp",
    "save_checkpoint_ddp",
    "load_checkpoint_ddp",
    "reduce_tensor",
    "gather_tensors",
    "DDPTrainingContext",
    "print_rank0",
    # Checkpointing
    "CheckpointManager",
    "MetricsLogger",
    "save_training_results",
    "load_training_results",
    # Trainer
    "LIDOPPTrainer",
    "SimplifiedTrainer",
    "TrainingState",
]
