"""
Distributed Data Parallel (DDP) utilities for LID-O++ training.

This module provides:
- DDP setup/cleanup functions
- Distributed dataloaders with proper samplers
- Checkpoint utilities that only save on rank 0
- Gradient synchronization helpers
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Optional, Tuple, Any, Dict
from pathlib import Path


def setup_ddp(rank: int, world_size: int, backend: str = "nccl", port: str = "12355") -> None:
    """
    Initialize DDP process group.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        port: Master port for coordination
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port

    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Set device for this process
    torch.cuda.set_device(rank)

    # Synchronize all processes
    dist.barrier()

    if rank == 0:
        print(f"[DDP] Initialized {world_size} processes with {backend} backend")


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def create_ddp_dataloader(
    dataset: Dataset,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
) -> Tuple[DataLoader, DistributedSampler]:
    """
    Create DataLoader with DistributedSampler for DDP training.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size per GPU (not global)
        rank: Process rank
        world_size: Total processes
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete batches (important for DDP)
        shuffle: Shuffle data (handled by sampler)

    Returns:
        Tuple of (DataLoader, DistributedSampler)

    Note:
        Call sampler.set_epoch(epoch) at the start of each epoch for proper shuffling.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )

    return dataloader, sampler


def wrap_model_ddp(
    model: torch.nn.Module,
    rank: int,
    find_unused_parameters: bool = False,
    broadcast_buffers: bool = True,
) -> DDP:
    """
    Wrap model in DistributedDataParallel.

    Args:
        model: PyTorch module (must already be on correct device)
        rank: Process rank (for device_ids)
        find_unused_parameters: Set True if some params not used in forward
        broadcast_buffers: Sync BatchNorm buffers across processes

    Returns:
        DDP-wrapped model
    """
    return DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=find_unused_parameters,
        broadcast_buffers=broadcast_buffers,
    )


def save_checkpoint_ddp(
    state: Dict[str, Any],
    filepath: str,
    rank: int,
) -> None:
    """
    Save checkpoint (only on rank 0).

    Args:
        state: Checkpoint state dict
        filepath: Path to save checkpoint
        rank: Process rank
    """
    if rank == 0:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, filepath)
        print(f"[Checkpoint] Saved to {filepath}")

    # Ensure all processes wait for checkpoint to be saved
    barrier()


def load_checkpoint_ddp(
    filepath: str,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint (all processes load, properly mapped).

    Args:
        filepath: Path to checkpoint
        map_location: Device to map tensors to

    Returns:
        Loaded state dict
    """
    if map_location is None:
        map_location = f"cuda:{get_rank()}"

    return torch.load(filepath, map_location=map_location, weights_only=False)


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all processes (average).

    Args:
        tensor: Tensor to reduce
        world_size: Number of processes

    Returns:
        Averaged tensor (only valid on rank 0, but returned on all)
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def gather_tensors(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Gather tensors from all processes to rank 0.

    Args:
        tensor: Tensor to gather
        world_size: Number of processes

    Returns:
        Concatenated tensor on rank 0, original tensor on others
    """
    if world_size == 1:
        return tensor

    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


class DDPTrainingContext:
    """
    Context manager for DDP training setup.

    Usage:
        with DDPTrainingContext(rank, world_size) as ctx:
            model = ctx.wrap_model(model)
            dataloader, sampler = ctx.create_dataloader(dataset, batch_size)
            # Training loop...
    """

    def __init__(self, rank: int, world_size: int, port: str = "12355"):
        self.rank = rank
        self.world_size = world_size
        self.port = port
        self.device = torch.device(f"cuda:{rank}")

    def __enter__(self):
        setup_ddp(self.rank, self.world_size, port=self.port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_ddp()

    def wrap_model(self, model: torch.nn.Module, **kwargs) -> DDP:
        """Wrap model in DDP."""
        model = model.to(self.device)
        return wrap_model_ddp(model, self.rank, **kwargs)

    def create_dataloader(self, dataset: Dataset, batch_size: int, **kwargs):
        """Create distributed dataloader."""
        return create_ddp_dataloader(
            dataset, batch_size, self.rank, self.world_size, **kwargs
        )

    def save_checkpoint(self, state: Dict[str, Any], filepath: str):
        """Save checkpoint (rank 0 only)."""
        save_checkpoint_ddp(state, filepath, self.rank)

    def print(self, *args, **kwargs):
        """Print only from rank 0."""
        if self.rank == 0:
            print(*args, **kwargs)


def print_rank0(*args, **kwargs):
    """Print only from rank 0."""
    if is_main_process():
        print(*args, **kwargs)


if __name__ == "__main__":
    # Test that imports work
    print("DDP utilities module loaded successfully")
    print(f"Available functions: setup_ddp, cleanup_ddp, create_ddp_dataloader, ...")
