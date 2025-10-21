import os, torch, torch.distributed as dist

"""
Torch Distributed 101

Ranks and World Size
  - World size                  total number of processes participating (usually = total #GPUs). Example: 2 nodes × 4 GPUs each → world_size = 8.
  - Global rank (RANK)          ID of a process in [0 … world_size-1]. Rank 0 is “main.”
  - Local rank (LOCAL_RANK)     index of the GPU on this node that a process uses (e.g., 0–3 on a 4-GPU box).
  - Node rank                   which node you are on among all nodes (0…nnodes-1). Used only at launch; PyTorch mostly cares about RANK/LOCAL_RANK.

Data Sharding (DistributedSampler)
  - Split the training dataset evenly across ranks so each process sees a disjoint shard each epoch.
  - For dataset size N and world_size = P, each rank gets ~ceil(N/P) indices (the sampler will pad if needed when drop_last=False).
  - Validation/test usually don’t need sharding—you do eval on rank 0 only.

Barriers
  - A synchronization point. Every rank in the group must reach the barrier; each process blocks there until all others arrive, then they all continue.

Backend (NCCL or gloo)
  - Backends are "how processes talk".
  - NCCL is the standard backend for GPUs. We don’t call NCCL directly. Pick nccl backend in PyTorch, and PyTorch uses NCCL under the hood.
  - gloo is the CPU backend if we don't have GPU.

DDP (DistributedDataParallel)
  - A wrapper module for distributed training.
"""

def ddp_is_available():
    return dist.is_available()  # checks whether PyTorch build has the torch.distributed package

def ddp_is_initialized():
    return ddp_is_available() and dist.is_initialized()

def ddp_setup():
    # If the env var TORCH_BACKEND is set, use it. Otherwise, default to nccl when CUDA GPUs are available (fastest for GPU), else gloo (CPU/MPS-safe).
    backend = os.environ.get("TORCH_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    
    if ddp_is_available() and not ddp_is_initialized() and ("RANK" in os.environ or "WORLD_SIZE" in os.environ):
        """
        A process group is the communication center that processes join to talk to each other (all-reduce, broadcast, barrier, etc.)
        The launcher (torchrun/Slurm) starts N separate processes. init_process_group(...) makes those processes join the same communicator, 
        gives them rank/world_size, and sets up the backend (NCCL/Gloo) + rendezvous so they can do all-reduce, broadcast, barrier, etc.
        """
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank() if ddp_is_initialized() else 0
    world_size = dist.get_world_size() if ddp_is_initialized() else 1

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device, world_size, rank, (rank == 0)

def ddp_cleanup():
    if ddp_is_initialized():
        dist.destroy_process_group()

def ddp_barrier():
    if ddp_is_initialized():
        dist.barrier()