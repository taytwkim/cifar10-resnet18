"""
smoke test for distributed training

The output should look like:
rank 0/2 backend=nccl device=cuda:0 sum=1.0
rank 1/2 backend=nccl device=cuda:0 sum=1.0
"""

import os
import socket
import torch
import torch.distributed as dist

def main():
    world = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Choose backend
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    # Init process group only if we are actually distributed
    if world > 1 and not dist.is_initialized():
        # Longer timeout can help on busy clusters
        dist.init_process_group(backend=backend, timeout=torch.distributed.timedelta(seconds=600))

    # Device setup
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    # Tiny collective: each rank starts with its rank id, then we all-reduce(sum)
    x = torch.tensor([rank], device=device, dtype=torch.float32)
    if world > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)

    # Helpful print
    host = socket.gethostname()
    print(f"[host {host}] rank {rank}/{world} backend={backend} device={device} sum={x.item()}", flush=True)

    # Clean exit
    if world > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
