import argparse, os, random, time, math
import torch
import numpy as _np
import torch.distributed as dist
from contextlib import nullcontext
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms, models

# These are the known mean and std of the CIFAR-10 training set, used to normalize the dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

"""
Torch Distributed Basics

Ranks and World Size
  - World size                  total number of processes participating (usually = total #GPUs). Example: 2 nodes × 4 GPUs each → world_size = 8.
  - Global rank (RANK)          ID of a process in [0 … world_size-1]. Rank 0 is “main.”
  - Local rank (LOCAL_RANK)     index of the GPU on this node that a process uses (e.g., 0–3 on a 4-GPU box).
  - Node rank                   which node you are on among all nodes (0…nnodes-1). Used only at launch; PyTorch mostly cares about RANK/LOCAL_RANK.

Data Sharding (DistributedSampler)
  - Split the training dataset evenly across ranks so each process sees a disjoint shard each epoch.
  - For dataset size N and world_size = P, each rank gets ~ceil(N/P) indices (the sampler will pad if needed when drop_last=False).
  - Validation/test usually don’t need sharding—you do eval on rank 0 only.

DDP (DistributedDataParallel)
  - A wrapper module that handles gradient averaging.

Backend (NCCL)
  - Backends are "how processes talk".
  - NCCL is the standard backend for GPUs.
  - We don’t call NCCL directly. Pick nccl backend in PyTorch, and PyTorch uses NCCL under the hood.
"""

# ---------- DISTRIBUTED UTILS ----------
def ddp_is_available():
    return dist.is_available()

def ddp_is_initialized():
    return ddp_is_available() and dist.is_initialized()

def ddp_setup():
    """
    Initialize torch.distributed if launched via torchrun.
    Sets CUDA device from LOCAL_RANK when available.
    Returns (device, world_size, rank, is_main)
    """
    backend = os.environ.get("TORCH_BACKEND", "nccl" if torch.cuda.is_available() else "gloo")
    
    if ddp_is_available() and not ddp_is_initialized() and ("RANK" in os.environ or "WORLD_SIZE" in os.environ):
        dist.init_process_group(backend=backend)
    
    rank = dist.get_rank() if ddp_is_initialized() else 0
    world_size = dist.get_world_size() if ddp_is_initialized() else 1

    # choose device
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

# ---------- UTILITIES ----------
def set_seed(seed: int, rank_offset: int = 0):
    s = seed + rank_offset
    random.seed(s)                              # seeds Python's built-in RNG
    torch.manual_seed(s)                        # seeds PyTorch CPU RNG
    torch.cuda.manual_seed_all(s)               # seeds CUDA RNG
    torch.backends.cudnn.deterministic = False  # allow cuDNN to use fast, non-deterministic kernels; fast but may not be reproducible
    torch.backends.cudnn.benchmark = True       # let cuDNN autotune the fastest kernel for your input shapes

def log_device_info(device: torch.device, amp_flag: bool, world_size: int, rank: int):
    """
    Print summary of runtime environment and accelerator
    """
    print(f"[env] torch={torch.__version__}")

    try:
        print(f"[env] numpy={_np.__version__}")
    except Exception:
        print("[env] numpy=NOT INSTALLED")
    
    print(f"[dist] world_size={world_size} rank={rank} main={rank==0}")
    
    """
    NVIDIA GPU → prints CUDA info and whether AMP is on.
    Apple Silicon Mac → prints MPS info and notes AMP is disabled.
    Otherwise → CPU.

    MPS (Metal Performance Shader) : Apple's GPU backend
    AMP (Automatic Mixed Precision) : Use mixed precision to speed up math and reduce GPU memory
    """

    if device.type == "cuda":
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(device.index or 0)
        print(f"[device] CUDA True | gpus={n} | current='{name}' (local_rank={device.index})")
        print(f"[amp] enabled={amp_flag}")

    elif device.type == "mps":
        print("[device] MPS (Apple Silicon) | AMP disabled")
    
    else:
        print("[device] CPU | AMP disabled")

def get_device():
    """
    decides which accelerator to use, in order of preference
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")

    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        dev = torch.device("mps")
    
    else:
        dev = torch.device("cpu")
    
    return dev

def maybe_download_cifar10(root: str, is_main: bool):
    """
    Only rank-0 attempts the download; others wait on a barrier.
    """
    if is_main:
        datasets.CIFAR10(root=root, train=True, download=True)
        datasets.CIFAR10(root=root, train=False, download=True)
    
    ddp_barrier()

def make_loaders(data_dir, batch_size, workers, pin_memory: bool):
    """
    Builds training & test DataLoaders; how data is transformed, batched, and fed to the GPU/CPU.
    """
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # Datasets (no download here; handled by rank-0 earlier)
    train = datasets.CIFAR10(root=data_dir, train=True,  download=False, transform=tf_train)
    test  = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=tf_test)
    
    # DDP sampler for training
    if ddp_is_initialized():
        train_sampler = DistributedSampler(train, shuffle=True, drop_last=False)
        shuffle = False
    else:
        train_sampler, shuffle = None, True

    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=shuffle, sampler=train_sampler,
        num_workers=workers, pin_memory=pin_memory, persistent_workers=workers > 0
    )

    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory, persistent_workers=workers > 0
    )
    
    return train_dl, test_dl, train_sampler

def accuracy(logits, y):
    """
    Top-1 accuracy helper; compare the predicted class with ground-truth label.
    Logits if of shape [# batch, # class], y id of shape [# batch].
    Returns [# batch] tensor of booleans.
    """
    return (logits.argmax(1) == y).float().mean().item()

def save_ckpt(path, model, opt, sched, epoch, best_acc, is_main):
    """
    Write a training checkpoint
    """
    if not is_main:
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # unwrap DDP if needed
    to_save = model.module if hasattr(model, "module") else model

    torch.save({
        "model": to_save.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "epoch": epoch,
        "best_acc": best_acc,
    }, path)

def load_ckpt(path, model, opt=None, sched=None, map_location="cpu"):
    """
    Load a training checkpoint
    """
    blob = torch.load(path, map_location=map_location)

    # load into (possibly) wrapped model
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(blob["model"])
    
    if opt and "opt" in blob and blob["opt"] is not None:
        opt.load_state_dict(blob["opt"])
    
    if sched and "sched" in blob and blob["sched"] is not None:
        sched.load_state_dict(blob["sched"])
    
    return blob.get("epoch", 0), blob.get("best_acc", 0.0)

# ---------- TRAIN MODEL ----------
def main(args):
    device, world_size, rank, is_main = ddp_setup()
    set_seed(args.seed, rank_offset=rank)
    use_cuda = (device.type == "cuda")
    amp_on = args.amp and use_cuda
    pin_mem = use_cuda

    log_device_info(device, amp_on, world_size, rank)

    # Download once on rank-0
    maybe_download_cifar10(args.data, is_main)

    # Model
    model = models.resnet18(num_classes=10)
    model.to(device)

    if ddp_is_initialized():
        # device_ids ensures correct GPU is used per process
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if use_cuda else None
        )

    # Data
    train_dl, test_dl, train_sampler = make_loaders(args.data, args.batch_size, args.workers, pin_mem)

    # Optim, loss, sched
    base_lr = args.lr * (world_size if args.scale_lr else 1.0) # Optional linear LR scaling by world size
    opt = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Cosine schedule with warmup (step-wise)
    warmup = max(0, args.warmup)
    total_steps = args.epochs * math.ceil(len(train_dl.dataset) / (args.batch_size * world_size))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        
        t = (step - warmup) / max(1, total_steps - warmup)
        
        return 0.5 * (1 + math.cos(math.pi * t))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # AMP (PyTorch 2.x)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if use_cuda else None
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_on)) if use_cuda else (lambda: nullcontext())

    start_epoch, best_acc = 0, 0.0
    
    if args.resume and os.path.isfile(args.resume):
        # Everyone loads the same weights; optimizer/sched states too
        start_epoch, best_acc = load_ckpt(args.resume, model, opt, sched, map_location="cpu")
        
        if is_main:
            print(f"[resume] from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.3f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        # -------- train --------
        model.train()
        total, loss_sum, acc_sum = 0, 0.0, 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with autocast_ctx():
                logits = model(xb)
                loss = loss_fn(logits, yb)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            sched.step()

            bs = xb.size(0)
            total += bs

            # For logging only (local metrics). Good enough; we print on rank-0.
            loss_sum += loss.detach().item() * bs
            acc_sum += accuracy(logits.detach(), yb) * bs

        # -------- eval (rank-0 only) --------
        te_loss = te_acc = None
        if is_main:
            model.eval()
            total_t, loss_t, acc_t = 0, 0.0, 0.0

            with torch.no_grad():
                for xb, yb in test_dl:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    
                    with autocast_ctx():
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    
                    bs = xb.size(0)
                    total_t += bs
                    loss_t += loss.item() * bs
                    acc_t += accuracy(logits, yb) * bs
            
            te_loss, te_acc = loss_t / total_t, acc_t / total_t

        # Optional: broadcast eval acc to all ranks (handy for early stop, etc.)
        if ddp_is_initialized():
            acc_tensor = torch.tensor([te_acc if te_acc is not None else 0.0], device=device)
            dist.broadcast(acc_tensor, src=0)
            te_acc = acc_tensor.item()

        tr_loss = loss_sum / max(1, total)
        tr_acc  = acc_sum / max(1, total)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        
        dt = time.time() - t0

        if is_main:
            print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
                  f"| test loss {te_loss:.4f} acc {te_acc:.3f} | {dt:.1f}s")

        # Save best (rank-0 only)
        if is_main and te_acc is not None and te_acc > best_acc:
            best_acc = te_acc
            save_ckpt(os.path.join(args.out_dir, "resnet18_cifar10_best.pt"),
                      model, opt, sched, epoch, best_acc, is_main=True)

        # Optional epoch snapshots
        if is_main and args.save_every and (epoch % args.save_every == 0):
            save_ckpt(os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
                      model, opt, sched, epoch, best_acc, is_main=True)

    if is_main:
        print(f"best test acc: {best_acc:.3f}")
        # final weights only (no optimizer/sched)
        to_save = model.module if hasattr(model, "module") else model
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(to_save.state_dict(), os.path.join(args.out_dir, "resnet18_final_weights.pt"))

    ddp_cleanup()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128, help="PER-GPU batch size")
    p.add_argument("--lr", type=float, default=0.1, help="base LR (per-GPU). Use --scale-lr for linear scaling by world size")
    p.add_argument("--scale-lr", action="store_true", help="linearly scale LR by world size")
    p.add_argument("--data", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default="./artifacts")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--amp", action="store_true", help="use mixed precision on CUDA")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=500, help="warmup steps for cosine schedule")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="", help="path to checkpoint")
    p.add_argument("--save-every", type=int, default=0, help="save snapshot every N epochs (0=off)")
    args = p.parse_args()
    main(args)