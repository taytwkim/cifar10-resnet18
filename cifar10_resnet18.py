import argparse, os, random, time, math
import torch
import numpy as _np
from contextlib import nullcontext
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# These are the known mean and std of the CIFAR-10 training set, used to normalize the dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def set_seed(seed: int):
    random.seed(seed)                           # seeds Python's built-in RNG
    torch.manual_seed(seed)                     # seeds PyTorch CPU RNG
    torch.cuda.manual_seed_all(seed)            # seeds CUDA RNG
    
    # Deterministic=False + benchmark=True is a good perf default for CNNs w/ fixed shapes
    torch.backends.cudnn.deterministic = False  # allow cuDNN to use fast, non-deterministic kernels; fast but may not be reproducible
    torch.backends.cudnn.benchmark = True       # let cuDNN autotune the fastest kernel for your input shapes

def log_device_info(device: torch.device, amp_flag: bool):
    """
    Print summary of runtime environment and accelerator
    """
    
    print(f"[env] torch={torch.__version__}")
    
    try:
        print(f"[env] numpy={_np.__version__}")
    except Exception:
        print("[env] numpy=NOT INSTALLED")
    
    """
    NVIDIA GPU → prints CUDA info and whether AMP is on.
    Apple Silicon Mac → prints MPS info and notes AMP is disabled.
    Otherwise → CPU.

    MPS (Metal Performance Shader) : Apple's GPU backend
    AMP (Automatic Mixed Precision) : Use mixed precision to speed up math and reduce GPU memory
    """

    if device.type == "cuda":
        n = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0)
        print(f"[device] CUDA available: True | gpus={n} | current='{name}'")
        print(f"[amp] enabled={amp_flag}")
    
    elif device.type == "mps":
        print("[device] MPS (Apple Silicon) available: True | using MPS device")
        print("[amp] disabled on MPS (float32 training)")
    
    else:
        print("[device] CPU")
        print("[amp] disabled on CPU")

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
    
    train = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=tf_train)
    test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)
    
    train_dl = DataLoader(
        train, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=pin_memory,
        persistent_workers=workers > 0
    )
    
    test_dl = DataLoader(
        test, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
        persistent_workers=workers > 0
    )
    
    return train_dl, test_dl

def accuracy(logits, y):
    """
    Top-1 accuracy helper; compare the predicted class with ground-truth label.
    Logits if of shape [# batch, # class], y id of shape [# batch].
    Returns [# batch] tensor of booleans.
    """
    return (logits.argmax(1) == y).float().mean().item()
 
def save_ckpt(path, model, opt, sched, epoch, best_acc):
    """
    Write a training checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
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
    model.load_state_dict(blob["model"])
    
    if opt and "opt" in blob and blob["opt"] is not None:
        opt.load_state_dict(blob["opt"])
    
    if sched and "sched" in blob and blob["sched"] is not None:
        sched.load_state_dict(blob["sched"])
    
    return blob.get("epoch", 0), blob.get("best_acc", 0.0)

def main(args):
    set_seed(args.seed)
    
    device = get_device()
    use_cuda = (device.type == "cuda")    # are we on NVIDIA?
    amp_on = args.amp and use_cuda        # are we on AMP?
    pin_mem = use_cuda                    # pin_memory - only helps on CUDA
    
    log_device_info(device, amp_on)

    train_dl, test_dl = make_loaders(args.data, args.batch_size, args.workers, pin_mem)

    # Model
    model = models.resnet18(num_classes=10)
    model.to(device)

    # Optim, loss, sched
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Cosine schedule with warmup
    warmup = max(0, args.warmup)
    total_steps = args.epochs * math.ceil(len(train_dl.dataset) / args.batch_size)
    
    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)     # learning rate scheduler - automatically changes the optimizer's lr during training
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)     # loss function

    # New AMP API (CUDA only)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_on) if use_cuda else None
    autocast_ctx = (lambda: torch.amp.autocast("cuda", enabled=amp_on)) if use_cuda else (lambda: nullcontext())

    start_epoch, best_acc = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_acc = load_ckpt(args.resume, model, opt, sched, map_location="cpu")
        print(f"[resume] from {args.resume} at epoch {start_epoch}, best_acc={best_acc:.3f}")

    for epoch in range(start_epoch + 1, args.epochs + 1):
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
            loss_sum += loss.detach().item() * bs
            acc_sum += accuracy(logits.detach(), yb) * bs

        # -------- eval --------
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

        tr_loss, tr_acc = loss_sum / total, acc_sum / total
        te_loss, te_acc = loss_t / total_t, acc_t / total_t
        dt = time.time() - t0
        print(f"epoch {epoch:3d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| test loss {te_loss:.4f} acc {te_acc:.3f} | {dt:.1f}s")

        # Save best
        if te_acc > best_acc:
            best_acc = te_acc
            save_ckpt(os.path.join(args.out_dir, "resnet18_cifar10_best.pt"),
                      model, opt, sched, epoch, best_acc)

        # Optional epoch snapshots
        if args.save_every and (epoch % args.save_every == 0):
            save_ckpt(os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt"),
                      model, opt, sched, epoch, best_acc)

    print(f"best test acc: {best_acc:.3f}")
    torch.save(model.state_dict(), os.path.join(args.out_dir, "resnet18_final_weights.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
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