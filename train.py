import argparse, os, time, math, torch
from torch import nn
from torchvision import models
import torch.distributed as dist
from ddp_utils import ddp_setup, ddp_cleanup, ddp_is_initialized
from data_utils import set_seed, log_device_info, maybe_download_cifar10, make_loaders, accuracy
from ckpt_utils import save_ckpt, load_ckpt
from contextlib import nullcontext

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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if use_cuda else None    # device_ids ensures correct GPU is used per process
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

    # AMP
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