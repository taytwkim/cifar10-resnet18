import os, torch

def save_ckpt(path, model, opt, sched, epoch, best_acc, is_main):
    """
    Write a training checkpoint
    """
    if not is_main:
        return
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
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