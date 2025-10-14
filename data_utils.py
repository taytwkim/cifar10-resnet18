import random, torch
import numpy as _np
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from ddp_utils import ddp_barrier, ddp_is_initialized

# The known mean and std of the CIFAR-10 training set, used to normalize the dataset.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

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

def maybe_download_cifar10(root: str, is_main: bool):
    # Only rank-0 attempts the download; others wait on a barrier.
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
    test = datasets.CIFAR10(root=data_dir, train=False, download=False, transform=tf_test)
    
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