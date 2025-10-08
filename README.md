# CIFAR-10 + ResNet18

Toy workload to learn how to train models on the cloud. ResNet18 trained on CIFAR-10.

## 🚀 Setup

1. Activate a venv
```bash!
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies

* If on CPU or Apple Silicon (MPS):
```bash!
pip install -r requirements.txt
```

* If on an NVIDIA GPU (CUDA), prefer CUDA wheels:
```bash!
pip install torch torchvision numpy pillow --index-url https://download.pytorch.org/whl/cu121
```

3. Run

```bash!
python3 cifar10_resnet18.py --epochs 5 --batch-size 128
```

* On NVIDIA GPUs, enable mixed precision for speed:
```
python3 cifar10_resnet18.py --epochs 20 --batch-size 256 --amp --workers 8
```

## 📁 Data and Output Directories

* `data/` [Not tracked by git]
    * `cifar-10-python.tar.gz`: the original compressed dataset that `torchvision` downloads.
    * `cifar-10-batches-py/`: extracted from the tarball. This is what `torchvision.datasets.CIFAR10` actually reads.
        * `data_batch_1` … `data_batch_5`: 5 training batches - 10,000 images each.
        * `test_batch`: 10,000 test samples.

* `artifacts/` [Empty directory tracked by git]
    * `*_final_weights.pt` (weights only)
        * Use for inference or fine-tuning from scratch LR.
        * Contains just `model.state_dict()`.
    * `*_best.pt` (full checkpoint)
        * Use to resume training with identical optimizer dynamics.
        * Contains model weights, optimizer state (e.g., momentum), scheduler state (e.g., where you are on the LR curve), plus metadata like epoch and best accuracy.
