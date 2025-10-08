# CIFAR-10 + ResNet18

Toy workload to learn how to train models on the cloud. ResNet18 trained on CIFAR-10.

## 🚀 Setup

1. Activate a venv.
```bash!
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install dependencies.

* If on CPU or Apple Silicon (MPS):
```bash!
pip install -r requirements.txt
```

* If on NVIDIA, prefer CUDA wheels:
```bash!
pip install torch torchvision numpy pillow --index-url https://download.pytorch.org/whl/cu121
```

3. Train model.

```bash!
python3 cifar10_resnet18.py --epochs 5 --batch-size 128
```

* On NVIDIA GPUs, enable mixed precision for speed:
```
python3 cifar10_resnet18.py --epochs 20 --batch-size 256 --amp --workers 4
```

## ☁️ Train in the Cloud

1. Open GCP console, launch Compute Engine with GPU attached (e.g., `n1-standard-4`/`T4`/`Ubuntu 22.04 LTS`).
* Allocate plenty of storage space (~50 GB to be safe). 

2. SSH to VM, check if GPU is attached.
```bash!
lspci | grep -i "nvidia"
```

3. Check if GPU driver is available.
```bash!
nvidia-smi
```

If not, [install driver](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#linux).

4. Setup venv and install dependencies.
```bash!
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install CUDA-enabled wheels (CUDA 12.1 build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow

# quick check
python - <<'PY'
import torch
print("cuda available?:", torch.cuda.is_available())
print("cuda build:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

5. Clone this repo.
```bash!
sudo apt-get install -y git
git clone https://github.com/taytwkim/cifar10-resnet18.git
cd cifar10-resnet18
```

7. Train model.
```
python3 cifar10_resnet18.py --epochs 20 --batch-size 256 --amp --workers 4
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
