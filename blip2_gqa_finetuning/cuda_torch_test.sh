#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --output=test_cuda_%j.out
#SBATCH --error=test_cuda_%j.err
#SBATCH --partition=GPU
#SBATCH --account=perun2501146
#SBATCH --qos=perun2501146
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:05:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate blip2

python - << 'EOF'
import sys
print(f"Python : {sys.version}")

import torch
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version  : {torch.version.cuda}")
print(f"cuDNN version : {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    print(f"VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Allocate a small tensor on GPU to confirm compute works
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.randn(1000, 1000, device="cuda")
    z = x @ y
    print(f"Matrix multiply on GPU: OK  (result shape {z.shape})")
else:
    print("ERROR: CUDA not available — check PyTorch wheel vs driver version")

import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
mem = pynvml.nvmlDeviceGetMemoryInfo(h)
print(f"pynvml VRAM   : {mem.total / 1024**3:.1f} GB total, {mem.used / 1024**3:.1f} GB used")
pynvml.nvmlShutdown()
print("pynvml        : OK")
EOF