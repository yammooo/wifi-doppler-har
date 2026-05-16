# Wi-Fi Doppler HAR

## Environment Setup

This project keeps the shared Python dependencies separate from the PyTorch
installation. PyTorch depends on the available hardware, so each contributor
should install the build that matches their machine.

First, create and activate a Python 3.11 environment, then install the common
dependencies:

```powershell
pip install -r requirements.txt
```

Then install PyTorch using one of the options below.

For CPU-only machines:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA CUDA machines, install the CUDA build that matches your driver. For
example, for CUDA 12.1:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

On Google Colab, PyTorch is usually already installed. In a notebook, run:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

If Colab is missing any project dependency, install only the shared
requirements:

```python
!pip install -r requirements.txt
```

Training code should use automatic device selection by default:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
