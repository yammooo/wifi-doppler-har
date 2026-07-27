# Wi-Fi Doppler HAR

## SHARP Reproduction

We reproduced the SHARP paper training setup with a shared single-antenna
classifier and decision fusion across four antenna streams. The reproduction
notebook is `notebooks/sharp_reproduction.ipynb`.

At epoch 25, the run reached:

```text
train_loss      : 0.4856
train_sum_acc   : 0.8698
train_sharp_acc : 0.8779
val_sum_acc     : 0.8810
val_sharp_acc   : 0.8804
best_val_acc    : 0.9553 @ epoch 21
```

![SHARP reproduction curves](imgs/sharp_reproduction_curves.png)

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

The training CLI probes cuDNN before loading the datasets. If Linux resolves
incompatible system-wide and environment-provided cuDNN sublibraries, it emits
a warning and keeps the run on the GPU using PyTorch's native CUDA convolution
implementation. This fallback can be slower than a correctly isolated cuDNN
installation.

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

## CSI-to-Doppler Training

The CSI-to-Doppler student is trained from the command line rather than a
notebook. The baseline configuration expects raw CSI in `data/CSI-80Mhz` and
paired SHARP targets in `data/doppler_traces_pi`. Convert those compressed
source files once into the memory-mapped training format:

```bash
conda activate wifi-doppler-har
python scripts/convert_csi_doppler_memmap.py
```

The converter preserves all 242 cleaned CSI subcarriers and direct float32
Doppler targets under `data/csi_doppler_memmap`. Conversion is resumable: files
already completed are skipped unless `--overwrite` is provided.

The downloaded `S*` Doppler archive is suitable for standalone SHARP
classification, but most recordings are not temporally aligned with the
available `AR-*` raw MAT files. Do not use it as a raw-to-Doppler target
source. Generate canonical, aligned AR and PC targets from the exact raw files:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python src/preprocessing/preprocess_sharp.py \
    --input-root data/CSI-80Mhz \
    --phase-root data/sharp_phase_recomputed \
    --processed-root data/sharp_processed_recomputed \
    --output-root data/doppler_traces_recomputed \
    --subsets AR,PC \
    --include-empty \
    --jobs 16
```

`--subsets` accepts exact names, family selectors `AR`, `PC`, `PI`, or `all`.
The H-estimation stage is checkpointed and completed outputs are reused unless
`--force` is supplied. Start with `--subsets AR-1a --limit 1 --jobs 1` and the
same command for `PC-1a` before launching the full run. The output manifest
records the source paths, shapes, preprocessing geometry, and Git revision.

`--jobs` uses local worker processes for both H estimation and post-processing,
so it must not exceed the CPUs allocated to one node. H estimation resumes at
completed antenna-stream boundaries by default. Packet-level checkpoints are
available with `--checkpoint-every N`, but they rewrite large partial arrays
and should be used only for highly preemptible runs. Keep BLAS thread counts at
one as shown above to avoid nested CPU oversubscription. On the DEI cluster,
put `--phase-root`, `--processed-root`, and other temporary files under `/ext`.

After generation completes, append the canonical AR and PC recordings to the
prepared memmap:

```bash
python scripts/convert_csi_doppler_memmap.py \
    --raw-root data/CSI-80Mhz \
    --doppler-root data/doppler_traces_recomputed \
    --output-root data/csi_doppler_memmap \
    --scenarios AR PC \
    --append
```

The chronological experiment record, linked W&B runs, negative results, and
current hypotheses are maintained in
[`docs/research/csi_to_doppler_log.md`](docs/research/csi_to_doppler_log.md).

Reusable training configs are provided for the legacy 1D U-Net, the 1D U-Net
with a spatial 2D head, its shared single-antenna variant, and the full 2D
decoder:

- `pi_cross_domain_unet1d_legacy.yaml`
- `pi_cross_domain_unet1d_spatial_head.yaml`
- `pi_cross_domain_unet2d_decoder.yaml`
- `pi_cross_domain_unet1d_spatial_head_motion_aware.yaml`
- `pi_cross_domain_unet1d_spatial_head_motion_aware_full_subcarriers.yaml`
- `pi_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware.yaml`
- `pi_ar_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware_full_subcarriers.yaml`

Treat experiment configs as immutable. Copy one to a descriptively named file
before changing architecture, input selection, loss, or dataset splits.

For example, train the full 2D decoder with:

```bash
conda activate wifi-doppler-har
python scripts/train_csi_to_doppler.py \
    --config configs/csi_to_doppler/pi_cross_domain_unet2d_decoder.yaml
```

Train the 1D U-Net with spatial 2D head and motion-aware objective using:

```bash
python scripts/train_csi_to_doppler.py \
    --config configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_motion_aware.yaml
```

Configuration values can be overridden without editing the YAML:

```bash
python scripts/train_csi_to_doppler.py \
    --config configs/csi_to_doppler/pi_cross_domain_unet2d_decoder.yaml \
    --set training.learning_rate=0.0005 \
    --set wandb.mode=disabled
```

Run the fixed-batch overfit diagnostic with:

```bash
python scripts/overfit_csi_to_doppler.py \
    --config configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml
```

Resume an interrupted local run from its complete training checkpoint:

```bash
python scripts/train_csi_to_doppler.py \
    --config configs/csi_to_doppler/pi_cross_domain_unet2d_decoder.yaml \
    --resume experiments/runs/<run-id>/training/latest.pt
```

Each run stores the best inference checkpoint at `model.pt`, resumable state
at `training/latest.pt`, and resolved configuration, pairing reports, metric
history, and final test metrics under `experiments/runs/<run-id>`.

The default trainer overlaps recording I/O and batch construction with GPU
work using a bounded prefetch thread, pinned host memory, and a CUDA transfer
stream. `training.prefetch_batches` controls the host-memory/performance
tradeoff. Keep W&B `watch` disabled for normal runs; gradient histogram
collection forces periodic GPU-to-CPU transfers and is intended for short
diagnostic runs only.
