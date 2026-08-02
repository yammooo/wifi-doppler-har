# Approximating SHARP Doppler Traces from Raw Wi-Fi CSI

This repository contains the code used for a deep-learning feasibility study:
can a neural network map raw complex Wi-Fi CSI directly to the Doppler
representation produced by SHARP?

The final evaluation compares:

1. the official precomputed SHARP Doppler traces;
2. a shared-antenna temporal U-Net with a coarse 2-D spectral head;
3. a shared-antenna frequency-time U-Net with a full-resolution 2-D decoder;
4. direct affine phase correction followed by the fixed SHARP Doppler
   transform.

The full-resolution student learned meaningful motion support and improved on
the smaller neural baseline, but it did not match the frozen SHARP classifier.
The affine-correction baseline remained substantially stronger.

## Repository Scope

The repository focuses on:

- pairing raw CSI with official precomputed SHARP Doppler traces;
- converting recordings to an efficient memory-mapped format;
- training and testing the reported CSI-to-Doppler students;
- reproducing the frozen SHARP classifier;
- evaluating reconstruction and downstream classifier fidelity;
- building the accompanying project report.

## Environment

Python 3.11 is recommended. Install PyTorch separately using the build
appropriate for the available CPU or CUDA device, then install the remaining
dependencies:

```bash
conda create -n wifi-doppler-har python=3.11
conda activate wifi-doppler-har

# Select the correct command for the machine from https://pytorch.org/get-started/
pip install torch
pip install -r requirements.txt
```

All commands below are run from the repository root. Training automatically
uses CUDA and AMP when available and falls back to the CPU.

## Data Layout

Place the raw 80 MHz CSI recordings and the official precomputed SHARP Doppler
traces as follows:

```text
data/
├── CSI-80Mhz/
│   ├── AR-1a/
│   ├── AR-1b/
│   └── ...
└── doppler_traces/
    ├── S1a/
    ├── S1b/
    └── ...
```

Here, a *Doppler trace* is a complete precomputed recording with four antenna
streams. Training cuts each trace into Doppler maps of shape
`[4, 340, 100]`: four antennas, 340 time frames, and 100 Doppler bins.

The official trace directories use `S*` names, while the corresponding raw
CSI directories use `AR-*` names. The dataset loader contains this mapping.
The final experiments use:

```text
S1a S1b S1c S2a S2b S3a S4a S4b S5a S6a S6b S7a
```

## Prepare the Dataset

Convert the paired recordings once to memory-mapped NumPy arrays:

```bash
python scripts/convert_csi_doppler_memmap.py \
  --raw-root data/CSI-80Mhz \
  --doppler-root data/doppler_traces \
  --output-root data/csi_doppler_memmap_legacy_sharp_ar \
  --scenarios S1a S1b S1c S2a S2b S3a S4a S4b S5a S6a S6b S7a \
  --legacy-alignment-only
```

`--legacy-alignment-only` skips recordings whose raw CSI and official Doppler
traces do not have the canonical temporal offset. The reported dataset
contains 92 aligned recordings and occupies approximately 14 GiB.

The resulting directory must contain `manifest.json` and one directory per
scenario:

```text
data/csi_doppler_memmap_legacy_sharp_ar/
├── manifest.json
├── S1a/
└── ...
```

## Reproduce the SHARP Classifier

Run:

```text
notebooks/sharp_reproduction.ipynb
```

The notebook trains the provided SHARP HAR classifier on `S1a`, `S1b`, and
`S1c`. Save its selected checkpoint, for example as:

```text
experiments/runs/sharp_baseline/checkpoint_sharp.pt
```

The classifier is trained on the `0–60%` temporal interval, selected on
`60–80%`, and evaluated on `90–100%`.

## Train the Reported Students

The student protocol uses all aligned scenarios for training on `0–80%` and
broad validation on `80–90%`. Early stopping uses `S1a/S1b/S1c` on `80–90%`
because downstream classifier evaluation is defined on those scenarios. The
final test interval is the disjoint `90–100%`.

Train the shared-antenna frequency-time U-Net with full-resolution decoder:

```bash
python scripts/train_csi_to_doppler.py \
  --config configs/csi_to_doppler/legacy_sharp_ar_unet2d_shared_antenna_full_resolution.yaml \
  --set wandb.mode=disabled
```

Train the smaller shared-antenna temporal U-Net with coarse spectral head:

```bash
python scripts/train_csi_to_doppler.py \
  --config configs/csi_to_doppler/legacy_sharp_ar_unet1d_spatial_head_shared_antenna_small.yaml \
  --set wandb.mode=disabled
```

The logging override keeps the run local. Each run directory contains:

```text
model.pt                    best inference checkpoint
training/latest.pt          complete resumable checkpoint
run.json                    run status and final summary
training/resolved_config.yaml
                            resolved configuration
training/pairing_reports.json
                            paired recording and window counts
training/history.json       epoch metrics
training/final_test.json    final test metrics
```

Resume interrupted training with the same configuration:

```bash
python scripts/train_csi_to_doppler.py \
  --config <CONFIG.yaml> \
  --resume experiments/runs/<RUN>/training/latest.pt \
  --set wandb.mode=disabled
```

## Capacity Sanity Check

The fixed-batch overfit test verifies that the model and optimizer can memorize
a small set of windows:

```bash
python scripts/overfit_csi_to_doppler.py \
  --config configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml \
  --set wandb.mode=disabled
```

This diagnostic uses the PI prepared dataset configured in the YAML. It is a
capacity check, not part of the final classifier-fidelity comparison.

## Frozen-Classifier Evaluation

Evaluate both students, the official SHARP target, raw CSI with the fixed
Doppler transform, and affine-corrected CSI with the same transform:

```bash
python scripts/evaluate_doppler_classifier_fidelity.py \
  --prepared-root data/csi_doppler_memmap_legacy_sharp_ar \
  --legacy-doppler-root data/doppler_traces \
  --classifier-checkpoint experiments/runs/sharp_baseline/checkpoint_sharp.pt \
  --student coarse_spectral=experiments/runs/<COARSE_RUN>/model.pt \
  --student full_resolution=experiments/runs/<FULL_RUN>/model.pt \
  --output experiments/runs/doppler_classifier_fidelity/results.json \
  --interval 0.9 1.0 \
  --device auto
```

The JSON output records checkpoint hashes, protocol details, reconstruction
metrics, motion-support metrics, classifier accuracy, per-class accuracy,
prediction agreement, and confusion matrices.

## Tests

Run the focused test suite:

```bash
python -m unittest \
  tests.test_csi_to_doppler_2d_shared \
  tests.test_distillation
```

The tests cover model tensor flow, antenna-sharing invariance, model
construction, data pairing, deterministic mixed-recording and
motion-stratified sampling, losses, metrics, checkpoint resume, and a CPU
training smoke test.

## Retained Files

### Entry Points and Configuration

| File | Purpose |
|---|---|
| `README.md` | Project scope, data layout, and reproduction instructions. |
| `requirements.txt` | Python dependencies other than the hardware-specific PyTorch installation. |
| `scripts/convert_csi_doppler_memmap.py` | Pairs raw CSI with official SHARP traces and writes the prepared memmap dataset. |
| `scripts/train_csi_to_doppler.py` | Main configurable student-training CLI, including validation, checkpointing, and final testing. |
| `scripts/overfit_csi_to_doppler.py` | Fixed-small-batch memorization diagnostic. |
| `scripts/evaluate_doppler_classifier_fidelity.py` | Frozen-classifier and reconstruction comparison for SHARP, neural, raw-STFT, and affine-STFT representations. |
| `src/preprocessing/preprocess_sharp.py` | Standalone SHARP preprocessing reproduction for generating Doppler traces from raw CSI; retained as a reference implementation but not used to produce the official targets in the final experiments. |
| `configs/csi_to_doppler/pi_cross_domain_unet1d_legacy.yaml` | Initial flattened temporal U-Net baseline configuration. |
| `configs/csi_to_doppler/legacy_sharp_ar_unet1d_spatial_head_shared_antenna_small.yaml` | Reported small shared-antenna temporal model with coarse spectral head. |
| `configs/csi_to_doppler/legacy_sharp_ar_unet2d_shared_antenna_full_resolution.yaml` | Reported final shared-antenna frequency-time model. |
| `configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml` | Tiny-set overfit sanity-check configuration. |

### Data Pipeline

| File | Purpose |
|---|---|
| `src/wifi_doppler/data/windowing.py` | Shared temporal split and window-index logic. |
| `src/wifi_doppler/data/recordings.py` | Lazy recording objects for compressed and prepared CSI/Doppler data. |
| `src/wifi_doppler/data/csi_to_sharp_doppler_dataset.py` | Raw-CSI/SHARP pairing, `AR-*` to `S*` scenario mapping, and aligned window loading. |
| `src/wifi_doppler/data/prepared_csi_doppler_dataset.py` | Memory-mapped version of the paired dataset used during training. |
| `src/wifi_doppler/data/doppler_dataset.py` | Doppler-only dataset used to reproduce and evaluate the SHARP classifier. |
| `src/wifi_doppler/data/download.py` | Download and extraction helper used by the reproduction notebook. |

### Models and Training

| File | Purpose |
|---|---|
| `src/wifi_doppler/models/csi_to_doppler.py` | Initial flattened temporal 1-D U-Net and reusable temporal blocks. |
| `src/wifi_doppler/models/csi_to_doppler_2d_head.py` | Temporal U-Net with a coarse 2-D spectral output head. |
| `src/wifi_doppler/models/csi_to_doppler_shared_antenna.py` | Shared-antenna wrapper for the coarse spectral-head model. |
| `src/wifi_doppler/models/csi_to_doppler_2d.py` | Earlier full 2-D decoder used by the overfit diagnostic and retained model registry. |
| `src/wifi_doppler/models/csi_to_doppler_2d_shared.py` | Final shared-antenna frequency-time encoder and full-resolution 2-D decoder. |
| `src/wifi_doppler/models/csi_to_doppler_builders.py` | Architecture registry used to construct models from saved configs and checkpoints. |
| `src/wifi_doppler/models/sharp.py` | Provided single-antenna SHARP HAR classifier and four-antenna decision fusion. |
| `src/wifi_doppler/training/distillation.py` | Recording-aware batching, motion-stratified sampling, losses, metrics, device prefetching, and checkpoint state. |

### Reproduction and Tests

| File | Purpose |
|---|---|
| `notebooks/sharp_reproduction.ipynb` | Reproduces the frozen SHARP classifier used for downstream evaluation. |
| `tests/test_csi_to_doppler_2d_shared.py` | Focused tests for the final architecture and model registry. |
| `tests/test_distillation.py` | Data, batching, loss, metric, smoke-training, and resume tests. |
