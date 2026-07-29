# CSI-to-Doppler Research Log

Last updated: 2026-07-29 (Europe/Rome)

This is the append-only research record for learning the mapping from raw CSI
to SHARP Doppler maps. Its purpose is to preserve the evidence, reasoning, and
failed attempts needed for later experiments and paper writing.

## Update Protocol

- Add new entries chronologically. Do not silently rewrite an earlier
  interpretation when later evidence changes it.
- Label statements as **Measurement**, **Observation**, **Hypothesis**, or
  **Decision**.
- Link every trained model to its W&B run and the exact config used. W&B stores
  the resolved config; repository links identify reusable configs and code.
- Record negative results and stopped/crashed runs.
- Keep metrics from different objectives separate. MSE values and the composite
  motion/Wasserstein loss are not directly comparable.
- When an earlier conclusion changes, add a dated correction and link back to
  the original entry.

Entry template:

```markdown
## YYYY-MM-DD

### Short experiment title

**Question**

**Run/config/code**

**Change**

**Measurements and observations**

**Interpretation and confidence**

**Decision**

**Next**
```

## Canonical Resources

- W&B project: [yammo-unipd/wifi-doppler-har](https://wandb.ai/yammo-unipd/wifi-doppler-har)
- Raw dataset archive: [Google Drive dataset](https://drive.google.com/file/d/1Kzepz_CIPnLmpTgrvMnge2Nxec0BAlJK/view)
- Cluster documentation: [DEI cluster overview](https://docs.dei.unipd.it/en/CLUSTER/Overview)
- Current reusable configs:
  - [Legacy 1D U-Net](../../configs/csi_to_doppler/pi_cross_domain_unet1d_legacy.yaml)
  - [1D U-Net with spatial 2D head](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head.yaml)
  - [Full 2D decoder](../../configs/csi_to_doppler/pi_cross_domain_unet2d_decoder.yaml)
  - [1D U-Net with spatial 2D head and motion-aware loss](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_motion_aware.yaml)
  - [AR motion-balanced full-resolution model](../../configs/csi_to_doppler/ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml)
- Training CLI: [train_csi_to_doppler.py](../../scripts/train_csi_to_doppler.py)
- Dataset pairing/windowing: [csi_to_sharp_doppler_dataset.py](../../src/wifi_doppler/data/csi_to_sharp_doppler_dataset.py)
- Loss and metrics: [distillation.py](../../src/wifi_doppler/training/distillation.py)
- SHARP target generation: [preprocess_sharp.py](../../src/preprocessing/preprocess_sharp.py)

The three architecture-comparison configs still use the
`motion_weighted_wasserstein` objective analyzed below. They are reproducible
architecture baselines. The separate motion-aware config implements the
corrected objective defined in the 2026-07-24 entry.

## Fixed Data Protocol

Unless a dated entry says otherwise:

- Input: raw CSI `[batch, 4 antennas, 30 subcarriers, 370 raw samples, 2 real/imag]`.
- Target: SHARP Doppler `[batch, 4 antennas, 340 time frames, 100 Doppler bins]`.
- The raw interval for a 340-frame target is 370 samples because each target
  frame uses a 31-sample window and stride 1.
- The raw preprocessing leaves 242 usable subcarriers after deleting 14
  invalid/guard/DC bins. The default student uses 30 fixed-uniform carriers.
- Train: PI-1a/PI-2a/PI-3a, 0-60%.
- Source validation: PI-1a/PI-2a/PI-3a, 60-80%.
- Target validation: PI-4a, 60-80%.
- Final target test: PI-4a, 80-100%, evaluated after restoring the best target
  validation checkpoint.
- Window size 340, stride 30, split guard 31.
- Current prepared dataset sizes:
  - train: 26,801 windows from 30 recordings;
  - source validation: 8,685 windows from 30 recordings;
  - target validation: 2,830 windows from 10 recordings.
- Adjacent windows overlap by 310/340 frames (91.2%). Window count therefore
  overstates independent data diversity.

## W&B Run Index

Dates and times in run names use Europe/Rome local time.

| Date | Run | Architecture/objective | Data path and batching | State | Key result |
|---|---|---|---|---|---|
| 2026-07-22 | [`nk1t6w2g`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/nk1t6w2g) | legacy 1D, MSE | compressed source, batch 8, recording-local | finished, 3 epochs | best target MSE 0.006207; final target MSE 0.014258; train 25 samples/s |
| 2026-07-22 | [`zl2a97qb`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/zl2a97qb) | legacy 1D, MSE | compressed source, batch 8, recording-local | finished, 1 epoch | target MSE 0.006682; train 170 samples/s |
| 2026-07-22 | [`v7ozbamb`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/v7ozbamb) | legacy 1D, MSE | compressed source, batch 256, recording-local | finished, 7 epochs | best target MSE 0.006074; strong validation oscillation |
| 2026-07-22 | [`6ebyrjjx`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/6ebyrjjx) | legacy 1D, MSE | compressed source, batch 256, host/CUDA prefetch | finished, 5 epochs | best target MSE 0.005788; train 263 samples/s |
| 2026-07-22 | [`t0vred1p`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/t0vred1p) | legacy 1D, MSE | memmap, batch 256, host/CUDA prefetch | finished, 21 epochs | best target MSE 0.005002 at epoch 11; target test MSE 0.004875; train 634 samples/s |
| 2026-07-23 | [`v4b08ztd`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/v4b08ztd) | legacy 1D, composite motion/Wasserstein | memmap, batch 256, recording-local | crashed after 11 epochs | best target loss 0.021240; vertical mean-spectrum output remained |
| 2026-07-23 | [`ppuncy34`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/ppuncy34) | legacy 1D, composite motion/Wasserstein | memmap, batch 256, 8 recordings/batch | finished, 22 epochs | best target loss 0.021125; best target MSE 0.005659; train 1,053 samples/s |
| 2026-07-23 | [`gzfqzvqv`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/gzfqzvqv) | 1D U-Net plus spatial 2D head, composite | memmap, batch 256, 8 recordings/batch | finished, 16 epochs | best target loss 0.019883; best target MSE 0.005333; train 712 samples/s |
| 2026-07-23 | [`oqqlhyvm`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/oqqlhyvm) | full 2D decoder, composite | memmap, batch 256, 8 recordings/batch | crashed after 5 epochs | best target loss 0.027407; train 75 samples/s |
| 2026-07-23/24 | [`0varxc4c`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/0varxc4c) | full 2D decoder, composite | memmap, batch 256, 8 recordings/batch | finished, 18 epochs | best target loss 0.025239 and MSE 0.007548 at epoch 8; train 78 samples/s |
| 2026-07-24 | [`cwj0myug`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/cwj0myug) | full 2D decoder, MSE, tiny overfit | 30 carriers, fixed batch of 16 windows | finished, 2,000 steps | final eval MSE 0.00003881; off-center MSE 0.00002805 |
| 2026-07-24 | [`tm8nyt19`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tm8nyt19) | full 2D decoder, MSE, tiny overfit | 242 carriers, same fixed 16 windows | finished, 2,000 steps | final eval MSE 0.00004194; off-center MSE 0.00003240 |
| 2026-07-24 | [`tio9b4ad`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tio9b4ad) | 1D U-Net plus spatial 2D head, motion-aware MSE | memmap, batch 256, 8 recordings/batch | finished, 13 epochs | best target loss 0.029761; target loss 0.031709 at epoch 13 |
| 2026-07-27 | [`ehotc8fp`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/ehotc8fp) | shared-antenna spatial-head model, full width, motion-aware MSE | AR+PC+PI, 242 carriers, batch 64 | crashed after epoch 1 | target loss 0.023761; exposed host data starvation |
| 2026-07-27 | [`6s1lm3dy`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/6s1lm3dy) | shared-antenna spatial-head model, full width, motion-aware MSE | AR+PC+PI, 242 carriers, optimized memmap iterator | crashed after epoch 3 | best target loss 0.020155; advanced-indexing copy remained |
| 2026-07-27 | [`kbjjrx1c`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/kbjjrx1c) | shared-antenna spatial-head model, small, motion-aware MSE | AR+PC+PI, 242 carriers, batch 448 on RTX 4090 | finished, 1 epoch | target loss 0.033159; exposed per-step CUDA synchronization |
| 2026-07-27 | [`3wrefrft`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/3wrefrft) | shared-antenna spatial-head model, small, motion-aware MSE | AR+PC+PI, 242 carriers, batch 768 on RTX 4090 | finished, 27 epochs | best target loss 0.018227 at epoch 26; motion remains attenuated and blurred |

The early MSE runs used historical revisions of
`pi_cross_domain_mse.yaml`. Their resolved configs remain attached to W&B.
Relevant repository snapshots include:

- [Initial trainer and MSE config at `ae4e3d4`](https://github.com/yammooo/wifi-doppler-har/blob/ae4e3d4/configs/csi_to_doppler/pi_cross_domain_mse.yaml)
- [Prefetch implementation at `18a8c84`](https://github.com/yammooo/wifi-doppler-har/commit/18a8c84)
- [Memmap implementation at `8e6b6d5`](https://github.com/yammooo/wifi-doppler-har/commit/8e6b6d5)
- [Composite loss at `7258434`](https://github.com/yammooo/wifi-doppler-har/commit/7258434)
- [Mixed-recording batches at `31afb8f`](https://github.com/yammooo/wifi-doppler-har/commit/31afb8f)
- [Full 2D decoder at `884ca37`](https://github.com/yammooo/wifi-doppler-har/commit/884ca37)
- [Spatial 2D head at `31f3cc9`](https://github.com/yammooo/wifi-doppler-har/commit/31f3cc9)
- [Architecture-specific configs at `72852e8`](https://github.com/yammooo/wifi-doppler-har/commit/72852e8)

## 2026-07-22

### Problem definition and training protocol

**Question**

Can a neural network approximate SHARP's expensive raw-CSI-to-Doppler
optimization while preserving the Doppler maps needed for HAR?

**Decision**

Train a supervised student directly against stored SHARP Doppler traces.
Notebooks remain analysis-only; training is a resumable local CLI with W&B.

**Implementation**

- Added recording-aware paired CSI/Doppler batching, AMP, Adam, validation,
  early stopping, local checkpoints, W&B metrics/artifacts, and final test
  evaluation.
- MSE was the initial objective.
- The target-validation split controls early stopping; target test remains
  untouched until training finishes.
- Checkpoint resume includes model, optimizer, scaler, RNG, epoch, step, best
  metric, history, and W&B identity.

**Code/config**

- [Implementation commit `ae4e3d4`](https://github.com/yammooo/wifi-doppler-har/commit/ae4e3d4)
- W&B runs [`nk1t6w2g`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/nk1t6w2g),
  [`zl2a97qb`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/zl2a97qb),
  and [`v7ozbamb`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/v7ozbamb)

### Baseline architecture

**Measurement**

The legacy model:

1. Flattens antenna, subcarrier, and real/imag into 240 Conv1d channels.
2. Uses a temporal 1D U-Net with residual blocks and dilated bottleneck blocks.
3. Produces 400 independent output channels (`4 antennas * 100 bins`).
4. Reshapes those channels to `[4, time, Doppler bin]` only after the last
   Conv1d.

**Observation**

The encoder and decoder model temporal locality but not Doppler-bin locality.
Each final antenna/bin channel can learn its own time-invariant bias. A
constant per-bin spectrum therefore appears as full-height vertical lines and
is an easy approximation to the training mean.

**Design decisions**

- Real and imaginary CSI remain separate input channels but are mixed jointly
  by learned convolutions.
- Interpolation was preferred over transposed convolution for temporal
  upsampling to avoid learned upsampling/checkerboard artifacts.
- Pre-activation residual blocks and temporal dilation enlarge the receptive
  field without changing sequence length.

### Initial MSE behavior

**Runs**

- [`v7ozbamb`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/v7ozbamb)
- [`t0vred1p`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/t0vred1p)

**Measurements**

- In `v7ozbamb`, target MSE was best at epoch 1 (`0.006074`) and then
  oscillated while train MSE declined.
- The final memmap run `t0vred1p` reached best target MSE `0.005002` and target
  test MSE `0.004875`.
- Visual outputs still emphasized the center ridge and did not establish that
  off-center motion was reconstructed.

**Interpretation**

Pixel-average MSE can be low because most target pixels are at the floor and
the stationary center ridge dominates. Low MSE alone is insufficient evidence
of useful motion reconstruction.

### GPU and input-pipeline optimization

**Question**

Why did GPU utilization alternate between 100% and 0%?

**Measurements**

- Batch 8 with compressed source files was I/O and CPU limited.
- Increasing batch size to 256 improved training throughput but compressed
  MAT/pickle loading still stalled the GPU.
- Host prefetch, pinned memory, a CUDA transfer stream, cuDNN benchmarking, and
  TF32 support were added.
- The dataset was converted once to uncompressed NumPy memmaps. Prepared data
  occupies approximately 17 GB and retains all 242 cleaned carriers.
- Training throughput progressed from roughly 25 samples/s in `nk1t6w2g` to
  634 samples/s in `t0vred1p`; validation exceeded 1,400 samples/s.

**Decision**

Memmap storage plus one bounded prefetch thread is the default. No worker
process or Slurm-specific trainer complexity was added.

**Code/runs**

- [Prefetch commit `18a8c84`](https://github.com/yammooo/wifi-doppler-har/commit/18a8c84)
- [Memmap commit `8e6b6d5`](https://github.com/yammooo/wifi-doppler-har/commit/8e6b6d5)
- [`6ebyrjjx`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/6ebyrjjx)
- [`t0vred1p`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/t0vred1p)

## 2026-07-23

### Composite motion/Wasserstein objective

**Question**

Can a motion-weighted active-map term and Wasserstein distance force the model
to reconstruct rare off-center motion rather than the center-biased average?

**Change**

The objective became:

```text
active SmoothL1
+ 0.1 * normalized Wasserstein over Doppler bins
+ 0.05 * raw MSE
```

Frames were weighted as `1 + 4 * off_center_mass_ratio`; only center bins
49-51 were excluded from the off-center mass.

**Rationale at the time**

- SmoothL1 was intended to reconstruct active amplitudes robustly.
- Wasserstein was intended to penalize moving Doppler power to the wrong bin
  according to bin distance.
- Raw MSE retained a weak full-map anchor.
- Multi-resolution STFT loss was not selected because the target is already a
  time-Doppler map; another spectral transform lacks a clear physical axis.

**Run**

[`v4b08ztd`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/v4b08ztd)

**Observation**

The model still produced vertical stripes and broad power around the center.
The new objective did not visibly recover off-center peaks.

**Status**

This objective is now considered flawed for the intended behavior. The
quantitative diagnosis is recorded in the 2026-07-24 entry.

### Mixed-recording batches and BatchNorm diversity

**Question**

Were recording-local batches causing BatchNorm to learn one recording's
statistics at a time?

**Change**

- Pools of eight recordings are opened once from memmaps.
- Windows are deterministically shuffled and round-robin interleaved.
- A full batch of 256 contains approximately 32 windows from each recording.
- Validation and test remain recording-local.
- Exact epoch coverage, deterministic order, resume behavior, and one memmap
  open per recording were tested.

**Measurements**

- A full production pass covered 26,801/26,801 windows in 107 batches.
- Four partial batches occur at the four pool boundaries.
- Data-only mixed throughput was approximately 857 samples/s locally.
- The mixed legacy run reached about 1,053 train samples/s.

**Run/code**

- [`ppuncy34`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/ppuncy34)
- [Commit `31afb8f`](https://github.com/yammooo/wifi-doppler-har/commit/31afb8f)

**Interpretation**

Mixed batches improved recording diversity and did not become a throughput
bottleneck. They did not eliminate validation instability or mean-spectrum
predictions, so recording-local batching was not the root cause.

### Architecture hypothesis: Doppler locality

**Observation**

The legacy final Conv1d treats Doppler bins as unrelated output channels.
Adjacent bins do not share a spatial kernel.

**Architectures implemented**

1. `unet1d_legacy`: original temporal U-Net and independent-channel head.
2. `unet1d_spatial_head`: inherits the complete legacy model and replaces only
   the output head with a coarse Doppler projection and two Conv2d layers.
3. `unet2d_decoder`: creates a coarse time-Doppler grid at the bottleneck and
   performs the complete decoder in 2D; 1D skips are broadcast as temporal
   conditioning.

**Static production-shape measurements**

| Architecture | Parameters | Conv MACs/sample | Conv output elements/sample |
|---|---:|---:|---:|
| legacy 1D | 3.245M | 0.591G | 0.885M |
| 1D plus spatial head | 3.071M | 0.561G | 1.138M |
| full 2D decoder | 2.411M | 0.740G | 3.163M |

Parameter count was misleading: the full 2D decoder is smaller in weights but
creates 3.57 times as many convolution output elements as legacy.

**Runs**

- Spatial head: [`gzfqzvqv`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/gzfqzvqv)
- Full 2D stopped/crashed run: [`oqqlhyvm`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/oqqlhyvm)
- Full 2D completed run: [`0varxc4c`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/0varxc4c)

**Measurements**

- Spatial head: best target loss `0.019883`, best target MSE `0.005333`, about
  712 train samples/s.
- Full 2D completed: best target loss `0.025239`, best target MSE `0.007548`,
  about 78 train samples/s.
- Full 2D required roughly six minutes per epoch in the observed run, versus
  less than a minute for the optimized legacy model.
- GPU utilization and memory were saturated during full 2D training. This was
  model compute/activation traffic, not input starvation.

**Decision**

Keep all three architectures as explicit configs. Do not infer that a more
spatial decoder improves the target mapping: the full 2D run was slower and
did not improve validation.

## 2026-07-24

### Config reproducibility

**Problem**

A single config file had repeatedly been edited from legacy to spatial-head to
full-2D settings. Its filename also said MSE after the objective had changed.

**Decision**

Experiment configs are immutable. New architecture, input, objective, or split
experiments receive a new descriptive file.

**Current configs**

- [pi_cross_domain_unet1d_legacy.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet1d_legacy.yaml)
- [pi_cross_domain_unet1d_spatial_head.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head.yaml)
- [pi_cross_domain_unet2d_decoder.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet2d_decoder.yaml)

All non-model settings are identical across these three files.

### Full 2D run: curve and image diagnosis

**Run**

[`0varxc4c`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/0varxc4c),
full 2D decoder, 30 carriers, mixed-recording batches, composite loss.

**Observed curves**

| Epoch | Train loss | Source-val loss | Target-val loss | Target-val MSE |
|---:|---:|---:|---:|---:|
| 1 | 0.028373 | 0.059804 | 0.025249 | 0.019222 |
| 8 | 0.017444 | 0.032716 | **0.025239** | **0.007548** |
| 12 | 0.016400 | 0.025823 | 0.027623 | 0.007575 |
| 16 | **0.015893** | 0.029412 | 0.061694 | 0.018936 |
| 18 | 0.016582 | 0.026364 | 0.035010 | 0.009948 |

**Measurements**

- Training improved rapidly through epoch 2 and then slowly through epoch 16.
  It did not plateau immediately.
- Source validation improved from `0.0598` to roughly `0.026`.
- Target validation never improved materially beyond epoch 1 and was highly
  unstable despite a deterministic 2,830-window split.
- The visual inspected after epoch 18 represented the latest model, not the
  best epoch-8 checkpoint.
- Predicted maps preserved the center ridge but missed or smudged off-center
  peaks and produced low broad energy where the target was at its floor.

**Interpretation**

This is source-domain overfitting: source learning continues while PI-4a
generalization does not. Overfitting does not require train loss to reach zero.
Train loss is also measured in training mode during updates, while validation
uses BatchNorm running statistics in evaluation mode, so their absolute gap is
not a clean generalization estimate.

### Target distribution analysis

**Method**

Using the current split protocol, every tenth unique Doppler frame was sampled
from each recording. Antennas were treated as separate frames. This avoids
counting the same frame approximately eleven times through overlapping
windows. Numbers are diagnostic approximations, not the exact window-weighted
training objective.

**Measurements**

| Statistic | Train | Target validation |
|---|---:|---:|
| sampled antenna-frames | 325,332 | 35,200 |
| mean target value | 0.12739 | 0.12920 |
| pixels equal to floor | 80.2% | 77.1% |
| mean bins/frame above 0.1 | 16.26 | 17.92 |
| mean bins/frame above 0.2 | 11.37 | 11.84 |

The marginal target distributions are similar. The target-domain failure is
not explained by a simple target-amplitude shift.

**Motion-band measurements on train**

| Excluded center half-width | Mean mass outside center | Mean configured frame weight | Frames with outside max > 0.2 |
|---:|---:|---:|---:|
| 1 (current) | 51.6% | 3.063 | 100% |
| 3 | 16.2% | 1.648 | 46.0% |
| 5 | 8.3% | 1.332 | 23.1% |
| 7 | 5.9% | 1.234 | 14.6% |
| 10 | 4.1% | 1.163 | 9.6% |

**Conclusion**

The configured `center_half_width: 1` treats the stationary FFT main lobe as
motion. Every sampled frame appears to have significant off-center motion, so
the intended motion weighting is nearly always active and does not correct the
imbalance. A half-width near 5 is a better starting definition for this target
generator, subject to validation by activity labels or downstream utility.

### Mean-spectrum baseline

**Method**

The per-antenna mean spectrum from sampled train frames was repeated at every
target-validation time step and evaluated with the current composite
objective.

**Measurements**

| Predictor | Composite loss | Active SmoothL1 | Wasserstein before 0.1 weight | MSE |
|---|---:|---:|---:|---:|
| floor everywhere | 0.08618 | 0.06259 | 0.21587 | 0.04008 |
| train mean spectrum | **0.02497** | 0.02074 | 0.03949 | 0.00559 |
| target-split mean spectrum (non-deployable diagnostic) | 0.02540 | 0.02051 | 0.04612 | 0.00550 |
| best full 2D checkpoint | 0.02524 | not logged separately | not logged separately | 0.00755 |

**Conclusion**

The full 2D network did not beat the constant train-mean spectrum on PI-4a
under the objective it was trained to optimize. Its best target MSE was also
worse than the approximate train-mean baseline MSE. This quantitatively
supports the visual mean-spectrum/center-ridge shortcut.

### Composite-loss failure analysis

**Code**

[motion_weighted_wasserstein_loss](../../src/wifi_doppler/training/distillation.py)

**Measurements and deductions**

1. `pred_active = clamp(prediction - floor, 0)` removes active-loss gradients
   below the floor. Only `0.05 * raw MSE` distinguishes the floor from a
   negative output.
2. For the train-mean baseline, raw MSE contributes approximately `0.00028` to
   a total `0.02497`, about 1% of the objective.
3. Negative predictions were common:
   - epoch-1 train negative fraction: 49.0%;
   - epoch-1 source-val negative fraction: 31.8%;
   - epoch-17 target-val negative fraction: 12.4%.
4. At epoch 1, source-val MSE was `0.2749` while source-val composite loss was
   only `0.0598`. The objective can hide very inaccurate raw outputs.
5. SmoothL1 becomes linear for errors above beta `0.1`; a missed high peak does
   not receive MSE's error-proportional gradient. Averaging over 100 bins
   further dilutes narrow peaks.
6. Low broad leakage just above the floor is cheap in SmoothL1's quadratic
   region.
7. Wasserstein normalizes away total active mass and is dominated by the
   stationary center ridge. It therefore weakly constrains absolute
   off-center power.
8. Target peak-bin MAE stayed near `0.37` bins even when off-center motion was
   visibly missing because the target's global maximum is normally the center
   ridge. Global peak MAE is not a useful motion metric here.

**Decision**

Do not use total composite loss alone to select or describe future models.
Log every component and compare against the train-mean baseline.

### Current diagnosis

The evidence supports multiple problems, ranked as follows.

1. **High confidence: objective mismatch.** The current loss rewards a
   center-biased average, weakly penalizes negative/background errors, and
   misclassifies the stationary main lobe as motion.
2. **High confidence: source-domain overfitting.** Train and source validation
   improve while PI-4a does not beat a constant target predictor.
3. **Medium confidence: BatchNorm domain sensitivity.** Source and target
   losses and negative fractions spike between epochs. Training uses batch
   statistics while target evaluation uses source-derived running statistics.
   This is consistent with the instability but has not been isolated.
4. **Medium confidence: incomplete or locally ambiguous input.** SHARP uses
   approximately every second cleaned carrier during H estimation (about 121),
   whereas the student sees 30. SHARP phase correction is also stateful across
   a complete recording, while the student sees one local 370-sample window.
5. **Medium confidence: limited independent data diversity.** The nominal
   26,801 windows come from 30 recordings with 91.2% overlap.
6. **Low confidence: raw sensor noise as the primary cause.** The targets are
   deterministic outputs of SHARP, but LASSO, phase unwrapping, framewise max
   normalization, and hard flooring can introduce algorithmic discontinuity.
   No evidence yet isolates raw measurement noise.
7. **Lower priority: model capacity.** The full 2D model reaches train loss
   `0.016`, substantially below the train-mean baseline's approximately
   `0.0253`, so it can learn source-specific structure. More capacity is not
   the first intervention.

### Proposed corrected objective

This is a proposal, not yet implemented.

1. Constrain predicted Doppler power to the target range `[floor, 1]`.
2. Make full-map MSE the primary term so large missed peaks receive larger
   gradients and all background/negative errors matter.
3. Define the stationary band initially as bins 45-55.
4. Add an off-center motion reconstruction term on frames with real target
   motion.
5. Add an explicit background-leakage penalty where the target equals the
   floor.
6. Apply low-weight Wasserstein only to off-center motion distributions on
   frames with sufficient target motion. Do not let the center ridge dominate.
7. Log full MSE, motion MSE, background leakage, Wasserstein, predicted mass,
   negative/out-of-range fractions, and motion-frame metrics separately.

Initial conceptual form:

```text
1.0 * full_map_mse
+ lambda_motion * motion_region_mse
+ lambda_background * false_active_power
+ small_lambda_wasserstein * motion_only_wasserstein
```

Weights should be chosen after measuring component scales, not copied from the
current objective.

### Required diagnostic experiments

In priority order:

1. Add exact train-mean and floor baselines to evaluation/W&B.
2. Add an evaluation-mode pass over a fixed train subset so train/validation
   comparisons use the same BatchNorm mode.
3. Run a one-batch and one-recording overfit test with primary MSE. Failure to
   fit would implicate architecture, alignment, or local input ambiguity.
4. Verify CSI-target temporal alignment independently on several recordings.
   Code inspection currently supports the `raw_start = 800 + target_frame`
   mapping, but it needs an empirical lag check.
5. Implement and ablate the corrected loss before another architecture sweep.
6. Compare BatchNorm against GroupNorm as a separate experiment.
7. Compare 30, 128, and 242 input subcarriers only after the objective is
   corrected. Do not combine this with a normalization change.
8. Evaluate downstream HAR utility:
   - frozen classifier trained on true SHARP maps, tested on generated maps;
   - separately, classifier retrained on generated maps.
   The first tests fidelity to the teacher representation; the second tests
   whether generated maps remain useful despite distribution shift.
9. Continue fixed target/prediction/error images, but add off-center-only views,
   common fixed color limits, motion energy curves, and the best-checkpoint
   image rather than only the latest epoch.

### Tiny-set overfit diagnostic

**Question**

Can the current model and training path reproduce a small set of CSI/Doppler
pairs, and does increasing the input from 30 to all 242 cleaned subcarriers
change that result?

**Scope**

A tiny-set overfit test diagnoses model, loss, data alignment, and optimization
failures. It does not establish that the CSI-to-Doppler mapping generalizes:
a sufficiently large network can memorize even incorrectly aligned or random
targets.

**Protocol**

1. Use the full 2D model under diagnosis, but train with plain full-map MSE.
   Do not use the current composite objective in this diagnostic.
2. Select 16 deterministic, non-overlapping, motion-rich windows from one
   source recording. Rank motion using target energy outside bins 45-55 so the
   test cannot pass by reproducing only the stationary center ridge.
3. Reuse the exact same complete batch for every optimizer step. Disable early
   stopping and validation, use a fixed seed, and initially disable AMP to
   remove numerical and scheduling variables.
4. Train until convergence or a fixed large step budget. Log full-map MSE,
   off-center MSE, background leakage, maximum absolute error, and fixed
   target/prediction/error maps.
5. Evaluate the fitted batch in both `model.train()` and `model.eval()` modes.
   A large eval-only regression directly implicates BatchNorm running
   statistics.
6. Repeat with the same recording, window indexes, initialization seed, and
   optimizer settings while changing only `num_subcarriers` from 30
   fixed-uniform carriers to all 242 cleaned carriers.
7. If both runs fit, increase to 64 motion-stratified windows from four
   recordings. Only after that, test held-out windows from the same recordings;
   this last step measures local generalization rather than memorization.

**Expected interpretation**

| Result | Evidence |
|---|---|
| 30 carriers cannot fit 16 windows | Optimization, objective/output, architecture, or alignment problem; more dataset data is not the next fix |
| 30 cannot fit, 242 can fit | Missing carrier information is likely important, although added parameters may also help memorization |
| Both fit in train mode but fail in eval mode | BatchNorm running-statistics problem |
| Both fit 16 windows but fail 64 windows | Capacity/optimization or strong input-target ambiguity |
| Both fit training windows but fail held-out windows from the same recording | Memorization only; investigate temporal alignment, missing full-recording phase state, and local-window ambiguity |
| Same-recording held-out windows work but PI-4a fails | Domain generalization and normalization are the primary problems |
| 242 is no better than 30 | Carrier count is unlikely to be the current bottleneck |

**Carrier-cost note**

The model input grows from `4 * 30 * 2 = 240` to
`4 * 242 * 2 = 1,936` channels. This makes the first 1x1 input projection about
8.1 times larger, but does not multiply the complete model by 8.1 because
subsequent layers retain the same channel widths. Input transfer and first
projection cost increase; the tiny diagnostic remains inexpensive.

**Decision**

Run the 30-carrier case first. The 242-carrier run is meaningful only as a
controlled second condition using exactly the same selected windows and
training setup.

**Implementation**

- Script: [overfit_csi_to_doppler.py](../../scripts/overfit_csi_to_doppler.py)
- Config: [diagnostic_overfit_unet2d.yaml](../../configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml)
- Fixed recording: `PI-1a_p03`
- Selected set: 16 deterministic, non-overlapping windows ranked by active
  target energy outside bins 45-55.
- Local outputs include the resolved config, selected window filenames and
  motion scores, metric history, final model, and W&B run identity.

Run the 30-carrier condition:

```bash
python scripts/overfit_csi_to_doppler.py \
    --config configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml
```

Run the controlled 242-carrier condition:

```bash
python scripts/overfit_csi_to_doppler.py \
    --config configs/csi_to_doppler/diagnostic_overfit_unet2d.yaml \
    --set data.num_subcarriers=242 \
    --set model.num_subcarriers=242
```

**Completed runs**

- 30 carriers: [`cwj0myug`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/cwj0myug)
- 242 carriers: [`tm8nyt19`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tm8nyt19)

Both runs used the same 16 selected windows, seed, architecture widths, plain
MSE, learning rate `1e-3`, and 2,000 optimizer steps. The only intended
difference was `data.num_subcarriers` and `model.num_subcarriers`.

**Convergence measurements**

| Step | 30-carrier eval MSE | 242-carrier eval MSE |
|---:|---:|---:|
| 1 | 0.061880 | 0.068409 |
| 100 | 0.003072 | 0.004945 |
| 200 | 0.000728 | 0.002361 |
| 500 | 0.000260 | 0.000287 |
| 1,000 | 0.00006905 | 0.00010577 |
| 1,500 | 0.00005534 | 0.00009248 |
| 2,000 | **0.00003881** | **0.00004194** |

Final diagnostics:

| Metric | 30 carriers | 242 carriers |
|---|---:|---:|
| train-mode MSE | 0.00003371 | 0.00004088 |
| eval-mode MSE | 0.00003881 | 0.00004194 |
| eval off-center MSE | 0.00002805 | 0.00003240 |
| eval background leakage | 0.001417 | 0.001431 |
| eval maximum absolute error | 0.17895 | 0.17296 |
| eval negative fraction | 0.00000643 | 0.00000551 |
| trainable parameters | 2,410,836 | 2,627,924 |
| W&B runtime | 127 s | 131 s |

**Visual observation**

The final fixed examples in both W&B runs reproduce narrow temporal changes,
center-ridge shape, and off-center Doppler lobes across all four antennas.
They no longer resemble the constant vertical mean-spectrum outputs seen in
full-dataset training. Residual error is low-amplitude and spatially diffuse;
the largest errors remain around sharp local peaks.

**Interpretation**

1. The full 2D decoder, optimizer, and plain-MSE path can clearly memorize
   these 16 motion-rich CSI/Doppler pairs. Basic representational capacity is
   not the immediate blocker.
2. Thirty fixed-uniform carriers are sufficient for this memorization test.
   All 242 carriers converged more slowly initially and did not improve the
   final error. This does not show that extra carriers are useless for
   held-out windows or cross-domain generalization.
3. Train-mode and eval-mode errors are close after 2,000 steps. BatchNorm
   running statistics can represent this one fixed recording once converged.
   This does not reject the separate hypothesis that source-derived BatchNorm
   statistics contribute to PI-4a instability.
4. The result weakens insufficient model capacity as the primary explanation
   for the failed full-data runs. It strengthens the case for objective
   mismatch, limited independent recording diversity, domain shift, and
   possibly missing full-recording context.
5. Memorization cannot validate temporal alignment: a network can associate a
   finite input window with an arbitrary target. Alignment still requires the
   independent lag diagnostic.

**Decision**

Do not move the full training pipeline to 242 carriers based on this result.
The next capacity/generalization diagnostic is the same plain-MSE test on 64
motion-stratified windows from four source recordings, followed by held-out
windows from those same recordings.

### Motion-aware objective v1

**Question**

How should the full-data objective emphasize narrow off-center motion without
again rewarding a center-biased mean spectrum or producing broad false power?

**Implementation/config**

- Loss name: `motion_aware_mse`
- Code: [distillation.py](../../src/wifi_doppler/training/distillation.py)
- Training config: [pi_cross_domain_unet1d_spatial_head_motion_aware.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_motion_aware.yaml)
- Architecture and data protocol use the 30-carrier 1D U-Net with spatial 2D
  output head.

For predictions and targets `p,y` with shape
`[batch, antenna, time, Doppler bin] = [B,A,T,100]`, define:

```text
floor f = 0.0630957344
stationary bins C = {45, ..., 55}
off-center bins O = {0, ..., 44} union {56, ..., 99}

q[b,a,t,d] =
    1[d in O] * clamp((y[b,a,t,d] - f) / (1 - f), 0, 1)
```

The loss is:

```text
L = 1.00 * L_full
  + 0.25 * L_motion
  + 0.25 * L_background
  + 0.05 * L_wasserstein
```

with:

```text
L_full = mean over B,A,T,D of (p - y)^2

L_motion =
    sum over B,A,T,D of q * (p - y)^2
    / sum over B,A,T,D of q

L_background =
    mean of relu(p - f) where y is equal to the floor
```

For Wasserstein, active off-center power is:

```text
p_active = relu(p - f) * 1[d in O]
y_active = relu(y - f) * 1[d in O]
```

Each `[b,a,t]` active spectrum is normalized across the last dimension
`D=100`. The first Wasserstein distance is the mean absolute difference
between cumulative distributions along `dim=-1`, divided by `99`. Frame
distances are weighted by `sum_d q[b,a,t,d]`, so frames with no target
off-center power contribute zero.

**Why full-map MSE remains**

- Approximately 77-80% of target pixels equal the hard floor. Full MSE gives
  every pixel a gradient, including negative predictions, missed center
  structure, and values above the target range.
- The previous composite loss applied only `0.05 * raw MSE`; it could report a
  moderate loss while raw source-validation MSE reached `0.2749`.
- The tiny overfit runs show that plain MSE and the full 2D model can reproduce
  the selected maps. MSE is therefore retained as the global anchor, not
  discarded as an unusable pointwise distance.

**Why target-active bin weighting**

- A frame-weighted MSE averaged across all 89 off-center bins would still
  dilute narrow peaks with many floor bins.
- `q` weights each target-active off-center bin directly. Exact-floor bins
  receive zero motion weight, a weak target peak remains nonzero, and a peak
  at `1` receives maximum weight.
- There is no `0.2` training threshold. On sampled source-training frames,
  23.1% had an off-center maximum above `0.2`, but peaks at or below `0.2`
  still represented 10.3% of total continuous motion weight. A hard gate
  could remove subtle motion without activity labels proving it was noise.
- The `0.2` statistic may remain a diagnostic, but it does not alter
  gradients.

**Why coefficient 0.25 is motion-dominant**

The coefficient cannot be interpreted without the term's reduction. On the
constant train-mean predictor evaluated against sampled PI-4a target frames:

| Component | Raw value | Weighted contribution |
|---|---:|---:|
| full-map MSE | 0.00559 | 0.00559 |
| target-active off-center MSE | 0.09233 | 0.02308 |
| background leakage | 0.01373 | 0.00343 |
| motion-only Wasserstein | 0.09275 | 0.00464 |

The weighted motion MSE is already about four times the full-map MSE. Relative
to full MSE alone, the total gradient at an off-center target value of `0.2`,
`0.5`, and `1.0` is approximately `3.7`, `9.6`, and `19.5` times larger.
Using coefficient `1.0` would raise the strongest peak gradient to roughly 75
times the full-MSE gradient and risk fitting source-specific target artifacts.

**Why L1 background leakage**

- The mask includes every location where the stored target equals the hard
  floor, across both center and off-center bins.
- `relu(p-f)` penalizes only predicted power above the floor. Full MSE still
  pushes predictions below the floor back toward the correct floor value.
- An L1 excess is intentional: another squared term would make broad,
  low-amplitude haze cheap. The constant L1 gradient directly suppresses the
  visual leakage observed in prior runs.

**Why motion-only Wasserstein**

- Wasserstein compares normalized off-center spectral shape and penalizes
  moving power farther in Doppler more than moving it to an adjacent bin.
- Center bins are zeroed but the distributions retain all 100 coordinates.
  Concatenating left and right off-center bands would incorrectly make bins 44
  and 56 adjacent and erase the excluded center-band distance.
- Normalization removes amplitude, so Wasserstein cannot replace motion MSE.
  Its coefficient remains `0.05`; amplitude and false mass are controlled by
  the other terms.

**Deliberate non-changes**

- Predictions remain unconstrained. A scaled sigmoid would add saturation and
  confound the loss ablation with a model-head change. Full MSE penalizes
  values outside `[f,1]`, and negative predictions remain logged.
- SmoothL1 is not used. Its linear high-error region weakened the incentive to
  correct large missed peaks in the previous objective.
- BatchNorm, architecture, selected carriers, batching, splits, and
  normalization are unchanged so this run isolates the objective.

**W&B logging**

The trainer records total `loss`, the existing full-map `mse`, and:

```text
loss_full_map_mse
loss_motion_mse
loss_background_leakage
loss_motion_wasserstein
motion_mse
background_leakage
motion_wasserstein
```

The four `loss_*` values are weighted contributions and sum to `loss`. The
three unprefixed auxiliary values expose their raw scales. They are logged for
train batches and for train/source-validation/target-validation epochs.

**Training command**

```bash
python scripts/train_csi_to_doppler.py \
    --config configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_motion_aware.yaml
```

Training began in W&B run
[`tio9b4ad`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tio9b4ad).
Preliminary measurements follow.

### Motion-aware run: preliminary utilization diagnosis

**Run**

[`tio9b4ad`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tio9b4ad),
running on an NVIDIA GeForce RTX 2070 with 8 GB VRAM.

**Measurements through epoch 4**

| Epoch | Train seconds | Train samples/s | Source-val seconds | Target-val seconds | Peak allocated VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 78.06 | 343 | 12.81 | 4.83 | 6.51 GB |
| 2 | 61.56 | 435 | 11.69 | 4.27 | 4.09 GB |
| 3 | 61.26 | 437 | 11.59 | 4.06 | 4.19 GB |
| 4 | 61.69 | 434 | 11.60 | 4.10 | 4.00 GB |

The earlier spatial-head run
[`gzfqzvqv`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/gzfqzvqv)
used the same architecture and batching with the legacy composite objective.
It trained at 467-537 samples/s in epochs 2-5. The new objective is therefore
approximately 7-19% slower after warm-up, not an order-of-magnitude
regression. Validation is roughly twice as slow because the motion-aware loss
adds target weighting, background masking, and two full 100-bin cumulative
distributions.

**GPU event-stream observation**

W&B samples the RTX 2070 at approximately 15-second intervals. Samples
alternate between active GPU clocks/utilization and zero utilization. The
zeroes are real, but the sparse chart mixes several different phases:

- approximately 62 seconds of training per epoch;
- approximately 16 seconds of source/target validation;
- CPU-side heatmap rendering, checkpoint serialization, and W&B artifact
  handling between epochs;
- intermittent host batch preparation and memmap page faults.

Logged groups of 20 training steps alternate between approximately 6.4
seconds and 12-15 seconds. The slower groups are consistent with transitions
between the four pools of eight recordings, where new memmaps begin serving
randomly shuffled windows. Steady groups correspond to approximately 800
samples/s, showing that the GPU is fed substantially faster once a pool is
warm.

Host memory was already 80-83% utilized, with only 2.6-3.2 GB available and
the training process using 3.6-4.2 GB RSS. A large pinned-memory prefetch queue
would risk paging. Doubling batch size is also unsafe because first-epoch peak
allocation already reached 6.51 GB on an 8 GB GPU.

**Decision**

- Do not alter the active run or its immutable config mid-experiment.
- Do not increase batch size from 256.
- The current `prefetch_batches: 2` remains conservative for this run.
- If input utilization remains a priority after evaluating model quality,
  benchmark only `prefetch_batches: 2` versus `4` on one epoch. This runtime
  setting can be changed on resume, but expected gains are limited and must be
  checked against host-memory pressure.
- Treat low utilization during validation, plotting, and artifact upload as
  expected epoch-end overhead rather than model-training starvation.

### Smooth center taper experiment

**Question**

The first motion-aware loss defines bins 45-55 as a rectangular stationary
region. Real no-motion Doppler ridges have tapered skirts rather than sharp
edges. The hard mask therefore gives bin 55 zero motion gradient from the
motion-specific terms and bin 56 full motion gradient, and can discard genuine
slow motion close to the center.

**Implementation/config**

- Code: [distillation.py](../../src/wifi_doppler/training/distillation.py)
- Config:
  [pi_cross_domain_unet1d_spatial_head_motion_mse_1_smooth_center.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_motion_mse_1_smooth_center.yaml)
- Input and output remain `[B,4,30,T_raw,2]` and `[B,4,340,100]`.
- Motion MSE weight is `1.0`; all other loss coefficients and training settings
  match the 30-carrier motion-aware experiment.

For Doppler-bin index `d` along `dim=-1` and center bin `c=50`, the smooth
motion weight is:

```text
w[d] = 1 - exp(-0.5 * ((d - c) / sigma)^2)
sigma = 5 bins
```

This gives weight `0` at bin 50, approximately `0.02` at +/-1 bin, `0.39` at
+/-5 bins, `0.86` at +/-10 bins, and approaches `1` farther away. Target
activity becomes:

```text
q[b,a,t,d] =
    w[d] * clamp((y[b,a,t,d] - floor) / (1 - floor), 0, 1)
```

The same `w[d]` multiplies prediction and target active power before
normalization and `cumsum(dim=-1)` in the Wasserstein term. Full-map MSE and
background leakage remain unmasked. Thus near-center errors still receive
their ordinary full-map gradient, while motion-specific emphasis increases
continuously with Doppler distance.

**Why this version**

- It removes the arbitrary 0-to-1 jump at bins 55/56.
- It can retain weak, slow motion near the stationary ridge without assigning
  the high-energy center bin full motion weight.
- A Gaussian-complement taper is one parameter and is easy to interpret.
- It does not claim to model the exact stationary spectrum, which may vary by
  antenna, recording, and environment. A learned or target-adaptive stationary
  profile should wait until reliable no-motion labels or calibration windows
  are available.
- Existing configs retain the rectangular mask so completed runs remain
  reproducible.

No W&B run is linked yet.

### Shared single-antenna architecture

**Question**

The existing model flattens all four antennas into the Conv1d channel axis and
predicts all four Doppler maps jointly. Its first projection can therefore
learn antenna-position-specific weights and every later feature can mix
antennas. This permits cross-antenna shortcuts that may fit the source
recordings but do not enforce the same raw-CSI-to-Doppler mapping for every
receiver stream.

**Implementation/config**

- Architecture: `unet1d_spatial_head_shared_antenna`
- Code:
  [csi_to_doppler_shared_antenna.py](../../src/wifi_doppler/models/csi_to_doppler_shared_antenna.py)
- Config:
  [pi_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware.yaml](../../configs/csi_to_doppler/pi_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware.yaml)
- Trainable parameters: `3,047,265`, compared with `3,070,524` for the joint
  spatial-head model.

The data loader remains unchanged and emits:

```text
x: [B, 4, 30, T_raw, 2]
y: [B, 4, 340, 100]
```

Inside the model only, input is reshaped to:

```text
[B * 4, 1, 30, T_raw, 2]
```

One `CsiToDopplerUNet1DSpatialHead` configured with `num_antennas=1` processes
all antenna examples with shared parameters. Predictions are reshaped back to
`[B,4,340,100]` before loss calculation, metrics, heatmaps, and checkpointing.
The same model can accept `[B,1,30,T_raw,2]` for single-antenna inference.

**Batching decision**

- The configured window batch is reduced from `256` to `64`.
- Flattening four antennas gives an effective network batch of `256`
  single-antenna examples.
- Each full batch still draws windows from eight recordings and contains every
  physical antenna equally. BatchNorm therefore sees mixed recordings and
  balanced antennas rather than four separate antenna-specific batches.
- Every antenna-window pair is used exactly once per epoch. No random or
  rotating antenna selection is added.
- W&B throughput remains reported in recording windows per second, not
  flattened antenna examples per second.

The temporal encoder now runs independently four times per recording window,
so its epoch compute is expected to approach four times the joint model's
temporal-core compute. The first projection and output work do not increase by
the same factor because their antenna/channel dimensions are correspondingly
smaller. Actual runtime and memory must be measured on the RTX 2070.

**Controlled comparison**

The first run intentionally reuses the completed `tio9b4ad` protocol:

```text
30 fixed-uniform subcarriers
rectangular center mask, bins 45-55
motion MSE weight = 0.25
background leakage weight = 0.25
motion Wasserstein weight = 0.05
```

The unrun `motion_mse_weight=1.0` and smooth-center experiments are excluded,
so any difference from `tio9b4ad` can be attributed primarily to antenna
weight sharing and independent processing. No W&B run is linked yet.

### Fixed training-reference images

Training previously logged scalar metrics but retained heatmaps only for
source and target validation. Capturing examples from the shuffled
gradient-enabled training pass would change filenames every epoch and would
show predictions made with training-mode BatchNorm, so those images would not
support a stable longitudinal comparison.

The trainer now selects `training.validation_examples` fixed windows from the
training split once using the run seed and mixed-recording iterator. After
each epoch it evaluates only that small batch in inference mode and logs the
same target, prediction, and absolute-error heatmaps under:

```text
train_reference/examples
```

The reference pass does not update weights, alter training metrics, or traverse
the full training set. With the default two examples, its cost is one
two-window inference pass per epoch. Together with source and target
validation images, this separates failure modes:

- poor training references indicate optimization, loss, or model limitations;
- good training references but poor source validation indicate source
  generalization or overfitting;
- good source validation but poor target validation indicates domain shift.

### Combined AR and PI scale experiment

**Dataset decision**

Use every AR Doppler scenario with a confirmed direct raw-directory mapping:

```text
S1a S1b S1c S2a S3a S4a S5a S6a S7a
```

These map to `AR-1a`, `AR-1b`, `AR-1c`, `AR-2a`, `AR-3a`, `AR-4a`,
`AR-5a`, `AR-6a`, and `AR-7a`. `S2b`, `S4b`, and `S6b` are excluded because
the current mapping would require raw directories `AR-2b`, `AR-4b`, and
`AR-6b`, which are absent from the available raw dataset. Extra raw AR
directories without Doppler counterparts are also excluded.

The protocol is:

```text
train:      paired AR domains + PI-1a/2a/3a, 0-60%
source val: paired AR domains + PI-1a/2a/3a, 60-80%
target val: PI-4a, 60-80%
target test: PI-4a, 80-100%
```

PI-4a remains completely excluded from optimization. The source-validation
metric is an aggregate over AR and PI source domains; separate domain metrics
can be added later if this aggregate hides a meaningful discrepancy.

**Prepared storage**

The existing memmap contains PI recordings. Rebuilding those files is
unnecessary. The converter now supports explicit manifest-preserving append:

```bash
python scripts/convert_csi_doppler_memmap.py \
    --raw-root data/CSI-80Mhz \
    --doppler-root data/doppler_traces \
    --output-root data/csi_doppler_memmap \
    --scenarios S1a S1b S1c S2a S3a S4a S5a S6a S7a \
    --append
```

Converted AR arrays are added under their canonical `S*` scenario names. The
manifest retains all PI entries and records both source roots. Conversion is
resumable and the manifest is replaced atomically only after all requested
recordings finish.

**Training config**

[pi_ar_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware_full_subcarriers.yaml](../../configs/csi_to_doppler/pi_ar_cross_domain_unet1d_spatial_head_shared_antenna_motion_aware_full_subcarriers.yaml)
uses:

```text
shared single-antenna 1D U-Net with spatial head
all 242 cleaned subcarriers
window batch 64, effective antenna batch 256
motion-aware objective with hard center mask and motion weight 0.25
```

This is a scale experiment rather than a one-variable ablation because both
the source-domain set and carrier count exceed `tio9b4ad`. No W&B run is
linked yet.

### Downloaded AR target alignment failure

**Observed failure**

The first combined AR+PI training attempt failed while stacking a
mixed-recording batch. The model was not reached: at least one raw CSI slice
was shorter than the expected `[4,242,370,2]` input window.

Manifest inspection compared cleaned raw packet count `N_raw` with downloaded
Doppler frame count `N_doppler`. For the standard SHARP geometry
`start=800`, `end=800`, `sample_length=31`, and `sliding=1`, an aligned pair
should satisfy:

```text
N_raw - N_doppler = 1631
```

An observed value of `1632` is a harmless one-frame implementation difference.
`S1a`, `S1b`, `S1c`, and `S7a` consistently satisfy `1631/1632`. Other
downloaded domains do not:

```text
S2a_E: raw 21626, Doppler 50227, delta -28601
S2a_L: raw 21025, Doppler 51052, delta -30027
S3a-S5a: heterogeneous deltas below and above 1631
S6a: deltas around 6300-6700
```

`S2a_E/L` cannot originate from their paired MAT files under any valid crop
because the target is much longer than the available raw stream. For the other
inconsistent domains, length differences cannot recover alignment: they reveal
only total trimming, not whether packets were removed from the beginning or
end. Padding, truncating, or inferring an offset would create unverified
training pairs.

**Why the SHARP reproduction still worked**

`notebooks/sharp_reproduction.ipynb` uses `DopplerWindowDataset`, which reads
only fixed-size windows from precomputed Doppler traces. It never pairs those
windows with raw CSI. Its default invocation also uses only `S1a/S1b/S1c` and
activities `E/L/W/R/J`; it is neither a full-AR run nor a raw-to-Doppler
alignment test.

**Decision**

- Do not use downloaded `S*` traces for raw-to-Doppler distillation.
- Recompute canonical targets from every available AR raw MAT recording.
- Generate PC targets in the same pass because distillation does not require
  activity labels.
- Keep the already aligned PI targets unless a single fully regenerated
  dataset version is later required.
- Preprocess PC now but initially hold it out from optimization as an
  additional unseen-domain evaluation set.
- Retain the invalid combined config only as historical provenance and do not
  associate it with a W&B run.

**Implementation**

The generic generator is
[`preprocess_sharp.py`](../../src/preprocessing/preprocess_sharp.py). The old
`preprocess_sharp_pi.py` path is now a compatibility entry point. The SHARP
objective, reconstruction equations, and Doppler geometry are preserved. Generic
discovery accepts exact subsets and the `AR`, `PC`, `PI`, and `all` family
selectors. Generated files preserve canonical source stems, for example:

```text
doppler_traces_recomputed/AR-1a/AR1a_W_stream_0.txt
doppler_traces_recomputed/PC-1a/PC1a_W_stream_0.txt
```

The pairing layer recognizes these canonical filenames directly, avoiding the
downloaded archive's `S* -> AR-*` alias. `manifest.json` records the source
path and shape, cleaned raw shape, aligned raw start, output stream shapes,
preprocessing parameters, repository revision/dirty state, and script checksum.
The generator validates the expected target frame count before accepting each
recording.

**Local smoke and scale estimate**

A bounded end-to-end smoke used one real AR MAT file and one real PC MAT file,
with unique temporary stems and 64 H-estimation packets per antenna. It
successfully produced four `[33,100]` streams per recording and a valid
manifest. With two worker processes, H estimation completed 512
antenna-packets at approximately 44 packets/s.

Header inspection of the complete local raw collection found:

```text
AR recordings:               164
PC recordings:                40
total recordings:            204
H-estimation antenna-packets: approximately 20.7 million
```

At the two-worker smoke rate, a local run would take roughly 5.5 days.
Ideal linear scaling to 100 workers would be about 2.6 hours, but real runtime
will be higher due to process startup, MAT/pickle I/O, memory bandwidth, and
filesystem contention. Each worker loads a full normalized signal recording
and retains complex H-estimation arrays, so the production job must benchmark
memory and throughput before requesting hundreds of workers. BLAS/OpenMP
thread counts should remain one to prevent nested oversubscription.

The uncompressed signal, H-estimation, and reconstructed-phase intermediates
may require several hundred GB for all AR and PC recordings. Run them on
cluster scratch rather than a small home quota and retain the checkpoint files
until Doppler generation has completed.

No W&B run is linked because target generation and the replacement training
configuration have not completed.

### SHARP target-generation performance audit

Profiling showed that H estimation, not Doppler FFT computation, dominates
target generation. The original inner loop rebuilt Fourier dictionaries,
sparse constraint matrices, and two OSQP workspaces for every antenna-packet.
A 32-packet controlled profile spent `0.404/0.930 s` constructing Fourier
dictionaries and `0.452/0.930 s` in the LASSO wrapper, including 64 OSQP
setups.

The generic generator now:

- builds Fourier dictionaries with vectorized NumPy operations;
- prepares the fixed coarse OSQP problem once per worker;
- caches refined OSQP problems by coarse delay bin;
- resets primal and dual state for each packet while reusing matrix
  factorizations;
- orders H tasks by stream so a large worker pool does not immediately load
  four copies of each recording;
- no longer unpickles large checkpoint files repeatedly for progress display;
- defaults to stream-boundary resume instead of rewriting approximately
  130 MB of partial arrays every 500 packets;
- accepts an explicit `--phase-root`, allowing large intermediates to live
  under cluster scratch instead of the repository;
- preprocesses raw signal recordings with at most eight worker processes;
- distributes reconstruction and Doppler post-processing by recording;
- batches Doppler FFTs in cache-sized groups and vectorizes phase detrending.

The optimized 32-packet profile took `0.337 s`, approximately `2.8x` faster.
The Fourier dictionaries are bit-identical to the original implementation.
Cached OSQP coefficients agreed within its numerical stopping tolerance
(maximum observed absolute difference `2.2e-4`). Doppler output agreed to
machine precision (`1.7e-15` maximum absolute difference), and vectorized
detrending agreed within `2.8e-15`.

A bounded real `AR1a_C` CLI smoke with four workers processed 256
antenna-packets in approximately `1.9 s` for H estimation, completed
post-processing, and resumed in under one second. This short measurement is
startup-dominated and is not a full-dataset runtime prediction.

The multiprocessing implementation spans CPUs on one node only. On the DEI
cluster it should be submitted as one Slurm task with multiple CPUs per task,
and `--jobs` should equal `SLURM_CPUS_PER_TASK`. A multi-node run must partition
subsets into independent output roots to avoid concurrent manifest writes.

### DEI cluster deployment

The target generator is CPU-only and does not benefit from requesting a GPU.
The deployment consists of:

- [`cluster/wifi-doppler-preprocess.def`](../../cluster/wifi-doppler-preprocess.def),
  a minimal Python 3.11 Singularity image containing only NumPy, SciPy, OSQP,
  tqdm, and the two required source files;
- [`cluster/preprocess_ar_pc.slurm`](../../cluster/preprocess_ar_pc.slurm), a
  single-node multiprocessing job;
- [`cluster/README.md`](../../cluster/README.md), containing transfer, image
  build, smoke, production, and monitoring commands.

The raw dataset location `~/CSI-80Mhz` is valid because DEI automatically
mounts home inside Singularity. It should remain persistent input rather than
being copied to node scratch before allocation. Its 248 MAT files comprise 204
AR+PC recordings and 44 PI recordings. Home quota must still be checked before
writing the approximately 15-20 GiB of final AR+PC Doppler traces there; a group
NAS path should be used when quota is insufficient.

Large phase, H-estimation, and reconstructed-CSI intermediates are directed to
`/ext/$USER/wifi-doppler-har/$SLURM_JOB_ID`. The job requires at least 300 GiB
free scratch, preserves scratch on failure, validates the persistent manifest
on success, and then removes its job-specific scratch directory. This follows
DEI's requirement that `/ext` be used only for temporary node data.

The production request defaults to one task with 48 CPUs, 96 GiB RAM, and 12
hours in `allgroups`. Forty-eight CPUs can be scheduled on more DEI nodes than
a 96-CPU request and should therefore have better availability. The code has
816 independent H-estimation streams, so a 96-CPU override remains available
after measuring the bounded smoke job with `seff` and `myjobinfo`.

The unused SHARP `r_vector` artifact was removed before deployment. Only
`Tr_vector` is consumed by reconstruction; dropping `r_vector` saves roughly
40 GiB across AR+PC and reduces each worker's resident memory. Packet-level
checkpointing remains disabled because completed `Tr_vector` streams already
provide the useful resume boundary without repeated large-array writes.

The image cannot be built in the current local environment because neither
Singularity nor Apptainer is installed. DEI documents building images inside a
short `sinteractive` allocation; the supplied definition includes `%test`
checks that run during build and can be repeated with `apptainer test`.

### 2026-07-27: combined-data run and GPU starvation

**Run**

[W&B `ehotc8fp`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/ehotc8fp)
is the first canonical combined-data run. It uses the shared single-antenna
1D U-Net with spatial head, all 242 subcarriers, and the motion-aware loss.
The training split contains 233 AR/PC/PI-1a/2a/3a recordings and 121,629
windows. PI-4a remains target-only.

Epoch 1 measured:

```text
train:       1,644.4 s, 74.0 windows/s
source val:    336.8 s, 114.7 windows/s
target val:     23.7 s, 119.5 windows/s
target-val loss: 0.023761
target-val MSE:  0.006431
```

W&B sampled the GPU at 15-second intervals. During the first approximately
27 minutes, GPU utilization ranged from 0% to 85% and averaged about 32%.
Allocated GPU memory peaked at approximately 4.7 GB, while system RAM reached
97%. The process used roughly one CPU core. Batch-log intervals normally
reached 165-210 windows/s but periodically fell below 35 windows/s, especially
around recording-pool transitions. This identifies host data preparation and
random memmap access as the limiting path; the 3.1-million-parameter model is
not saturating the GPU.

**Root causes**

- A full batch contains roughly 210 MiB of CSI and targets before transfer.
- One producer thread performed all slicing, real/imaginary conversion,
  finite scans, stacking, and pinning.
- Each sample was allocated once and then copied again by the final batch
  stack.
- Every heavily overlapping window was scanned for finite values, repeating
  most reads about twelve times at stride 30 and raw window length 370.
- Fully random window order turned memmapped recordings into random reads and
  caused page-cache stalls under high RAM pressure.
- Validation opened one recording per pool, increasing mapping and partial
  batch boundaries.

**Optimization prepared after `ehotc8fp`**

- Fill one preallocated float32 batch directly from complex memmaps.
- Remove redundant per-window finite scans; non-finite model/loss output still
  fails immediately.
- Use eight-recording pools for deterministic validation as well as training.
- Add `train/source_val/target_val data_wait_seconds` and
  `data_wait_fraction` metrics.
- Add explicit `training.window_shuffle_chunk_size: 8` to the combined-data
  config. Chunks are shuffled, but windows remain chronological within each
  chunk. With batch 64 and eight recordings, a full batch still contains about
  eight windows from every recording while reading approximately 580 adjacent
  raw frames per recording instead of as many as 2,960 random frames.

Chunked shuffling changes optimization order, though not epoch coverage, and
therefore requires a distinct W&B run rather than being interpreted as a
continuation of `ehotc8fp`. Batch size was not increased: system RAM, not GPU
capacity, was already the tighter resource. The next run should compare epoch
throughput, GPU utilization, and `data_wait_fraction` before considering
multiple producer workers or a larger batch.

**First optimized run**

[W&B `6s1lm3dy`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/6s1lm3dy)
tested preallocated batches, chunk-local shuffling, and mixed validation pools.
It did not improve the first comparable 1,230 seconds of training:

```text
                         ehotc8fp    6s1lm3dy
mean GPU utilization       34.0%       30.1%
step 1480 runtime        1,280 s     1,358 s
mean process RSS          2.26 GB     3.55 GB
disk bytes read            82 GB       95 GB
```

The run still alternated between fast 200-250 windows/s intervals and
20-40-second stalls. Inspection found that the full `0..241` subcarrier view
was still supplied to NumPy as an integer array. This invokes advanced
indexing and creates an additional complex array for every window before the
preallocated batch is filled. On a local prepared PI memmap, filling 100
full-carrier windows took `0.344 s` with advanced indexing and `0.030 s` with
the equivalent basic slice, approximately `11x` faster for that operation.

The iterator now detects contiguous subcarrier views once and uses a basic
slice. Sparse 30-carrier views retain indexed selection. This change preserves
sample values and order and passed the full 22-test distillation suite. It must
be measured in a new run; adding producer workers before removing this
unnecessary copy is not justified.

### 2026-07-27: Smaller shared-antenna capacity ablation prepared

Config:
[`ar_pc_pi_cross_domain_unet1d_spatial_head_shared_antenna_small_motion_aware_full_subcarriers.yaml`](../../configs/csi_to_doppler/ar_pc_pi_cross_domain_unet1d_spatial_head_shared_antenna_small_motion_aware_full_subcarriers.yaml)

The temporal channel widths are halved from `128/192/256` to `64/96/128`,
and the spatial-head width is halved from `8` to `4`. All data, split, loss,
optimizer, batch, and training settings remain identical to the full-width
AR+PC+PI configuration. Parameter count falls from `3,101,537` to `792,609`
(`25.6%`), while receptive field and output geometry remain unchanged.

This tests whether the full-width model's capacity contributes to a persistent
train/validation gap. A smaller model is expected to train faster and may
generalize better if that gap is genuine variance. It will not fix domain
shift, target noise, missing input information, or a loss whose easiest
solution is an averaged central spectrum. The run must therefore be judged by
both train-versus-validation behavior and motion-peak heatmaps. No W&B run is
linked yet because this entry records the experiment before execution.

**Interim run diagnostics**

[W&B `vbrol6v2`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/vbrol6v2)
uses this 792,609-parameter model with window batch size 160, equivalent to
640 single-antenna examples per model forward. It includes the contiguous
subcarrier slice fix at commit `24f264e`.

Before the first validation completed, W&B measured peak allocated GPU memory
at 7.04 GB on the 8 GiB RTX 2070. Mean GPU compute utilization was 26.7%, with
28.6% of samples below 1% and bursts reaching 98%. Mean GPU memory-controller
utilization was only 8.9%, and mean power was 45% of the limit. A direct live
sample found the GPU at 52 degrees C and 56 W of its 185 W limit. The GPU is
therefore neither too small-batched, thermally throttled, nor continuously
memory-bandwidth limited; it alternates between computing and waiting.

The host has six CPU cores and 16 GB RAM. W&B measured mean system RAM at
86.8%, a 95% peak, and mean process RSS at 4.77 GB. Its system disk-read
counter increased by approximately 131 GB during the roughly 1,453-second
training phase. The 121,629-window phase therefore achieved approximately
83.7 windows/s despite the model being one quarter the baseline size.
Repeated materialization of heavily overlapping memmap windows by one producer
thread remains the dominant bottleneck.

Batch size should not be increased further: 160 already approaches the GPU
memory limit, and a linearly scaled batch of 192 would exceed it while adding
CPU and pinned-memory pressure. The next performance work should target batch
production and overlapping-window I/O. Capacity conclusions must wait for the
completed losses and heatmaps.

**Grouped-slab batch preparation**

The iterator now groups each batch's windows by recording and subcarrier view,
then partitions them into overlapping temporal runs. Each run's raw CSI and
Doppler target slab is copied from its memmap once; individual windows are
filled from that contiguous in-memory slab. This preserves exact batch
membership, order, filenames, values, mixed-recording balance, and epoch
coverage while avoiding one backing-array read per overlapping window.

Batch groups can also be filled concurrently through
`training.batch_preparation_workers`. On the local PI full-subcarrier memmaps,
a cache-warm 1,600-window microbenchmark measured approximately 205 windows/s
with one worker and 420 windows/s with two. Four workers showed no repeatable
gain over two, so the small-model AR+PC+PI config uses two workers for the
six-core RTX 2070 host. These are iterator-only measurements, not expected
end-to-end training rates; the next W&B run must verify GPU utilization,
`data_wait_fraction`, physical disk reads, RAM pressure, and epoch duration.

The parallel path writes disjoint sample positions in preallocated arrays and
uses only small per-recording slabs, rather than preparing multiple complete
545 MB batches concurrently. Twenty-three distillation tests pass, including
exact tensors for shuffled mixed-recording batches, deterministic ordering,
single-slab read counts, config validation, checkpoint resume, and the CPU CLI
smoke test.

### 2026-07-27: RTX 4090 first-epoch profile and launch synchronization

[W&B `kbjjrx1c`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/kbjjrx1c)
uses the 792,609-parameter small shared-antenna model on an RTX 4090 with all
242 subcarriers, window batch size 448, eight recordings per pool, four batch
preparation workers, and two prefetched host batches. Epoch 1 contained
121,629 windows in 291 batches and took 557.81 seconds, or 218.05 windows/s.
Peak allocated GPU memory was 12.30 GB. The full epoch, including source and
target validation, took approximately 766 seconds.

Measured training data wait was only 7.18 seconds, or 1.29% of training time.
This rejects the earlier hypothesis that the optimized iterator is still the
main cause of the 4090's low sampled utilization. The training loop instead
forced a device synchronization every batch through both
`torch.isfinite(loss)` in Python control flow and `loss.item()` for objective
accumulation. Those barriers prevented CUDA work from being queued across
steps and amplified Python and kernel-launch gaps for the small model.

The loop now accumulates detached scalar objectives on the GPU, checks
finiteness at the existing periodic logging boundary and epoch end, and
synchronizes once before final timing. CUDA training also uses fused Adam.
Exact epoch metrics, periodic failure detection, validation behavior, and
checkpoint state are retained. The distillation suite passes all 24 tests.
This implementation change must be benchmarked in a new run rather than
attributed retroactively to `kbjjrx1c`.

For the 12-core, 64 GB Vast instance, the aggressive next-run settings are a
window batch of 768, 12 recordings per pool, 11 preparation workers, and eight
prefetched batches. The observed 12.30 GB allocation at batch 448 projects to
approximately 21.1 GB at batch 768, leaving a narrow but plausible margin on a
24 GB card. Each queued full-subcarrier batch is approximately 2.5 GiB, so an
eight-batch queue consumes about 20 GiB of host memory and leaves the remainder
for Python, active recording slabs, and filesystem page cache. This is an
intentional throughput stress configuration; an out-of-memory result should
fall back to batch 704 without changing the model or loss.

### 2026-07-27: full-data run `3wrefrft` and end-to-end failure audit

**Run and exact protocol**

[W&B `3wrefrft`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/3wrefrft)
used
[`ar_pc_pi_cross_domain_unet1d_spatial_head_shared_antenna_small_motion_aware_full_subcarriers.yaml`](../../configs/csi_to_doppler/ar_pc_pi_cross_domain_unet1d_spatial_head_shared_antenna_small_motion_aware_full_subcarriers.yaml).
The run finished at epoch 27. It used:

- 233 training recordings from AR, PC, and PI-1a/2a/3a;
- 121,629 overlapping training windows and all 242 usable subcarriers;
- PI-4a only for target validation;
- the 792,609-parameter shared single-antenna temporal U-Net;
- a 25-bin, four-channel spatial output head;
- window batch 768, equivalent to 3,072 independent antenna examples per
  network forward;
- 12 recordings per pool, BatchNorm, Adam at constant `1e-3`, AMP, and the
  motion-aware objective.

The W&B pairing reports contain 233 source pairs and 10 PI-4a pairs, with no
unmatched raw files or Doppler keys. This rejects a missing-file or accidental
pairing explanation for this run.

**Curve measurements**

| Metric | Epoch 1 | Epoch 16 | Epoch 26 best | Epoch 27 |
|---|---:|---:|---:|---:|
| train loss | 0.047992 | 0.028208 | 0.026286 | 0.026379 |
| train MSE | 0.014760 | 0.007978 | 0.007508 | 0.007562 |
| train motion MSE | 0.082467 | 0.058114 | 0.053626 | 0.053983 |
| source-val loss | 0.043444 | 0.029590 | 0.029270 | 0.033952 |
| target-val loss | 0.034088 | 0.019177 | **0.018227** | 0.020391 |
| target-val MSE | 0.006533 | 0.004330 | 0.004524 | 0.005432 |
| target peak-bin MAE | 0.374 | 0.371 | 0.369 | 0.370 |

The training objective did not become perfectly flat at epoch 2; it continued
to decline slowly. The relevant failure is conditional underfitting: the
network improves the marginal center/background spectrum while weakly fitting
the timing, amplitude, and Doppler extent of motion.

At the best epoch-26 checkpoint, the target objective decomposes as:

| Weighted component | Value | Fraction of total |
|---|---:|---:|
| full-map MSE | 0.004524 | 24.8% |
| motion MSE | 0.009027 | 49.5% |
| background leakage | 0.001723 | 9.5% |
| motion Wasserstein | 0.002954 | 16.2% |

Motion MSE is already the largest term. The missing peaks are therefore not
explained by coefficient `0.25` making motion numerically irrelevant.

Target peak-bin MAE is almost constant because the stationary center bin is
normally the global maximum. It does not measure whether off-center motion is
present.

**Heatmap progression**

All logged fixed examples were inspected from epochs 1 through 27, not only
the two final screenshots.

- The prediction becomes a stable center ridge during the first epochs.
- The fixed AR training example `AR-9b_J1_d5190-5530_view0` never recovers the
  narrow off-center events, including the clear antenna-2 events.
- The second fixed training example is nearly stationary and is reproduced
  much more closely. This makes the image set itself imbalanced toward the
  easy marginal solution.
- Source-validation examples eventually show broad responses at approximately
  some correct event times. Their width and amplitude remain wrong.
- PI-4a predictions contain faint responses at some target event times, but
  strong target spikes are usually attenuated below their correct amplitude
  and spread into haze.
- This behavior appears early and persists; it is not a late-epoch collapse.

The images therefore refine the phrase "constant prediction." The model is not
strictly input-independent. It learns a weak motion detector on some windows,
then expresses it through an over-smoothed, poorly calibrated spectrum.

**Matched baselines and support diagnostics**

An exact window-weighted shared-antenna PI-1a/2a/3a mean spectrum was evaluated
on PI-4a validation. With globally aggregated reductions it obtains:

```text
loss=0.036725
MSE=0.005682
motion MSE=0.093611
background leakage=0.014111
Wasserstein=0.082260
```

Thus `3wrefrft` beats a literal constant spectrum on aggregate. The remaining
problem is not equivalent to producing the exact training mean.

For a deterministic sample of 100 PI-4a validation windows, three predictors
were compared with the same batches and thresholds:

| Predictor | Loss | MSE | Active-frame recall at 0.2 | Precision | Motion-energy correlation | Predicted/target motion mass |
|---|---:|---:|---:|---:|---:|---:|
| PI train mean | 0.02697 | 0.00535 | 0.0% | 0.0% | undefined | 0.97 |
| affine phase correction + fixed STFT | 0.02210 | 0.00496 | 35.0% | 88.4% | 0.769 | 0.62 |
| epoch-26 model | **0.01872** | **0.00433** | **61.2%** | 46.2% | 0.768 | 1.37 |

An active antenna-frame has a target maximum above `0.2` outside bins 45-55.
The model detects more events than the deterministic baseline, but creates
many more false active frames and excess diffuse mass. This is consistent
with the heatmaps: weak event timing is present, while support, sharpness, and
amplitude calibration are poor.

On the exact displayed PI window, the final model detects only 3 of 46 active
antenna-frames at the same threshold. Performance is therefore strongly
recording- and antenna-dependent, which aggregate loss hides.

**What the raw CSI contains**

The stored input is not simply a noisy version of the target image. SHARP's
teacher performs the following nontrivial deterministic map separately for
each antenna:

1. Normalize each raw packet over subcarrier magnitude.
2. Solve a complex LASSO over a delay dictionary using approximately every
   second usable subcarrier.
3. Select the strongest reconstructed path and use its conjugate as a packet
   phase reference.
4. Reconstruct the CFR, unwrap phase, and remove packet-dependent affine phase
   error across subcarriers.
5. For every target frame, apply a Hann window to 31 sanitized packets, take a
   100-point FFT over time, square magnitude, and sum over subcarriers.
6. FFT-shift, normalize each 100-bin frame by its own maximum, and hard-floor
   values below `10^-1.2`.

The LASSO, strongest-path `argmax`, sequential phase unwrapping, framewise max,
and hard floor make the teacher nonlinear and partly discontinuous.

On the displayed PI window, applying the 31-sample STFT directly to normalized
raw complex CSI gives MSE `0.1731` and visually noise-like Doppler power across
the spectrum. Amplitude-only and adjacent-subcarrier conjugate STFTs collapse
to the center ridge and miss motion.

A much simpler deterministic transformation is unexpectedly informative:
unwrap raw phase across subcarriers, fit and remove each packet's affine phase
difference relative to the first packet in the 370-sample window, and then
apply SHARP's fixed STFT/normalization. On the displayed window this obtains
MSE `0.001316` and motion-energy correlations of `0.849`, `0.933`, `0.356`,
and `0.359` over the four antennas. It visibly recovers many target lobes.

This one-window result is not a replacement for the full SHARP algorithm, but
the 100-window measurements above confirm that it is a meaningful baseline.
It also shows that explicit phase invariance is substantially easier to learn
around than asking a generic U-Net to discover the full radio model.

**Target distribution and domain mixture**

All 972 available Doppler stream files were scanned, covering approximately
26.6 million antenna-frames. With bins 45-55 excluded:

| Domain | Antenna-frames | Frames with off-center max > 0.2 | Mean max off-center |
|---|---:|---:|---:|
| AR | 12.39M | 21.9% | 0.149 |
| PC | 7.01M | 43.5% | 0.219 |
| PI | 7.19M | 24.8% | 0.147 |

Across all domains, only 28.4% of frames exceed `0.2` and 16.5% exceed `0.4`.
Approximately 16.3% of total target power lies above the floor outside the
center band. Most pixel and frame gradients therefore describe stationary
background.

PC is not drawn from the same target marginal: it is almost twice as often
strongly active and its stationary center profile is broader. PI target
validation can have lower loss than the mixed training set because it is an
easier distribution, not because cross-domain generalization is solved.

The four antennas create another imbalance. A dynamic recording often has a
strong event on only one or two antennas. The shared-antenna implementation is
still the correct weight-sharing choice, but event balancing must operate on
antenna-frames or motion support, not only on recording windows.

AR/PC targets were recomputed with the current preprocessing code, while PI
targets came from the older `doppler_traces_pi` set. Before a definitive mixed
run, PI must either be recomputed with the same revision and parameters or
numerically checked for exact pipeline parity. Otherwise the model is being
asked to fit possible generator-version differences in addition to domain
differences.

**Data loader and temporal geometry**

Code inspection supports the dataset pairing and source-target slice:

```text
target frame t -> raw packets 800+t through 800+t+30
340 target frames -> 370 raw packets
```

The split guard also prevents adjacent train/validation windows from sharing
the 31-packet teacher context. No evidence currently identifies the memmap
iterator, real/imag split, antenna reshape, or target slicing as corrupt.

There is, however, a model-side temporal alignment error. After producing 370
temporal features, every implemented model calls interpolation to resize time
to 340. With `align_corners=False`, output index `t` samples approximately:

```text
(t + 0.5) * 370 / 340 - 0.5
```

The natural center of target frame `t` is raw index `t+15`. The discrepancy is
about `-15` packets at the beginning, zero near the middle, and `+15` at the
end. A translation-equivariant convolutional network cannot cleanly undo this
position-dependent warp. The correct operation after same-length temporal
processing is a fixed crop `[..., 15:-15]`, or an explicitly valid
31-sample front end. Tiny-set memorization did not validate this alignment
because a high-capacity network can memorize the fixed position warp.

**Architecture audit**

The current model has several mismatches with the teacher:

1. **Frequency information is destroyed too early.** For one antenna, the
   ordered `[242 subcarriers, real/imag]` input becomes 484 Conv1d channels.
   A kernel-1 projection immediately compresses 484 values to 64, followed by
   BatchNorm and ReLU, before any frequency-axis or phase-aware operation.
   SHARP's difficult step is precisely nonlinear path separation across
   subcarriers. Deeper temporal layers cannot recover information removed by
   this first projection.
2. **The network has no complex or phase-error inductive bias.** Real and
   imaginary inputs are retained, but arbitrary real Conv1d weights are not
   equivariant to packet-wise complex rotations or affine phase slopes.
3. **The output head is a spectral low-pass bottleneck.** It projects 64
   temporal channels to only `4 x 25` coarse spectral features, bilinearly
   upsamples 25 bins to 100, and applies two small 3x3 convolutions. The whole
   head has only about 6.7k parameters. Narrow one-to-three-bin peaks are
   difficult to synthesize from this representation.
4. **Time is warped instead of valid-cropped.** This directly smears transient
   alignment as described above.
5. **The temporal receptive field is already larger than the 31-packet
   teacher window.** Adding more generic temporal residual blocks is not the
   first fix.
6. **Capacity is secondary to representation.** The small model has 0.79M
   parameters and may be too small for all domains, but the prior 2.4M full-2D
   model also learned a center-biased shortcut. The 16-window overfit tests
   prove memorization, not held-out reconstruction.

**Loss audit**

The motion-aware loss improved materially over the previous SmoothL1
objective, but it cannot repair missing conditional features:

- Full-map MSE still has a conditional-mean optimum under uncertain sparse
  events.
- Background leakage applies a constant downward gradient at every exact-floor
  pixel. Under location uncertainty, this makes conservative smooth output
  safer than a sharp but slightly misplaced peak.
- Wasserstein computes `relu(prediction-floor)`. A prediction below the floor
  receives no Wasserstein gradient that could create missing active mass.
- Wasserstein normalizes each spectrum, so it constrains location only after
  mass exists and cannot recover amplitude.
- The target is normalized independently per frame. A center ridge usually
  reaches one, while off-center motion occupies few bins, so global map
  reductions remain highly imbalanced.

Raising `motion_mse_weight` alone is not justified: its weighted term is
already half the best target loss. Multi-resolution STFT, SSIM, or image
gradient losses may later refine shape, but none can reconstruct event timing
that the front end discards or warps.

A future direct-map loss should separate two tasks:

1. off-center support/event detection, using event-balanced focal BCE,
   Tversky, or Dice-style supervision;
2. amplitude and shape regression only where motion is present, using MSE or
   log-amplitude error plus a differentiable motion-only transport term.

Output should also be constrained to the physical target range without a dead
`clamp_min` path. Background suppression should be introduced only after
motion recall is established.

**Training and normalization audit**

- Batch 768 was selected to fill the RTX 4090, not to optimize learning. After
  antenna flattening it produces 3,072 single-antenna examples and only about
  172 optimizer updates per epoch. The whole run made 4,640 updates.
- Rare, localized motion gradients are averaged with thousands of mostly
  stationary antenna-frames in every step.
- Adjacent windows overlap by 91.2%; 121,629 windows do not represent 121,629
  independent examples.
- BatchNorm running statistics are vulnerable to recording pools. With 233
  recordings and pools of 12, the final pool has only five recordings.
  Momentum `0.1` gives those final batches disproportionate influence, and the
  identity of the last pool changes every epoch. This is consistent with the
  abrupt source/target validation spikes despite smooth training curves.
- There is no learning-rate schedule. Constant `1e-3` is not proven wrong, but
  tuning it before correcting representation and geometry would not isolate
  the main failure.
- Combining AR, PC, and PI is not intrinsically invalid because the physical
  map should be shared. It is currently a confound because target marginals,
  generator provenance, and BatchNorm statistics differ. Failure on a fixed
  training example shows that domain shift is not the sole cause.

For the next diagnostic, use event-balanced batches of roughly 64-128 windows
(256-512 antenna examples), GroupNorm or LayerNorm, and one source domain.
GPU saturation is not an optimization objective when it reduces update count
and motion diversity per gradient.

**Comparison with successful literature**

- [SHARP](https://arxiv.org/abs/2103.09924) makes phase cleaning the central
  signal-processing contribution, separates sparse paths, references the
  strongest path, and only then computes micro-Doppler. Its neural network is
  a classifier over Doppler, not a generic raw-CSI-to-Doppler image translator.
- [Widar3.0](https://cswu.me/papers/mobisys19_widar3_paper.pdf) explicitly
  derives a physics-based body-coordinate velocity profile before learning.
  Its paper reports substantially poorer cross-environment performance from
  raw CSI and ordinary Doppler/DFS than from the derived invariant feature.
- [SLNet](https://www.usenix.org/system/files/nsdi23-yang-zheng.pdf) keeps
  STFT as an explicit front end, uses complex-valued spectral enhancement, and
  fuses multiple time-frequency resolutions. It is signal-processing/learning
  co-design, not an unconstrained flatten-and-regress architecture.
- [Optimal preprocessing of WiFi CSI for sensing
  applications](https://arxiv.org/abs/2307.12126) models receiver gain, timing,
  and common phase errors explicitly and shows that they materially hinder
  sensing.
- [LISTA](https://icml.cc/2010/papers/449.pdf) is the established pattern for
  accelerating LASSO-like sparse inference: unroll a small fixed number of
  shrinkage iterations and learn their parameters while retaining the
  dictionary/data-consistency structure.

The common lesson is not simply "use a larger CNN." Successful systems expose
the physical frequency/time structure, explicitly handle complex phase
nuisances, and learn around known transforms.

**Revised primary direction**

The current direct U-Net remains a useful negative baseline, but should not be
scaled further in its present form. The preferred model is a hybrid:

1. Preserve input as `[batch*antenna, 2, subcarrier, time]`.
2. Apply a deterministic phase-error correction baseline first. If it is not
   sufficiently faithful, replace or augment it with a frequency-aware
   complex network or an unrolled LISTA block over SHARP's delay dictionary.
3. Predict an intermediate sanitized complex CFR or phase-correction
   parameters, rather than 34,000 normalized image pixels directly.
4. Apply the known 31-packet Hann STFT, 100-point FFT, magnitude square,
   subcarrier sum, FFT shift, framewise normalization, and floor as fixed
   differentiable layers.
5. Use the exact valid temporal geometry; do not interpolate 370 positions to
   340.

This decomposition makes the learned task "approximate expensive phase/path
sanitization," which matches the original research question. The cheap and
known Doppler transform should not be relearned.

**Decision and next experiment sequence**

1. Do not continue `3wrefrft`; preserve epoch 26 as the direct-map baseline.
2. Recompute PI with the same generator revision or prove exact parity against
   current AR/PC targets.
3. Evaluate the deterministic affine phase-correction baseline on all splits
   and through the frozen original SHARP classifier. This may already provide
   the required speed/accuracy trade-off without a neural surrogate.
4. Add motion-support precision/recall/AUPRC, active-energy recall,
   false-active mass, temporal motion-energy correlation, centroid/quantile
   error, and metrics broken down by domain, activity, recording, and antenna.
5. Correct the 15/15 temporal crop and build a motion-stratified,
   same-recording held-out benchmark. Tiny-set memorization is no longer an
   acceptance test.
6. Build the hybrid sanitized-CFR model. Start on one coherent domain and
   event-balanced batches; add mixed domains only after held-out source
   reconstruction is demonstrated.
7. For final utility, run both:
   - the frozen SHARP classifier trained on teacher maps and evaluated on
     generated maps, testing representation fidelity;
   - a classifier retrained on generated maps, testing whether the surrogate
     remains useful despite distribution shift.

## 2026-07-28

### AR motion-balanced full-resolution experiment

**Question**

Can the corrected full-resolution architecture reconstruct held-out AR motion
when sparse active windows are no longer overwhelmed by stationary windows?

**Run/config/code**

- W&B run: not started.
- Config:
  [`ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml`](../../configs/csi_to_doppler/ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml)
- Model: `unet2d_shared_antenna_full_resolution`.
- Sampler and metrics:
  [`distillation.py`](../../src/wifi_doppler/training/distillation.py).

**Change**

- **Decision:** Use all 18 recomputed AR scenarios for training and source
  validation. Keep `AR-1a/1b/1c` as target validation and final test because
  those scenarios align with the original SHARP classifier workflow.
- **Decision:** Use all 242 subcarriers, exact 31-packet temporal alignment,
  shared per-antenna weights, GroupNorm, and direct 100-bin 2D decoding.
- **Decision:** Increase stride from 30 to 170. This cuts redundant overlap
  from 91.2% to 50% and makes window counts less misleading.
- **Decision:** Score a window by the fraction of antenna-time frames whose
  maximum outside bins 45-55 exceeds 0.2. Per recording, the top quartile is
  motion-rich.
- **Decision:** Every epoch includes each rich window once and an equal number
  of ordinary windows without replacement. Ordinary coverage rotates
  deterministically across epochs. Full batches contain eight rich and eight
  ordinary windows.
- **Decision:** Keep the existing motion-aware loss unchanged. This isolates
  architecture and sampling from loss-weight changes.
- **Decision:** Validation and test remain unbalanced and exhaustive; balancing
  them would hide real-distribution performance.

**Measurements and observations**

- **Measurement:** Implementation tests pass: deterministic rich selection,
  ordinary rotation, 50/50 batches, motion support metrics, unchanged
  evaluation coverage, CPU training, and checkpoint resume.
- **Measurement:** The full AR manifest contains 164 recordings across 18
  scenarios. The configured split produces 10,684 train windows, 3,377 source
  validation windows, and 487 windows in each classifier-compatible target
  validation/test split.
- **Measurement:** Per-recording top-quartile selection marks 2,747 train
  windows rich. Each epoch therefore uses 2,747 rich and 2,747 ordinary
  windows: 5,494/10,684 windows (51.4%). The 7,937 ordinary windows complete a
  deterministic rotation in about 2.9 epochs; no ordinary window is
  permanently excluded.
- **Measurement:** Motion-score quantiles over all train windows are
  `0, 0, 0.0103, 0.0662, 0.2884, 0.9603, 0.9956, 1.0` at
  `0/10/25/50/75/90/99/100%`. The fraction scoring above 0.1 is 43.1%.
- **Observation:** Per-recording ranking is a mild rather than absolute motion
  rebalance: 55.4% of rich windows and 38.8% of ordinary windows score above
  0.1. Ten of 164 recordings have a zero rich cutoff. This occurs because AR
  contains both nearly stationary recordings and recordings active for most
  of their duration. The rule preserves recording/activity coverage but
  “rich” means relative to that recording, not necessarily objectively active.
- **Measurement:** New logged metrics are off-center MSE, active-frame
  precision/recall/F1, and predicted-to-target off-center motion-mass ratio.
- **Measurement:** W&B heatmaps now use fixed motion-rich references for train,
  source validation, and target validation.
- **Observation:** There is no model result yet. This entry documents the
  hypothesis and acceptance gate, not evidence that balancing works.

**Decision**

Use this as the final direct-map diagnostic. Good held-out peaks justify the
frozen SHARP classifier test. Training success with held-out failure points to
representation generalization and explicit phase correction. A persistent
center ridge even on the fixed rich training references terminates further
direct-map CNN tuning.

**Next**

Run the config unchanged, attach the W&B run ID here, and evaluate the support
metrics and fixed heatmaps before interpreting global MSE.

**2026-07-28 pre-run correction:** The final config expands training to
0-80% of every AR recording. Source validation is all AR at
80-90%; target validation is `AR-1a/1b/1c` at 80-90%; final test is the same
three scenarios at 90-100%. This produces 14,342 train, 1,538 source
validation, and approximately 217 target validation/test windows. The earlier
10,684-window and 51.4%-per-epoch measurements describe the superseded
0-60% training split; the sampler behavior is otherwise unchanged.

**Measurement:** Batch 64 caused CUDA OOM on the RTX 4090. The reproducible
config therefore uses batch 32, corresponding to 16 rich and 16 ordinary
windows and an effective shared-antenna network batch of 128.

**Run measurement:** W&B run
[`xhg1tsta`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/xhg1tsta)
completed epoch 1 at 24.46 train windows/s and 239 optimizer steps. Sampled
GPU utilization averaged 30.9% and alternated between 100% compute bursts and
zero; GPU allocation peaked near 71%, while total CPU averaged 11.5% and
process RAM about 4.6 GiB. This identifies batch preparation as the immediate
bottleneck. The config raises preparation workers from four to eight, matching
the eight-recording pool; larger prefetch alone cannot accelerate a producer
that is not filling the queue.

**Correction after warm-up:** Do not stop `xhg1tsta`. Epoch 2 train throughput
rose to 49.67 windows/s, twice epoch 1. GPU utilization averaged 81.2% from
runtime 400-631 seconds, versus 23.9% during the first 400 seconds. The initial
chart was dominated by motion-profile construction, cold filesystem pages,
and the first pass through the memmaps. Eight preparation workers remain a
reasonable next-run setting, but are not a reason to restart this run.

## 2026-07-29

### Full-resolution AR result: center-ridge shortcut remains

**Run/config/code**

- W&B:
  [`xhg1tsta`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/xhg1tsta)
- Config:
  [`ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml`](../../configs/csi_to_doppler/ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml)

**Measurements and observations**

- **Measurement:** Early stopping finished at epoch 25 and restored epoch 17,
  the minimum target-validation loss (`0.020927`).
- **Measurement:** From epoch 1 to 25, train loss fell `0.04991 -> 0.03028`,
  source-validation loss `0.03816 -> 0.02957`, and target-validation loss
  `0.02412 -> 0.02140`. Train and validation did not diverge.
- **Measurement:** At epoch 25, train/source/target active-frame F1 was
  `0.493/0.606/0.329`. The loss-selected epoch 17 had target F1 `0.219`.
- **Measurement:** Restored-best target-test loss was `0.01983`, but
  active-frame precision/recall/F1 was `0.119/0.996/0.212` and off-center
  motion-mass ratio `3.89`. The model labels almost every frame active and
  produces far too much off-center mass.
- **Observation:** Fixed rich training, source-validation, target-validation,
  and final-test heatmaps all retain a dominant center ridge. Large,
  time-varying off-center target structures are replaced by a smooth halo.
  This failure is present on training references, not only held-out data.
- **Observation:** Global and off-center MSE improve because a smooth
  conditional mean reduces average pixel error. Peak-bin MAE is nearly
  constant because the central ridge remains the maximum; neither establishes
  reconstruction of motion peaks.
- **Observation:** `motion_mass_ratio` overflowed to infinity on large AMP
  batches because FP16 predictions were reduced before conversion to FP32.
  This affected logging only, not loss or gradients. Metric reductions now
  cast prediction and target motion maps to FP32.

**Interpretation and confidence**

This is not classic overfitting: training and validation losses improve
together and the model still fails on fixed training examples. It is
underfitting in the functional sense, specifically convergence to a
center-spectrum conditional-mean shortcut. Confidence is high because the
support metrics and all fixed heatmaps agree.

**Decision**

Run the planned frozen SHARP-classifier test for downstream evidence, but do
not enlarge this direct-map CNN or retune its pixel-loss weights. The audit's
termination gate is met. The next modeling direction should explicitly correct
phase or predict sanitized complex CSI before applying the known
differentiable Doppler transform.

### Frozen SHARP classifier comparison

**Protocol and artifacts**

- Frozen classifier:
  `experiments/runs/sharp_baseline/checkpoint_sharp.pt`, SHA-256
  `4f9cc539d4d85f17101cb206d9b0b98747233fb0216acfda64a539e6bc5229d9`.
  It was trained by `notebooks/sharp_reproduction.ipynb` on legacy
  `S1a/S1b/S1c`, labels `E/L/W/R/J`, interval 0-60%, and selected on 60-80%.
- Test data: classifier-compatible `AR-1a/1b/1c`, interval 90-100%, stride 30,
  guard 31. This interval is unseen by the classifier and both students.
  Canonical `J1/J2` recordings are both evaluated as class `J`.
- Students:
  [`xhg1tsta`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/xhg1tsta)
  best epoch 17 and
  [`3wrefrft`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/3wrefrft)
  best epoch 26.
- The classifier's training loader subtracts each method's recording-mean
  spectrum. The evaluation applies the same method-specific centering without
  using target statistics. Reconstruction metrics remain on uncentered maps.
- Reproducible evaluator:
  [`evaluate_doppler_classifier_fidelity.py`](../../scripts/evaluate_doppler_classifier_fidelity.py).
  Full machine-readable output is
  `experiments/runs/doppler_classifier_fidelity/results.json` (ignored run
  artifact, retained on the Windows evaluation machine).

| Method | SHARP fusion accuracy | Off-center MSE | Active precision | Active recall | Active F1 | Motion-mass ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy SHARP maps | **91.36%** | n/a | n/a | n/a | n/a | n/a |
| Recomputed SHARP target | 36.86% | 0 | 100% | 100% | 100% | 1.00 |
| Raw CSI + fixed STFT | 21.70% | 0.12964 | 14.00% | 99.92% | 24.56% | 68.08 |
| Affine correction + fixed STFT | **91.52%** | 0.00141 | 48.63% | 74.77% | **58.93%** | 2.01 |
| `xhg1tsta`, full-resolution 2D | 40.89% | 0.00194 | 14.06% | 99.78% | 24.65% | 3.24 |
| `3wrefrft`, small spatial head | 23.64% | **0.00097** | 41.06% | 17.16% | 24.20% | 0.49 |

**Findings**

- The frozen classifier remains valid: native legacy SHARP test maps score
  91.36%. Affine phase correction plus the fixed 31-packet STFT matches that
  upper bound at 91.52%; their 95% Wilson intervals overlap
  (`89.08-93.21%` and `89.25-93.34%`). Raw STFT collapses to predicting `R`
  for every window.
- The recomputed SHARP targets score only 36.86%. This is a target-generator
  compatibility failure, not a student-only failure. The students were trained
  to imitate maps that do not preserve the frozen classifier's original input
  distribution.
- `xhg1tsta` predicts nearly every frame as active: 99.78% recall, 14.06%
  precision, and 3.24 times the target off-center mass. This quantitatively
  confirms the broad off-center halo seen in its heatmaps.
- `3wrefrft` has the lowest pixel and off-center MSE but only 23.64% classifier
  accuracy. It suppresses motion instead: 17.16% recall and 0.49 motion-mass
  ratio. Lower MSE therefore does not imply better HAR preservation.
- Affine STFT has slightly worse pixel MSE than `3wrefrft` but almost four times
  its classifier accuracy. For the report, downstream accuracy and motion
  support metrics should lead; global MSE should be secondary.

**Decision**

Use affine phase correction plus fixed STFT as the practical result and
primary baseline. Do not spend the remaining time tuning the direct-map CNNs.
Before making claims about neural approximation of SHARP, first reconcile the
recomputed target generator with the legacy SHARP traces; otherwise target
fidelity and student quality remain confounded.

### Legacy-SHARP target retraining setup

The `36.86%` frozen-classifier accuracy of the recomputed targets makes the
previous student comparison unsuitable as the final test of direct
regression. The two report models will therefore be retrained against the
official precomputed SHARP traces used by the classifier.

The complete legacy AR set contains 12 scenario folders. Exact label and frame
matching established the raw-CSI aliases:

| Legacy | Raw CSI | Legacy | Raw CSI |
| --- | --- | --- | --- |
| `S1a` | `AR-1a` | `S1b` | `AR-1b` |
| `S1c` | `AR-1c` | `S2a` | `AR-1d` |
| `S2b` | `AR-1e` | `S3a` | `AR-2a` |
| `S4a` | `AR-3a` | `S4b` | `AR-3b` |
| `S5a` | `AR-4a` | `S6a` | `AR-5a` |
| `S6b` | `AR-5b` | `S7a` | `AR-7a` |

Ninety-two recordings have the expected raw/target frame offset of 1631 or
1632. Five mismatched legacy recordings are excluded rather than cropped or
silently paired: `S4a_L`, `S4b_J1`, `S4b_J2`, `S5a_L`, and `S6b_J1`.

Two configurations are prepared:

- `legacy_sharp_ar_unet2d_shared_antenna_full_resolution.yaml`: the final
  1.314M-parameter model, stride 170 and motion-stratified sampling.
- `legacy_sharp_ar_unet1d_spatial_head_shared_antenna_small.yaml`: the smaller
  spatial-head model, stride 30 and the previous Vast throughput settings.

Both train on all 12 legacy scenarios over 0-80%. Source validation covers all
12 over 80-90%; target validation and final test use classifier-compatible
`S1a/S1b/S1c` over 80-90% and 90-100%, respectively. W&B runs and downstream
classifier results are pending.

## Open Paper-Level Questions

- Is exact SHARP-map reconstruction necessary, or is preserving classifier
  decisions and motion geometry sufficient?
- How much of SHARP's phase sanitization is recoverable from a local raw window?
- Does using all 242 cleaned carriers close the PI-4a gap?
- Is target-domain failure primarily normalization statistics, environment
  shift, person shift, or missing carrier/history information?
- Which metrics correlate with downstream HAR accuracy? Global MSE and global
  peak-bin MAE currently do not establish motion fidelity.
- Should the student approximate intermediate sanitized CSI instead of the
  final normalized Doppler map?
