# CSI-to-Doppler Research Log

Last updated: 2026-07-24 (Europe/Rome)

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
- Training CLI: [train_csi_to_doppler.py](../../scripts/train_csi_to_doppler.py)
- Dataset pairing/windowing: [csi_to_sharp_doppler_dataset.py](../../src/wifi_doppler/data/csi_to_sharp_doppler_dataset.py)
- Loss and metrics: [distillation.py](../../src/wifi_doppler/training/distillation.py)
- SHARP target generation: [preprocess_sharp_pi.py](../../src/preprocessing/preprocess_sharp_pi.py)

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
| 2026-07-24 | [`tio9b4ad`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tio9b4ad) | 1D U-Net plus spatial 2D head, motion-aware MSE | memmap, batch 256, 8 recordings/batch | running | epochs 2-4 trained at 434-437 samples/s |

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
