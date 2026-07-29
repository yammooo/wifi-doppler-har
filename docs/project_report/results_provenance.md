# Report Result Provenance

This file maps quantitative report claims to saved artifacts. Percentages in
the paper are rounded only for presentation.

## Training Runs

- Final AR model: W&B
  [`xhg1tsta`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/xhg1tsta).
  Epoch losses and the epoch-17 checkpoint selection are reproduced in
  `figures/data/xhg1tsta_epoch_history.csv`.
- Tiny 30-subcarrier overfit: W&B
  [`cwj0myug`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/cwj0myug).
  The final fixed-set MSE is `0.00003881`; its sampled curve is in
  `figures/data/cwj0myug_overfit_history.csv`.
- Tiny 242-subcarrier overfit: W&B
  [`tm8nyt19`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tm8nyt19).
  The final fixed-set MSE is `0.00004194`; its sampled curve is in
  `figures/data/tm8nyt19_overfit_history.csv`.
- Small shared-antenna spatial head: W&B
  [`3wrefrft`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/3wrefrft).
- The final split, sampler, loss, and architecture are defined in
  `../../configs/csi_to_doppler/ar_motion_balanced_unet2d_shared_antenna_full_resolution.yaml`.
  Its run summary reports 3,646 rich and 3,646 ordinary windows per epoch.
- Model parameter counts and architectural limitations are recorded in
  `../research/csi_to_doppler_log.md` and can be regenerated with the model
  builders in `../../src/wifi_doppler/models/`.

## Frozen-Classifier Evaluation

All entries in Table II, including precision, recall, F1, off-center MSE, and
motion-mass ratio, come from:

`../../experiments/runs/doppler_classifier_fidelity/results.json`

The reported `sharp_fusion_accuracy` values are:

| Representation | Exact value |
|---|---:|
| Legacy SHARP | 0.9136490251 |
| Recomputed SHARP | 0.3685674548 |
| Raw fixed STFT | 0.2169680111 |
| Affine correction + fixed STFT | 0.9151599444 |
| `xhg1tsta` | 0.4089012517 |
| `3wrefrft` | 0.2364394993 |

The 95% Wilson intervals quoted in the report are derived from 656 correct
legacy predictions out of 718 windows and 658 correct affine-STFT predictions
out of 719 windows.

## Dataset and Figures

- The measured 164-recording AR inventory and the 28.4% active-frame statistic
  are recorded in `../research/csi_to_doppler_log.md`.
- `figures/data/ar1a_c_16920_comparison.npz` contains the fixed target, neural,
  and affine-STFT arrays used in Figure 4. It was exported by
  `export_qualitative.py` using the saved `xhg1tsta` checkpoint and the same
  `affine_correct` and `fixed_stft` functions as the classifier evaluation.
- Regenerate all report figures with:

  ```bash
  python figures/generate_figures.py
  ```
