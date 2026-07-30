# Report Result Provenance

Percentages in the report are rounded only for presentation.

## Training Runs

- Full-resolution model: W&B
  [`az9k6ori`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/az9k6ori),
  best epoch 45. Its epoch history is
  `figures/data/az9k6ori_epoch_history.csv`; the checkpoint SHA-256 is
  `01e0b11e389df3c4d6f40e5d7528de53cbe676dc81e28585476e4fe1e2dcebec`.
- Small spatial-head model: W&B
  [`hpm4mjl4`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/hpm4mjl4),
  best epoch 21. Its epoch history is
  `figures/data/hpm4mjl4_epoch_history.csv`; the checkpoint SHA-256 is
  `67a682227b393b348ea7fee0d2777a4d9521e51bbdb073e14ed09b1d2e4b2243`.
- Tiny 30-subcarrier overfit: W&B
  [`cwj0myug`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/cwj0myug),
  final fixed-set MSE `0.00003881`.
- Tiny 242-subcarrier overfit: W&B
  [`tm8nyt19`](https://wandb.ai/yammo-unipd/wifi-doppler-har/runs/tm8nyt19),
  final fixed-set MSE `0.00004194`.

The final configurations are
`../../configs/csi_to_doppler/legacy_sharp_ar_unet2d_shared_antenna_full_resolution.yaml`
and
`../../configs/csi_to_doppler/legacy_sharp_ar_unet1d_spatial_head_shared_antenna_small.yaml`.

## Frozen-Classifier Evaluation

All entries in Table II come from:

`../../experiments/runs/doppler_classifier_fidelity/legacy_sharp_results.json`

The evaluator is
`../../scripts/evaluate_doppler_classifier_fidelity.py`. It uses 718
S1a/S1b/S1c windows from the 90--100% interval and the frozen classifier
checkpoint with SHA-256
`4f9cc539d4d85f17101cb206d9b0b98747233fb0216acfda64a539e6bc5229d9`.
The direct distributed-map loader and prepared official target produce identical
predictions.

| Representation | Accuracy | Off-center MSE | Active F1 | Mass ratio |
|---|---:|---:|---:|---:|
| Official precomputed SHARP Doppler | 0.9136490251 | 0 | 1.000000 | 1.000000 |
| Raw fixed STFT | 0.2172701950 | 0.128147833 | 0.356539 | 33.280733 |
| Affine-STFT | 0.9136490251 | 0.000154666 | 0.942881 | 0.982350 |
| `az9k6ori` | 0.6309192201 | 0.001061870 | 0.758329 | 1.211515 |
| `hpm4mjl4` | 0.4331476323 | 0.002757908 | 0.504890 | 2.216600 |

The 95% Wilson intervals are `89.08-93.21%` for SHARP and affine-STFT,
`59.50-66.54%` for `az9k6ori`, and `39.74-46.97%` for `hpm4mjl4`.

## Dataset and Figures

- The prepared manifest contains 92 aligned recordings from all 12 official
  official AR scenario folders. It excludes `S4a_L`, `S4b_J1`, `S4b_J2`,
  `S5a_L`, and `S6b_J1` because their raw/target frame offsets are invalid.
- `figures/data/s1a_c_16920_legacy_comparison.npz` contains the official
  target, epoch-45 neural output, and affine-STFT output used in Figure 4.
- Regenerate all report figures with:

  ```bash
  python figures/generate_figures.py
  ```
