# PI Experiment Log

This file records person-identification experiments on SHARP Doppler traces. Entries are intentionally factual and may include preliminary results; final report text should be written later from the cleaned-up subset of these runs.

## Current Modeling Choice

We are using the SHARP-style CNN as a baseline encoder/classifier for person identification. The input is the four-antenna Doppler window from `data/doppler_traces_pi`.

For PI, the training objective should match the test-time decision surface: the model predicts identity from the fused four-antenna output. Earlier SHARP activity-recognition reproduction trained the shared single-antenna CNN by applying cross-entropy to each antenna independently, then fused antenna decisions. For PI we changed the objective to compute cross-entropy on the fused model output instead.

Current baseline:

- Backbone: SHARP `SingleAntennaModel`
- Multi-antenna wrapper: shared backbone over four antenna streams
- Fusion: mean of antenna logits
- Loss: cross-entropy on fused logits
- Task label: person identity token parsed from filename, e.g. `p03`

Rationale: the model is evaluated using all four antennas, so optimizing the fused output is more consistent than optimizing each antenna as a standalone classifier.

## 2026-05-25 - 5-Person Same-Domain Sharp Model Mean Fusion

### Question

Can SHARP Doppler traces support basic person identification in a same-domain PI setting?

### Setup

- Data: `data/doppler_traces_pi`
- Train domain: `PI-1a`
- Validation domain: `PI-1a`
- Split: temporal split inside each trace
  - train: `split=(0, 0.6)`
  - validation: `split=(0.6, 0.8)`
- Persons: `p03`, `p05`, `p06`, `p07`, `p08`
- Chance accuracy: 20%
- Model: SHARP CNN backbone with four-antenna wrapper
- Fusion: `mean`
- Objective: cross-entropy on fused four-antenna logits
- Epochs: 25
- Window size: 340
- Window stride: 30

### Result

- Best validation accuracy: `0.8464`
- Restored validation accuracy: `0.8464`
- Best epoch: 20

### Artifacts

- Run directory: [outputs/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103](/C:/Users/gianm/Development/wifi-doppler-har/outputs/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103)
- Model checkpoint: [model.pt](/C:/Users/gianm/Development/wifi-doppler-har/outputs/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103/model.pt)
- Training curves: [training_curves.png](/C:/Users/gianm/Development/wifi-doppler-har/outputs/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103/training_curves.png)
- Confusion matrix: [confusion_matrix.png](/C:/Users/gianm/Development/wifi-doppler-har/outputs/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103/confusion_matrix.png)

### Interpretation

This is a sanity-check result, not the final contribution. Accuracy is clearly above chance, which suggests SHARP Doppler contains person-specific information for these five identities in the same PI domain.

The result does not yet establish domain robustness because training and validation are both from `PI-1a`. The temporal split may also contain correlated neighboring windows, so this should be treated as an optimistic baseline.

### 2026-05-25 10-Person Cross-Domain Sharp Model Mean Fusion

- Expand to all available PI identities.
- Evaluate cross-domain transfer with leave-one-PI-domain-out splits.

### Question

Does the SHARP Doppler classifier generalize across PI domain shifts?

### Intended Setup

- Data: `data/doppler_traces_pi`
- Persons: all available PI identities `p03`, `p05`-`p13`
- Model: SHARP CNN backbone with four-antenna wrapper
- Fusion: `mean`
- Objective: cross-entropy on fused four-antenna logits
- Primary protocol: train on three PI domains and test on the held-out fourth domain

- Train: `PI-1a`, `PI-2a`, `PI-3a`
- Test: `PI-4a`
