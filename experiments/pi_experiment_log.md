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

- Run directory: [experiments/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103](../experiments/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103)

![5-person training curves](../experiments/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103/training_curves.png)

![5-person confusion matrix](../experiments/pi_classification/pi_5persons_same_domain_sharp_model_20260525_152103/confusion_matrix.png)

### Interpretation

This is a sanity-check result, not the final contribution. Accuracy is clearly above chance, which suggests SHARP Doppler contains person-specific information for these five identities in the same PI domain.

The result does not yet establish domain robustness because training and validation are both from `PI-1a`. The temporal split may also contain correlated neighboring windows, so this should be treated as an optimistic baseline.

## 2026-05-25 - 10-Person Cross-Domain Sharp Model Mean Fusion

This rerun logs both source-domain validation and target-domain validation, so we can separate ordinary train/validation learning from held-out-domain generalization.

### Question

Does the SHARP Doppler classifier generalize across PI domain shifts when trained on three PI domains and tested on a fourth?

### Setup

- Data: `data/doppler_traces_pi`
- Persons: all available PI identities `p03`, `p05`-`p13`
- Model: SHARP CNN backbone with four-antenna wrapper
- Fusion: `mean`
- Objective: cross-entropy on fused four-antenna logits
- Protocol: train on three PI domains, validate on both source domains and a held-out fourth domain
- Train domains: `PI-1a`, `PI-2a`, `PI-3a`
- Source validation domains: `PI-1a`, `PI-2a`, `PI-3a`
- Target validation domain: `PI-4a`
- Split: temporal split inside each trace
  - train: `split=(0, 0.6)`
  - validation: `split=(0.6, 0.8)`
- Epochs: 25
- Window size: 340
- Window stride: 30

### Result

- Best target validation accuracy: `0.3159`
- Restored target validation accuracy: `0.3159`
- Best target epoch: 14
- Final train eval accuracy: `0.7643`
- Final source-domain validation accuracy: `0.7032`
- Final target-domain validation accuracy: `0.2986`
- Final train eval loss: `0.7294`
- Final source-domain validation loss: `1.4194`
- Final target-domain validation loss: `2.2990`

### Artifacts

- Run directory: [outputs/pi_classification/pi_all_persons_123_train_4_test_sharp_model_20260525_165437](../experiments/pi_classification/pi_all_persons_123_train_4_test_sharp_model_20260525_165437)

![10-person training curves](../experiments/pi_classification/pi_all_persons_123_train_4_test_sharp_model_20260525_165437/training_curves.png)

![10-person target-domain confusion matrix](../experiments/pi_classification/pi_all_persons_123_train_4_test_sharp_model_20260525_165437/confusion_matrix.png)

### Interpretation

This result makes the domain gap clearer than the previous run. The model learns the source domains: final train accuracy is about `76%`, and source-domain validation reaches about `70%` using a held-out temporal split from the same PI domains. However, performance on the unseen `PI-4a` domain peaks at only about `32%` and finishes around `30%`.

This suggests the SHARP Doppler representation contains person-identification signal, but a standard fused softmax classifier does not learn identity features that transfer cleanly across the PI domain shift. The gap is large despite the meeting-room label being the same, because `PI-4a` differs in monitor position, Tx/Rx link, NLOS obstruction, and TP-Link receiver configuration.

For the project direction, this is a useful baseline rather than a blocker. It motivates few-shot target-domain enrollment: instead of expecting zero-shot transfer to `PI-4a`, the next question is whether a small number of target-domain examples per identity can adapt the embedding through prototype inference.

## Planned - Few-Shot Target-Domain Enrollment

### Question

Can a small number of target-domain enrollment examples improve person identification under PI domain shift?

### Important Distinction

This experiment is not true unseen-person few-shot identification. The dataset only has 10 PI identities, so holding out people would leave too few identities for a strong training/evaluation protocol.

The intended first few-shot setting is known-identity target-domain adaptation:

- Train identities: all 10 PI people
- Train domains: source PI domains, e.g. `PI-1a`, `PI-2a`, `PI-3a`
- Target domain: held-out PI domain, e.g. `PI-4a`
- Target enrollment: sample `K` windows per person from the target domain
- Target query: classify the remaining target-domain windows

This evaluates whether a few target-domain examples can compensate for the domain shift observed in the zero-shot softmax baseline.

### Protocol

1. Train an encoder on source domains.
2. Remove or ignore the final softmax classifier.
3. Extract embeddings from target-domain windows.
4. For each person, sample `K` target-domain enrollment windows.
5. Average their embeddings to form one prototype per person.
6. Classify target-domain query windows by nearest prototype.

Candidate values:

- `K = 1, 3, 5, 10`
- Distance: cosine similarity or Euclidean distance after embedding normalization
- Repeat random enrollment sampling several times and report mean/std accuracy

### Baselines

Softmax-trained encoder:

- Use the current fused softmax classifier as a cheap embedding baseline.
- This does not optimize a few-shot objective.
- It only tests whether ordinary cross-entropy training accidentally learns an embedding useful for prototype inference.

Supervised contrastive encoder:

- Train with same-person positives and different-person negatives.
- Evaluate with the same K-shot prototype protocol.
- This better matches the few-shot objective because the training loss directly shapes embedding distances.

### Expected Comparison

Report at least:

- Zero-shot softmax prediction on held-out target domain
- Softmax-trained embedding + K-shot prototype inference
- Supervised contrastive embedding + K-shot prototype inference

The key result is whether target-domain enrollment closes part of the gap between source-domain validation accuracy and held-out-domain zero-shot accuracy.
