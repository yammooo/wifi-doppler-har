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

## 2026-05-26 - Few-Shot Softmax Embedding Evaluation

### Question

Can target-domain K-shot prototype inference improve over zero-shot softmax prediction when using the already-trained softmax SHARP model as an embedding extractor?

### Setup

- Encoder checkpoint: softmax SHARP model trained on `PI-1a`, `PI-2a`, `PI-3a`
- Target domain: `PI-4a`
- Persons: all 10 PI identities, `p03`, `p05`-`p13`
- Enrollment source: target-domain windows
- Query source: held-out target-domain windows
- Embedding source: pre-classifier SHARP CNN features from the softmax-trained model
- Prototype method: sample `K` enrollment windows per person, average embeddings into one prototype per person, classify query embeddings by nearest prototype
- Trials per K: 20

### Result

| K enrollment windows/person | Mean query accuracy | Std |
| ---: | ---: | ---: |
| 1 | `0.2046` | `0.0328` |
| 3 | `0.2517` | `0.0210` |
| 5 | `0.2765` | `0.0228` |
| 10 | `0.3044` | `0.0144` |
| 25 | `0.3248` | `0.0184` |
| 50 | `0.3269` | `0.0125` |
| 100 | `0.3416` | `0.0091` |

### Artifacts

- Run directory: [experiments/few_shot_softmax_evaluation/few_shot_softmax_evaluation_20260526_171738](../experiments/few_shot_softmax_evaluation/few_shot_softmax_evaluation_20260526_171738)
- Results JSON: [pi_few_shot_softmax_results.json](../experiments/few_shot_softmax_evaluation/few_shot_softmax_evaluation_20260526_171738/pi_few_shot_softmax_results.json)

![Few-shot softmax embedding accuracy](../experiments/few_shot_softmax_evaluation/few_shot_softmax_evaluation_20260526_171738/pi_few_shot_softmax_accuracy.png)

### Interpretation

Accuracy increases with K, which suggests the prototype evaluation is behaving sensibly: more enrollment windows produce more stable person prototypes. However, the improvement is modest. Small-K settings are weak, and even `K=100` reaches only about `34%` query accuracy.

This indicates that the softmax-trained SHARP encoder is not a strong metric embedding model for few-shot prototype inference. The classifier was trained to separate known identities through a learned linear head, not to make same-person windows cluster tightly under cosine or Euclidean distance. This result strengthens the motivation for supervised contrastive training, where the objective directly optimizes embedding geometry for prototype-style inference.

## 2026-05-27 - Prototypical Training With Raw SHARP Feature Maps

### Question

Does episodic prototypical training improve the SHARP representation when the embedding is the flattened SHARP feature map?

### Setup

- Data: `data/doppler_traces_pi`
- Persons: all 10 PI identities, `p03`, `p05`-`p13`
- Train domains: `PI-1a`, `PI-2a`, `PI-3a`
- Source validation domains: `PI-1a`, `PI-2a`, `PI-3a`
- Target validation domain: `PI-4a`
- Target validation protocol: target-domain enrollment support windows from `PI-4a`, target-domain query windows from held-out `PI-4a`
- Model: SHARP multi-antenna model
- Embedding: flattened pre-classifier SHARP convolutional feature maps
- Objective: 10-way prototypical loss with cosine prototype logits
- Episode shape: `K=5` support windows/person, `Q=10` query windows/person
- Training length: about 4500 sampled prototypical steps

### Result

- Training loss decreased from about `2.30` to roughly `2.15`-`2.20`.
- Training episodic accuracy improved above chance but remained noisy, mostly around `25%`-`35%`.
- Source-domain episodic validation accuracy plateaued around `25%`-`30%`.
- Target-domain episodic validation briefly reached about `50%`, then fell back and stabilized around `30%`.
- Final K-shot comparison against the softmax embedding baseline was not completed for this diagnostic run.

### Artifacts

- Run directory: [experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722](../experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722)

![Prototypical training without embedding head](../experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722/2026-05-27-no-embedding-head.png)

### Interpretation

This run showed that episodic training was not completely random: both loss and accuracy moved away from the 10-way chance baseline. However, the improvement was weak and unstable. The raw flattened SHARP feature map is likely a poor metric-learning space because it is high-dimensional and was originally designed for activity-classification logits, not for compact identity clustering.

The target-domain validation curve should not be interpreted as zero-shot generalization, because target prototypes are built from `PI-4a` enrollment windows. The temporary target-domain spike suggests that some identity structure exists in the Doppler features, but the representation did not settle into a robust prototype space. This motivated adding an explicit projection head.

## 2026-05-28 - Prototypical Training With 128-D Embedding Head

### Question

Does adding a trainable 128-D projection head after the SHARP backbone produce a better prototypical identity embedding?

### Setup

- Data: `data/doppler_traces_pi`
- Persons: all 10 PI identities, `p03`, `p05`-`p13`
- Train domains: `PI-1a`, `PI-2a`, `PI-3a`
- Source validation domains: `PI-1a`, `PI-2a`, `PI-3a`
- Target validation domain: `PI-4a`
- Model: SHARP backbone with multi-antenna encoder
- Embedding head: `Flatten -> LazyLinear(256) -> ReLU -> Dropout -> Linear(128)`
- Embedding normalization: enabled
- Fusion: mean of antenna embeddings
- Objective: 10-way prototypical loss with cosine prototype logits
- Episode shape: `K=5` support windows/person, `Q=16` query windows/person
- Training length: 3000 sampled prototypical steps

### Result

- Training loss decreased more clearly than in the raw-feature run, reaching roughly `2.0` by the end.
- Source-domain validation loss also decreased, reaching roughly `2.05`-`2.10`.
- Target-domain validation loss decreased but remained noisy.
- Episodic train/source/target accuracies stayed low, mostly around `20%`-`30%`.
- The final K-shot comparison against the softmax embedding baseline crashed due to memory pressure, so this run is not a complete method comparison.

### Artifacts

- Run directory: [experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722](../experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722)

![Prototypical training with embedding head](../experiments/few_shot_proto_evaluation/proto_multi_antenna_vs_softmax_baseline_20260527_164722/2026-05-28-with-embedding-head.png)

### Interpretation

The projection head improved the loss behavior, which suggests the model was learning a more suitable embedding space than the raw feature-map baseline. However, the accuracy curves did not improve enough to make this a strong result by themselves. The gap between decreasing loss and weak accuracy may come from compressed cosine logits: cosine similarities are bounded in `[-1, 1]`, so cross-entropy may receive a weak class-separation signal unless logits are temperature-scaled.

This run suggests that architecture alone is not sufficient. The next prototypical run should save the trained model before K-shot evaluation, reduce memory pressure during evaluation, and test temperature-scaled prototype logits such as `temperature=0.1`.
