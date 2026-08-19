BASELINE (undisguised attacks):
  Random Forest: 99.0%
  LSTM:          92.2%

UNDER EVASION (detection rate per operator):
  comment     | RF 98.2% | LSTM 92.0%
  case        | RF 99.0% | LSTM 92.2%
  whitespace  | RF 99.0% | LSTM 94.0%
  encoding    | RF 38.6% | LSTM 22.0%   <-- headline finding

KEY FINDING: URL-encoding evasion collapses both detectors.
The deep model (LSTM) degraded worse than the classical (RF).


ADVERSARIAL TRAINING (Random Forest, encoding attack):
  BEFORE:  detection on encoded attacks = 38.8% | benign false positives = 0.1%
  AFTER:   detection on encoded attacks = 99.6% | benign false positives = 0.1%

KEY FINDING: Adversarial training repaired the encoding vulnerability
(39% -> 99.6%) with no increase in false positives on benign traffic.
Replicates Floris et al. (2025) on a standalone ML classifier.

LSTM ADVERSARIAL TRAINING (encoding attack):
  BEFORE: detection on encoded attacks = 64.0% | benign FP = 0.4%
  AFTER:  detection on encoded attacks = 99.3% | benign FP = 0.1%

FULL PICTURE (both detectors):
  Encoded-attack detection BEFORE -> AFTER adversarial training:
    Random Forest: 38.8% -> 99.6%   (benign FP steady at 0.1%)
    LSTM:          64.0% -> 99.3%   (benign FP 0.4% -> 0.1%)
  Adversarial training repaired both detectors at no cost to benign accuracy.
  Replicates Floris et al. (2025) for standalone ML classifiers.