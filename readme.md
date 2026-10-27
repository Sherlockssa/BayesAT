# BayesAT: Bayes-Guided Progressive Distillation for Semi-Supervised Adversarial Training

## 💡 Introduction

Existing Semi-Supervised Adversarial Training (SSAT) methods typically rely on static supervision—either hard pseudo-labels or soft labels with a fixed temperature. This assumes a stationary learning process, which fails to accommodate the student's evolving capability, especially when the teacher's performance plateaus and becomes over-confident.

To address this, we reframe SSAT as knowledge distillation from a Bayes teacher and formalize a **stage-dependent bias-variance tradeoff**. We propose **BayesAT**, a Bayes-guided progressive distillation framework that dynamically balances this tradeoff through two complementary mechanisms:
1. **Variable-Temperature Progressive Distillation**: A low-to-high temperature warming schedule that transitions from sharp supervision (for boundary establishment) to smooth supervision (for variance control).
2. **Confidence-Adaptive Sample Reweighting**: A sample-wise mechanism that uses teacher confidence as a proxy to down-weight ambiguous supervision.

By operating along complementary axes (stage-wise vs. sample-wise), BayesAT achieves a superior accuracy-robustness trade-off across CIFAR-10/100 and ImageNet-200.

## Requirements

### Dependencies 
```markdown
python==3.10
coloredlogs==15.0.1
matplotlib==3.10.6
numpy==2.0.1
pandas==2.3.3
Pillow==11.3.0
PyYAML==6.0
torch==2.9.0+cu126
torchattacks==3.5.1
torchvision==0.24.0+cu126
tqdm==4.67.1
```

### Verified working hardware environment

- Ubuntu 22.04.5 LTS
- CUDA 12.6
- NVIDIA RTX 4090


## Example usage
```
python tunner.py
```
Obtain an MT teacher, and then replace the method with BayesAT, and simultaneously modify train_mt to train_BayesAT.
Run again
```
python tunner.py
```
