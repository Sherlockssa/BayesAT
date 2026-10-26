# BayesAT: Bayes-Guided Progressive Distillation for Semi-Supervised Adversarial Training

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