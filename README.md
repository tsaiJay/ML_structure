# ML_structure
Template for deep learning models using PyTorch

1. Modify or add parameters to be used in the cfg_setting.yaml.

2. Start training using
```
python main.py --c cfg_setting.yaml
```

## Optional
Some packages are utilized for optimizing the interface and are unrelated to the main functionality. If they are not desired, they can be optionally excluded from installation. Consequently, the corresponding code pertaining to these packages needs to be removed.
```
pip install wandb
pip install tqdm
```
