# DDML
This repository contains code for DDML. Some of the methods are adopted from the [ARM](https://github.com/henrikmarklund/arm). We use three datasets to do expeirments on, including rotated MNIST, affNIST and rotated Tiny ImageNet-C.

## Environment
You can install required packages via pip.
```
pip3 install -r requirements.txt
```
## Rotated MNIST Experiment Setting
We rotate the original MNIST dataset by different degrees under this setting.

Train:
```
python train.py --dataset mnist --experiment_name mnist_ddml --num_epochs 200 --n_test_per_dist 300 --epochs_per_eval 10 --epochs_per_eval 10 --log_wandb 0
```

Test: Use the checkpointed model from ```CKPT_FOLDER```
```
python test.py --dataset mnist --eval_on test --ckpt_folders CKPT_FOLDER --log_wandb 0
```

## AffNIST Experiment Setting
The dataset is accessible via this link (https://www.cs.toronto.edu/~tijmen/affNIST/).

Train:
```
python train.py --dataset affnist --experiment_name affnist_ddml --pretrained 1 --prediction_net resnet50 --num_epochs 200 --n_test_per_dist 300 --epochs_per_eval 10 --epochs_per_eval 10 --log_wandb 0
```

Test: Use the checkpointed model from ```CKPT_FOLDER```
```
python test.py --dataset affnist --eval_on test --pretrained 1 --prediction_net resnet50 --ckpt_folders CKPT_FOLDER --log_wandb 0
```

## Rotated Tiny ImageNet-C Experiment Setting.
The dataset is accessible from this link (https://github.com/hendrycks/robustness).

Train:
```
python train.py --dataset rimagenet --experiment_name rimage_ddml --pretrained 1 --prediction_net resnet50 --num_epochs 200 --n_test_per_dist 300 --epochs_per_eval 10 --epochs_per_eval 10 --log_wandb 0
```

Test: Use the checkpointed model from ```CKPT_FOLDER```
```
python test.py --dataset rimagenet --eval_on test --pretrained 1 --prediction_net resnet50 --ckpt_folders CKPT_FOLDER --log_wandb 0
```

## References
Zhang, M., Marklund, H., Gupta, A., Levine, S., & Finn, C. (2020). Adaptive Risk Minimization: A Meta-Learning Approach for Tackling Group Shift. arXiv preprint arXiv:2007.02931.
