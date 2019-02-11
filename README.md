# Implementation of Classification
Chainer implementation of SqueezeNet, ResNet, and Knowledge Distillation.


## Implementation Details
- I use pretrained caffe models for SqueezeNet and ResNet
- Data Augmentation: horizontal flip and random erasing
- Input size: 227 $\times$ 227
- KD hyperparameters: $T=2, \alpha=0.8$
- I evaluate our models on evaluation dataset.

## Results

| Model | Accuracy | Speed per epoch |
|:-----------|------------:|:------------:|
| SqueezeNet_v1.1 | 90.5% | 170s |
| ResNet101 | 97.8% | 2200s |
| Squeezenet_v1.1 with KD | 92.8% | 170s |

Knowledge Distillation improved accuracy!
