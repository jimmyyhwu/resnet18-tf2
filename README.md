# resnet18-tf2

The official TensorFlow ResNet [implementation](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet) does not appear to include ResNet-18 or ResNet-34.

This codebase provides a simple ([70 line](resnet.py)) TensorFlow 2 implementation of ResNet-18 and ResNet-34, directly translated from PyTorch's torchvision [implementation](https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py). The model outputs have been verified to match those of the torchvision models with floating point accuracy.

This code was tested with the following package versions:

```
tensorflow==2.4.1
pytorch==1.2.0
torchvision==0.4.0
```
