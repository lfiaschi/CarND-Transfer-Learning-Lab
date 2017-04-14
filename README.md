# Transfer Learning Lab with VGG, Inception and ResNet
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lab, I used Keras to explore feature extraction with the VGG, Inception and ResNet architectures. The models used were trained for days or weeks on the [ImageNet dataset](http://www.image-net.org/). Thus, the weights encapsulate higher-level features learned from training on thousands of classes.

Bottleneck features were precomputed **bottleneck features** for each (network, dataset) pair. Dataset are:

- cifar dataset
- german traffic sign

Because the base network weights are frozen during feature extraction, the output for an image will always be the same. Thus, once the image has already been passed once through the network we can cache and reuse the output.

The files are encoded as such:

- {network}_{dataset}_bottleneck_features_train.p
- {network}_{dataset}_bottleneck_features_validation.p

network can be one of 'vgg', 'inception', or 'resnet'
dataset can be on of 'cifar10' or 'traffic'

These can be downloaded from the following links:

- [![vgg](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b432_vgg-100/vgg-100.zip)]
- [![resnet](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b634_resnet-100/resnet-100.zip)]
- [![inception](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5834b498_inception-100/inception-100.zip)]

The file `main.py` contains my implementation of training and testing on the bottleneck features.
The file `Baseline CIFAR.ipynb` contains a baseline on the CIFAR dataset using Lenet implented in Keras

