---
layout: post
title: 'Learning Deep Learning: Lesson 3 - draft'
date: 2026-06-04
---

# Guessing Handwritten Numbers

## This post is part of my Practical Deep Learning journey

This is my experience with [Lesson 3](https://course.fast.ai/Lessons/lesson3.html) of the 
Practical Deep Learning course. If you're just tuning in, I'm working through this material 
in a study group while doing a half-batch at [The Recurse Center](https://www.recurse.com/) 
in May/June 2026. See [Learning Practical Deep Learning (fastai)](2026-06-02-hello-deep-learning.md) for why.

The lessons for Practical Deep Learning are about learning by doing. *Doing* means writing code to 
create, run, and analyze the results of models. The authors thought this would be an effective way for 
students to build intuition about how models trained on real-world data perform on particular tasks, and how 
results vary when you change qualities of the model, data, and training, validation, and testing (architecture, batch sizes, 
learning rate, metric, methods for handling missing data, etc.).

The first two lessons use wrapper functions from the `fastai` library to abstract away the 
details of creating models like Microsoft Research's [ResNet-18 and ResNet-34](https://github.com/kaiminghe/deep-residual-networks)
(published in 2015), which were trained on [ImageNet-1k](https://ieeexplore.ieee.org/document/5206848) (both originally and
[currently available in the pytorch library](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)).
The Practical Deep Learning authors wanted to show just how good image classification models have become since ResNet's 
breakthrough with "residual" or "skip" connections in 2015, and what kind of classification accuracy you can get by fine-tuning 
general models like ResNet on particular labeled image datasets suited to your own purposes. You can try out my 
cloud classifying model (fine-tuned from ResNet-18 using clouds sourced from duck duck go) [here](https://huggingface.co/spaces/emmjab/dllearning?logs=container), 
see the code [here](https://huggingface.co/spaces/emmjab/dllearning/tree/main), and read a stream of 
consciousness blog post of me getting the model into a hugging face repo and website [here](2026-06-03-cloud-classifying.md).

## Guessing Handwritten Numbers

For lesson 3, the task was to get inside the `fastai` wrappers. We're walked through strategies for annotating images 
with labels, separating the images into training, validation, and test sets, writing a linear model, initializing random
weights, and writing a training loop to train the model with a supplied learning rate on our input data.

The task at hand for our linear model is the binary classification of hand-written 3s and 7s.

The toy model has two linear layers and one ReLU ("rectified linear unit") sandwiched between them. It's written as a python 
function that receives a tensor of input data and randomized weights. Each image is passed into the first layer of the model
as a vector of length mxn (the number of pixels in the image). But the final layer has to return a vector of length k 
(the number of different categories, "classes", or "labels" that we could classify the image into). , the from scratch for classifying images of handwritten digits (0-9). The dataset is the famous MNIST database (modified NIST database -- NIST is the National Institute of Standards and )



## "Hidden" is a misleading term

Some layers in a neural network are referred to as "hidden". I think this word is misleading, because it suggests
that they are un-knowable. Each layer of a neural net is a line of code that performs a transformation on a tensor, 
either preserving or reducing the size of the tensor. The inputs are the data that you start with -- images, text, tebles -- and the outputs
depend on you. 
They are "hidden" in the sense that in the sequence of transformations  I would instead call these "internal" layers.

## Historical notes
-- include, but at the bottom in an "extra history" section.

The ResNet developers figured out how to overcome an issue plaguing image classification models where training error
frustratingly increased as depth exceeded ~20 layers. They did this by including "residual" or "skip" "connections" in 
the model. The connections kept a copy of the previous layer's activations to combine with the next layer's transformed activations
so that if it turned out learned activations were already good, they wouldn't be modified during gradient descent.

In 2015, ResNet-152 (152 layers) outperformed the other models in Fei Fei Li's 
[ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/abs/1409.0575) with an error of 3.57%, and also beat
Andrej Karpathy's [5.1% human error rate baseline](https://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/). 


The ResNet model hit the scene just before this Practical Deep Learning course was developed! 

which was trained on the [CIFAR-10 image dataset](https://en.wikipedia.org/wiki/CIFAR-10) 
(published in 2009): 60k images from [80m tiny images](https://en.wikipedia.org/wiki/80_Million_Tiny_Images) 
(published by MIT and NYU in 2008), which were divided into 10 classes, labeled by paid students in an undertaking 
by the Canadian Institute for Advanced Research (CIFAR). 

The idea was to show students that deep learning is in reach: that with the help of NVIDIA GPUs available freely 
or cheaply from cloud providers, it's easy and relatively fast to "transfer" learn on new practical problems by 
fine-tuning multi-layer neural networks that have been pre-trained on huge and general image datasets.

In this lesson, the authors use the MNIST dataset to 