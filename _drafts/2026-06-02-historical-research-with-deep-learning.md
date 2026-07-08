---
layout: post
title: historical research with deep learning
date: 2026-06-02 20:40 -0400
---

# Mess from writing up lesson 3: MNIST

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

### Common Kaggle Metrics

Common ones by problem type:

  Classification:
  - Accuracy — fraction correct, what we've been discussing
  - AUC-ROC — how well the model separates classes across all confidence thresholds, not just at 0.5
  - F1 score — balances precision and recall, useful when classes are imbalanced (e.g. 99% of images are
  not fraud) 
  - Log loss — actually the cross entropy loss, used when Kaggle wants to reward confident correct
  predictions and penalize confident wrong ones 

  Regression:
  - RMSE — root mean squared error
  - MAE — mean absolute error
  - RMSLE — root mean squared log error, used when the target spans many orders of magnitude (e.g. house
  prices)

  Object detection / segmentation:
  - mAP — mean average precision, measures how well bounding boxes overlap with ground truth
  - IoU — intersection over union, fraction of overlap between predicted and true region

  NLP:
  - BLEU score — for translation tasks, measures overlap with reference translations
  - F1 on tokens — for question answering, measures overlap between predicted and true answer spans

  The choice of metric is driven by what actually matters for the problem — F1 when class imbalance
  matters, RMSLE when you care more about relative than absolute error, etc. One of the first things to
  do on any Kaggle competition is understand why they chose that specific metric.