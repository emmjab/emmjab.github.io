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
  
## Mess from writing up lesson 4: patent NLP
## HISTORY. PUT SOMEWHERE LATER

RNNs (for text) and CNNs (for images) have historically been two distinct lineages of deep learning model architecture, 
though today the principles of recurrence exist in a range of contemporary models. The transformer architecture, introduced
with its catchy title, ["Attention is all you Need" (2017)](https://arxiv.org/abs/1706.03762), was a recent development 
on the RNN lineage. Researchers had long been chasing improvements to the reigning language model architecture, [Long Short-Term Memory (LSTM)](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf), 
introduced in 1997 to fix issues with overfitting, 
-- the issue The original fastbook NLP lesson, [(chapter 10)](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb) 
was based on a training strategy called [ULMFiT](https://arxiv.org/abs/1801.06146) (Universal Language Model Fine-Tuning) 
developed by Jeremy Howard, together with Sebastian Ruder, that vastly improved LSTM by mimicking the successes of vision 
model transfer learning. When they developed ULMFiT, they used Salesforce Research's state-of-the-art [AWD-LSTM](https://docs.fast.ai/text.models.awdlstm.html), 
which, like transformers, was also [introduced in 2017](https://arxiv.org/abs/1708.02182). The model was named for its 
components:
- **A**verage-SGD (Stochastic Gradient Descent), 
- **W**eight-**D**ropped,
- **LSTM** (Long Short-Term Memory)


The path to Transfomers for langauge tasks (classification, generation, translation, etc.) can be paralleled with the path 
to ResNet for image classification.
While ResNet was developed using the ImageNet dataest, 
Penn Treebank (PTB). PTB was a tiny, heavily cleaned dataset of 1989 Wall Street Journal articles. It had a restricted vocabulary of only 10,000 words. 

In 2017-2018, these improved LSTMs reigned, but by the time Jeremy Howard prepared the 2022 version of lectures, it was 
clear transformers had won out.


"Unlike a traditional deep neural network, which uses different parameters at each layer, RNN shares the same parameters 
(above) across all steps. This reflects the fact that we are performing the same task at each step, just with different 
inputs. This greatly reduces the total number of parameters we need to learn."
FINISH THE HISTORY SECTION LATER, it's too interesting for right now. Remember this happened last time.

Researchers made many improvements to RNNs , trying to remove the shortcomings of LSTMs onwards from 
their [debut in 1997](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf). Similar to ResNet development, text
datasets were developed as


relationship, researchers came up with Recurrent Neural Networks (RNNs). The main difference between CNNs and RNNs is that
CNNs have different weights between layers, and in an RNN the weights are shared between layers.

The model's task will be to predict the next word, which is initially hidden (just like the label of image training dataset); but then the next word will be sent into
the model! Thus, the model learns the correct "label" for each word before the end of the batch, and that knowledge is 
baked into the parameters with backpropogation at the end of the training loop. Here was the tension for NLP models: 
needing to "remember" sequential relationships between words, including grammars that control words that are not next to each 
other and pronouns whose meanings carry between sentences, but not overfitting. This kind of challenge doesn't exist for 
image classifying.

=======


i like this paragraph that i wrote but it's not exactly correct:
Lesson 4 is the NLP lesson. Here we move from vision models (trained to pick out features of images) to natural language 
processing models (trained to pick out features of text). Since researchers have produced vision models that perform so 
well (e.g. the ResNet from 2015 that we met earlier), wouldn't it be great to just figure out how to make documents 
of text fit into the same kinds of tensors of numbers that we used for encoding the images (normalized pixel values), and 
send it through those models? "Why yes, that would be a great strategy!" say the authors of the course, as Jeremy Howard 
and Sebastian Ruder proceeded to create what would have been the state-of-the-art training method for Recurrent Neural 
Networks (RNNs) like Long Short-Term Memory (LSTM) in 2018, had `transformers` in the form of OpenAI's GPT (June) and 
Google's BERT (October) not been released the same year.