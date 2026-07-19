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


The idea was then to give data types to each of the columns, mainly distinguishing "categorical" (like "fiModelDescriptor"), 
from "continuous" (like "MachineHoursCurrentMeter") variables. Categorical variables could also be ordered, e.g. ProductSize 
from small to big. Much attention was also paid to methods for handling the dates. The authors expected that there might 
be e.g. seasonal patterns. So, instead of treating the dates like a continuous increasing number or making each day a unique 
category, they wrote a function to create a bunch of extra columns representing different parts of the date (e.g. year, 
month, day) and then more granular things like "day of week".

The lesson also involved discussion on what to do about missing data. In the video lecture, Jeremy gave an example of a 
case where a particular field was only populated if a sale was completed, and that all sold orders were updated in the db 
on Sunday. In a case like this, the model might simply learn that if the date is Sunday, then the sale would go through.
This is unfortunately not useful for predicting whether a sale would happen.

## 

## The Metric

For this task, the metric we're working with is the "root mean squared log error (RMSLE) between the actual and predicted 
auction prices". So we'll preprocess the SalePrice column to be the log of the SalePrice so when we calculate RMSE later
it will already be on the log(SalePrice).

## Making a Decision Tree



ProductGroup == Excavator


## ROGII - Wellbore Geology Prediction

// remove this paragraph
I could continue this post with this data, but I'm itching to start applying what I've learned in these lessons to untried problems. The lessons are great for
building a mental model from the model (transfer learning??) and for providing a worked example with parameters pre-selected
to perform well that I can refer to when updating dependencies and trying to make sense of train/validation loss and metrics
(is .21 good? bad? everything is relative!).
//

I found an active competition that uses tabular data and has a kind of "timeseries" feel to it, like the bulldozer problem. 
This is the [ROGII - Wellbore Geology Prediction Competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).
I know practically nothing about geology and even less about drilling for oil! So this is mostly just a test of how **well** 
I can figure out how to apply the tabular data lesson to other problems. Depending on how **well** this goes, I might try 
to do some research/make improvements to whatever my first pass amounts to. **Well**, let's get started.

Here's the task:
> The competition data comprises horizontal well trajectories and vertical reference logs (Typewells) used for geological 
> prediction. Your goal is to predict the TVT (True Vertical Thickness) for the evaluation zone of each horizontal well.

The competition organizers ([ROGII](https://rogii.com/)) claim that helping solve this real world problem can make "drilling 
safer, more efficient, and lower in emissions." ROGII is a Houston, Texas-based software company in the geosteering and geoscience 
domain that builds sw to help oil and gas "subsurface teams plan, steer, and monitor wells".

### The Data: Gamma Rays detected at intervals

Horizontal Wells with geological formation reference depths from Vertical Type Wells



## HISTORY OF GEOLOGY; maybe add it in and the earthquake cold war document but not right now

Geology research gained individual recognition in the US government in 1878-1879 when the US Geological Survey was 
separated out from the US Coast and Interior Survey; both of these surveys were to report their findings to the 
Land Office "for the purposes of intelligent administration". [*Surveys of the territories.*](https://hdl.loc.gov/loc.law/llserialsetce.01861_00_00-006-0005-0000)
*Letter from the acting president of the National Academy of Sciences transmitting a report on the surveys of the territories. December 3, 1878.*

> "The best interests of the public domain require, for the purposes of
> intelligent administration, a thorough knowledge of its geological struct
> ure, natural resources, and products. The domain embraces a vast
> mineral wealth in its soils, metals, salines, stones, clays, &c. To meet
> the requirements of existing laws in the disposition of the agricultural,
> mineral, pastoral, timber, desert, and swamp lands, a thorough investi
> gation and classification of the acreage of the public domain is impera
> tively demanded. The committee, therefore, recommend that Con
> gress establish, under the Department of the Interior, an independent
> organization, to be known as the United States Geological Survey, to be
> charged with the study of the geological structure and economical re
> sources of the public domain, such survey to be placed under a director,
> who shall be appointed by the President, and who shall report directly
> to the Secretary of the Interior.
> It should be specially provided that the director and members of the
> geological survey, charged as they are with the investigation of the
> natural resources of the public domain, shall have no personal or private
> interests in the lands or mineral wealth of the region under survey, and
> shall execute no surveys or examinations for private parties or corpora
> tions."

in 1879. This was by recommendation in of the National Academy of Sciences. Surveyors would travel westward, checking out rocks for their "geological structure, 
mineral resources, and products of the national domain". Was there coal or oil to be claimed? Seventy years later, geology
gained another boost when Cold War nuclear determent policies created a need to be able to better monitor earthquakes.






======
history to move later;
The course 
authors also included some history of the development of deep learning methods, and I spent some time skimming papers they
shared and some others to contextualize some of the incentives for these developments. Emphasis on the word "some", because
if left to my own devices I could spend a long, long time in that rabbit hole.
