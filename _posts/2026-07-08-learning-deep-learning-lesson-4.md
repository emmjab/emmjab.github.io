---
layout: post
title: US patent classifying in Lesson 4
date: 2026-07-08 21:53 -0400
---

# Intro to Huggingface Transformers via a US Patent Classification Task

**This post is part of my Practical Deep Learning journey.**

This is my experience with [Lesson 4](https://course.fast.ai/Lessons/lesson4.html) of the 
Practical Deep Learning course. If you're just tuning in, I'm working through this material 
in a study group while doing a half-batch at [The Recurse Center](https://www.recurse.com/) 
in May/June 2026. See [Learning Practical Deep Learning (fastai)](https://emmjab.github.io/2026/06/02/hello-deep-learning.html) for why.

Lesson 4 is the NLP lesson. Here we move from vision models (trained to pick out features of images, e.g. for image 
class prediction) to natural language processing models (trained to pick out features of text, e.g. for next word prediction). 
Since researchers have produced vision models that have saved researchers so much time solving new problems by transfer 
learning between datasets (e.g. the 2015 ResNet trained on ImageNet that we practiced fine-tuning on a new dataset, in 
my case, images of [different types of clouds](https://emmjab.github.io/2026/06/03/cloud-classifying.html)), wouldn't it be great to do this for text? "Why yes, that would be a great idea!" say the authors of the course, as Jeremy 
Howard and Sebastian Ruder proceeded in 2018 to create what would have been a state-of-the-art transfer learning method 
for Recurrent Neural Networks (RNNs) like Long Short-Term Memory (LSTM), had `transformers` in the form of OpenAI's GPT 
(June) and Google's BERT (October) not been released the same year.

This lesson is a great example of the authors sticking to their goal of showing students deep learning is still evolving. 
The first version of the NLP lesson ([fastbook chapter 10](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)) 
was built on what had been the state-of-the-art RNN at the time, a [Long Short-Term Memory (LSTM) model](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf) 
called [AWD-LSTM](https://arxiv.org/abs/1708.02182) released by Salesforce Research in 2017. The fastai authors, inspired by
the success of transfer learning for image models, developed [ULMFiT](https://arxiv.org/abs/1801.06146) (Universal Language,
Model Fine-Tuning) a training strategy for transfer learning for language models. The lesson involved using the AWD-LSTM
trained on curated data from Wikipedia ([WikiText-103](https://arxiv.org/abs/1609.07843), 103 million tokens), and 
fine-tuning it with a curated set of [50k IMDb (Internet Movie Database) movie reviews](https://aclanthology.org/P11-1015.pdf).
The goal was to classify the sentiment of IMDb reviews, akin to classifying an image as a dog, cat, etc. The idea was to 
see how fine-tuning with the IMDb reviews improved the model's ability to then classify sentiment, because the vocabulary 
would be augmented with IMDb-specific words and contexts.

By 2022, the transformer architecture was outperforming LSTMs. The authors responded by [building a new NLP lesson](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) 
using the [huggingface transformer library](https://huggingface.co/docs/transformers/en/index), with the data and objectives 
from a [US Patent kaggle competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/) (active 
Mar-June 2022). The goal was to compute similarity scores for phrases found in US patents to be able to find existing patents 
that match incoming patent applications. In this lesson, the authors chose Microsoft's [deberta-v3-small](https://arxiv.org/abs/2111.09543) (2021).
From [the docs](https://huggingface.co/microsoft/deberta-v3-small), it has: "6 layers and a hidden size of 768. It has 
44M backbone parameters with a vocabulary containing 128K tokens".

## Preparing datasets for NLP
Preparing the text dataset, whether for an LSTM model or a transformer, is largely the same. We have to have a strategy
for turning text into numbers, since all deep learning models transform tensors of numbers. It will involve `tokenization`, 
`numericalization`, and `embedding`.
- **tokenization** is the process of splitting up the text into useful chunks, like words or word-parts, punctuation, starts/ends of sentences, called tokens; the list of unique tokens makes up the "vocabulary" for a given model; words or word-parts that are not yet part of the vocabulary are all mapped to a special token
- **numericalization** is the process of mapping each token in the vocabulary to a number, since our model will have to process numbers
- **embedding** is the process initializing a matrix where each row is a vector representation for each token without ordinal relationships (e.g. we don't want the fact that dog:458 > cat:457 to mean that dog > cat)

Embedding research has its own historical trajectory; originally embedding vectors were trained separately (e.g. 
[origin of embeddings for NLP](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 2003; [Word2Vec](https://arxiv.org/abs/1310.4546), 
2013), but it turned out researchers got great results by initializing embedding vectors randomly and training them as 
part of the first layer of the model. 

We then begin the process of training our model, using the embedding matrix in the first layer as a trainable parameter, 
like `w` in our image model from the last lesson. This embedding matrix gets updated with backpropagation at the end of 
a training loop. For the image classification task:
- the last layer of the model returned a vector whose length was the number of classes the image could be sorted into, and
- the value of each element was the probability that the image belonged to that particular class

The same is true for the last layer for a next word predictor model:
- the last layer of the model should return a vector whose length is the number of tokens in the vocabulary, and
- the value of each element should be the probability that it is the next word

But we're actually going to build a classification model, so:
- the last layer of the model should return a vector whose length is the number of classes the input text could be sorted into, and
- the value of each element should be the probability that the intput text belongs to that particular class

In the RNN AWD-LSTM model case, the class was a sentiment for the IMDb movie review:
- positive movie review
- negative movie review

For the transformer model case, the class was a similarity score for phrases (e.g. "TV set" and "Television set") found in the patent database:
- 1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
- 0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- 0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- 0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- 0.0 - Unrelated.

Even though transformer model classes are numbers, they look a lot like the MNIST handwritten digit classes we predicted 
in the last lesson.

### One last comparison between text models and image models before we get started
For image modeling we care about understanding similar features between different images, e.g. whether a dog appears in 
the top right corner or the bottom left corner, it is still a dog. This can be handled by processing `convolutions` (sliding
windows), over small patches of each image. We didn't implement the sliding window part in the model-from-scratch lesson, 
but is baked into ResNet.

Text has different kinds of "spatial" relationships: the order of the words and the sequence of the sentences matter. Because 
these relationships are more "before/after" and "earlier/later" than "up/down" and "left/right", they're considered "temporal" 
or "sequential" or "ordered". To train a model for next word prediction, instead of e.g. inputting a list of sentence 
fragments (e.g. a sentence fragment would correspond to an image to be classified) and return the most probable next word 
from the vocabulary (the "label" for the image classification task), we send the document's words as token (chopped up 
into sequential blocks) through the model. Unlike e.g. a bunch of images of dogs found with an internet search, the sentences 
in a document all depend on each other. When LSTMs send tokens through sequentially, the model immediately compares its 
prediction with the actual next token, and this error is backpropagated. It's like guessing cloud and then baking in "yes
that's a cloud" into the parameters.

This was the tension for NLP models: needing to "remember" sequential relationships between words, including grammars that 
control words that are not next to each other and pronouns whose meanings carry between sentences, even ones way at the 
beginning of the document, without forgetting long-range context or overfitting. Moreover, LSTM models were slow because 
sequential relationships are difficult to parallelize over GPUs! These sequence challenges don't exist for image classification 
(though they would if you were using snapshots from a video recording...). But these NLP model challenges are what motivated 
researchers to develop the transformer architecture. The LSTMs used a recurrent hidden state to encode long-distance
relationships to carry forward information from earlier tokens when processing one word of a document at a time. The transformer 
model encodes long-distance relationships by including layers whose learned parameters compute, in a given context window, 
how much attention a word should pay to other words (called "self-attention"). Perhaps obvious to note, this "self-attention"
approach means the transformer model is doing mannny manyyyy pairwise token calculations, but luckily these can be parallelized.

### Just kidding, one more image/text comparison: benchmarks
For the image classification, there were a few benchmark datasets that were developed (e.g. MNIST and ImageNet 
that we discussed in the last post) and used to evaluate model performance. What did NLP use?

NLP models were historically built to accomplish particular tasks: next word prediction, sentiment classification, machine 
translation, "named-entity recognition" (NER), part of speech tagging, question answering, inference, semantic similarity.
Each of these tasks is associated with particular curated benchmark datasets to evaluate model performance. For next-token
prediction, the typical metric is "perplexity", or how surprised the model is by the correct next token in e.g. the 
[Penn Treebank](https://huggingface.co/datasets/FALcon6/ptb_text_only) - a million words of 1989 Wall Street Journal material, 
[WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2) - a hundred million tokens from good/featured articles 
on wikipedia, and [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) - 103 million tokens datasets.

In 2018, just before transformer models were being released, researchers developed a benchmark called General Language 
Understanding Evaluation ([GLUE](https://arxiv.org/abs/1804.07461) so models could be evaluated on performance over all 
these independent tasks. Soon after, when BERT smashed all the performance records, so researchers developed 
[SuperGLUE](https://arxiv.org/abs/1905.00537) to evaluate model performance on more difficult tasks.

## NLP: fine-tuning a transformer model: Patent phrase similarities
```
// And... is where the code is going
```
