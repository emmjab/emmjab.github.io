---
layout: post
title: US patent phrase classifying in Lesson 4
date: 2026-07-08 21:53 -0400
---

# Intro to Transformers via a US Patent Phrase Similarity Scoring

**This post is part of my Practical Deep Learning journey.**

This is my experience with [Lesson 4](https://course.fast.ai/Lessons/lesson4.html) of the 
Practical Deep Learning course. If you're just tuning in, I'm working through this material 
in a study group while doing a half-batch at [The Recurse Center](https://www.recurse.com/) 
in May/June 2026. See [Learning Practical Deep Learning (fastai)](https://emmjab.github.io/2026/06/02/hello-deep-learning.html) for why.

Lesson 4 is the Natural Langauge Processing (NLP) lesson. Here we move from vision models (which learn features of images, 
e.g. for image label prediction) to NLP models (which learn features of text, e.g. for next word prediction). 
Many novel applied computer vision tasks no longer require models trained from scratch. Instead, researchers can start 
from freely available models that were pretrained using huge datasets and lots of compute, and fine-tune them with datasets
specific to the novel task. This is called "transfer learning"; I practiced it in Lesson 2 when I started from the 2015 
ResNet trained on ImageNet (available in the pytorch library), and fine-tuned it on [different types of clouds](https://emmjab.github.io/2026/06/03/cloud-classifying.html))
downloaded from duck duck go.

Text can models take much longer to run than image models (more on this later) -- so wouldn't it make sense to try transfer 
learning for text? "Why yes, that would be a great idea!" say the authors of the course in 2018, as Jeremy Howard and Sebastian 
Ruder proceeded to create what would have been a state-of-the-art transfer learning method to use on Recurrent Neural Network 
architectures like Long Short-Term Memory (LSTM). But competition was steep! In 2017, the `transformer` architecture hit 
the scene with its catchy academic title, ["Attention is all you Need"](https://arxiv.org/abs/1706.03762). Implementations 
swiftly followed in 2018: OpenAI's [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) 
(June) and Google's [BERT](https://arxiv.org/abs/1810.04805) (October).

This lesson is a great example of the authors sticking to their goal of showing students deep learning methods are still 
in active development. The 2020 version of the NLP lesson ([fastbook chapter 10](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)) 
was built on the state-of-the-art RNN at the time, a [Long Short-Term Memory (LSTM) model](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf) 
called [AWD-LSTM](https://arxiv.org/abs/1708.02182) released by Salesforce Research in 2017. The fastai authors had been
inspired by the success of transfer learning for image models, and developed a training strategy called [ULMFiT](https://arxiv.org/abs/1801.06146) 
(Universal Language, Model Fine-Tuning) for transfer learning for language models. The lesson used ULMFiT with the AWD-LSTM
model trained on a Wikipedia dataset ([WikiText-103](https://arxiv.org/abs/1609.07843), 103 million tokens), and involved
fine-tuning with [50k IMDb (Internet Movie Database) movie reviews](https://aclanthology.org/P11-1015.pdf). The goal was 
to classify the sentiment of a subset of IMDb reviews, like we had classified images as dogs, cats, etc. in a prior lesson.
Fine-tuning with the IMDb reviews improved the model's ability to classify sentiment, because the Wikipedia vocabulary 
would be augmented with IMDb-specific words and contexts.

By 2022, the transformer architecture was outperforming ULMFiT-assisted AWD-LSTMs. The authors responded by 
[building a new NLP lesson](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners) 
using the [huggingface transformer library](https://huggingface.co/docs/transformers/en/index). The lesson's task was to 
try to score well on a [US Patent kaggle competition](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/) 
(active Mar-June 2022) using a transformer model. The competition task was to compute similarity scores for pairs of phrases 
found in US patents. Why? Patent officers need to deny patent applications if it turns out the invention is already patented, 
but often people use different phrases to describe what is actually the same invention. Ideally, they could use the model
to find patents similar to a new application submission, even if the exact phrase does not match (e.g. "TV set" 
vs "Television set"). The authors chose Microsoft's [deberta-v3-small](https://arxiv.org/abs/2111.09543) (2021). From 
[the docs](https://huggingface.co/microsoft/deberta-v3-small), it has: "6 layers and a hidden size of 768," and "44M backbone 
parameters with a vocabulary containing 128K tokens". One of the lessons discusses how to choose a model, but I forgot which one.

## Preparing datasets for NLP
The method for preparing a text dataset is largely the same for an LSTM model or a transformer. We need a strategy for 
turning text into numbers, since all deep learning models transform tensors of numbers. It will involve `tokenization`, 
`numericalization`, and `embedding`.
- **tokenization** is the process of splitting up the text into useful chunks, like words or word-parts, punctuation, starts/ends of sentences, called tokens; the list of unique tokens makes up the "vocabulary" for a given model; words or word-parts that are not yet part of the vocabulary are all mapped to a special token
- **numericalization** is the process of mapping each token in the vocabulary to a number, since our model will have to process numbers
- **embedding** is the process initializing a matrix where each row is a vector representation for each token without ordinal relationships (e.g. we don't want the fact that dog:458 > cat:457 to mean that dog > cat)

Embedding research has its own historical trajectory. Originally embedding vectors were trained separately (e.g. 
[origin of embeddings for NLP](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf), 2003; [Word2Vec](https://arxiv.org/abs/1310.4546), 
2013). But, it turned out researchers got great results by initializing embedding vectors randomly and training them as 
part of the first layer of the model, like `w` in the image model from the last lesson. This embedding matrix gets updated 
with backpropagation at the end of each training loop.

So the embedding matrix is a parameter in the first layer of the model. What about the last layer of the model?

For the image classification task:
- the last layer of the model returned a vector whose length was the number of classes the image could be sorted into, and
- the value of each element was the probability that the image belonged to that particular class

A chunk of text could be sorted into classes as well, e.g. "type of document" or "sentiment". But we can also see a next 
word prediction task as a classification problem. In this case, the next word is the most likely selection (class) from a vector 
of the vocabulary:
- the last layer of the model could return a vector whose length is the number of tokens in the vocabulary, and
- the value of each element should be the probability that it is the next word

Besides making the last layer perform "classification", we could also make it do "regression". More on this in a sec. 
In this case:
- the last layer of the model will return just one number

## Choosing classification or regression for the final layer of the NLP model
In the original notebook, the task was to use the RNN AWD-LSTM model to predict sentiment of an IMDb movie review:
- positive movie review
- negative movie review

In the updated lesson, the task was to use the transformer model to predict a similarity score for phrases (e.g. "TV set" 
and "Television set") found in US patents, given particular categories:
- 1.0 - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. “the”, “and”, “or”).
- 0.75 - Close synonym, e.g. “mobile phone” vs. “cellphone”. This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- 0.5 - Synonyms which don’t have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- 0.25 - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- 0.0 - Unrelated.

Bot of these tasks could be either regression or classification tasks. Even though transformer model classes are numbers, 
they're not continuous -- they look a lot like the MNIST handwritten digit classes we predicted in the last lesson. And 
even though the sentiments are words ("pos" and "neg"), we could also see them as values between 0 and 1, pinned to 0 if 
<0.5 and 1 if >0.5.

For the transformer lesson, the authors chose to set num_labels = 1 for the model. Just one label means they chose the 
regression. I think I missed the explanation, but it's probably because they want the model to learn the ordinal relationships
from 1 to 5. Smaller and larger similarity scores mean something -- better and worse similarities; they're not just 
re-orderable categories.

## NLP: fine-tuning a transformer model: Patent phrase similarities
So, what does this all look like in code? WHat kind of stuff does the transformer library come with to turn datasets into 
something the transformer model can use, and then how do we train the model?

The training data is a csv where each row has the following columns:
- id
- anchor phrase: the first phrase to compare
- target phrase: the second phrase to compare
- category code: the category by which the scores will be meaningful
- score: a float of value 0, .25, .5, .75, 1 that corresponds to the phrases' similarity given the category

The basic steps are the following:
1. Prepare the dataset
   1. create a `training dataset object` from the training df with one string of text per row (concat the phrase & category cols together with separator chars)
   1. create a `tokenizer object` from your choice of pre-trained model, and run it on the training dataset column that has the concatenated string
   1. rename the training dataset's 'score' column to 'labels' because that's what transformers expect the labels column to be named
   1. partition the training dataset object into `train` and `test` (test is validation)
   1. create the `model object` for the task at hand; args include the pretrained model's name (e.g. `'deberta-v3-small'`) and num_labels (1=regression; >1=classification)
   1. set up the `training args object`; args include: learning rate (and learning rate sched args), batch sizes for train and eval, number of epochs, machine spec args, and more
   1. write a function to `compute metrics` to print for each epoch; patent task asks for pearson coefficient (`from scipy.stats import pearsonr`)
   1. set up the `trainer object`; args include: `training args object`, `model`, `training dataset object's 'train'` (train), `training dataset object's 'test'` (validation), `tokenizer object`, `compute metrics func`
1. Train the model
   1. run the `trainer object`'s `trainer.train()` function, which updates the `model object` and the `tokenizer object`
   1. on my macbook pro m4 with 24gb ram, this runs for 1 hour; after each epoch, it prints the train loss, validation loss, and metrics -> these numbers tell us if our model is training properly
   1. optional: (becomes necessary if you want to submit a competition notebook on kaggle): save the trained model & tokz: `trainer.save_model("./patent_model_dir")` and then zip it up for upload
2. Generate predictions (scores) for the test.csv data to submit to the kaggle competition
   1. create the `test dataset object` the same way we made the training dataset object in "prepare the dataset" step 1 above, but using the test df (note: this is not the validation dataset, which was just a split of the training data) create a `training dataset object` 
   1. use the same `tokenizer object` from "prepare the dataset" step 2 above, and run it on the test dataset column that has the concatenated string
   1. generate the predictions with the `trainer object` that we trained before: `preds = trainer.predict(test_ds).predictions.astype(float)`
   1. check your predictions -- we had some <0 and >1 values but the submission expects `preds = np.clip(preds, 0, 1)`
   1. create the `submissions.csv` file with the ids and scores (`preds`) columns

Here's the [jupyter notebook]({% link patent-nlp-notebook.md %}) where I did all of this. It also involved a lot of
figuring out values that worked on my macbook pro instead of a GPU.

## Submitting the similarity scores to Kaggle
So... submitting the similarity score csv to Kaggle was actually pretty confusing at first. I thought I could just upload 
the output `submissions.csv` file, but this competition makes you upload a kaggle notebook that generates the file. After 
googling around a bunch and having a meandering conversation with ChatGPT, I managed to figure it out and submit successfully 
on my first try.

Basically the steps in mid-July 2026 are the following.

### Creating the Submission Notebook and getting the competition data and your trained model in there
1. make sure you're logged into your kaggle account
1. create a notebook from the kaggle competition page, so that the competition datasets are accessible to the notebook
1. upload your trained model to the [Your Work](https://www.kaggle.com/work) page on Kaggle by +Create-ing a new model
1. when the model shows up on the Your Work page, click the three dots and "add to collection" and choose the the contest
1. go back to the notebook you just created, and click "add input" on the right; look for your model by filtering by "Your Work"
1. now that you have the competition dataset and the model, click on the little "copy path" buttons next to the competition `test.csv` and the model folder and paste each into variables in the notebook

### What you need to put in the submission notebook
1. make sure you write the imports you need for the code you're about to write/run
1. create the tokenizer from the model path, and copy-paste in the tokenization function you wrote to initially train your model
1. create the same `test_ds` from the test.csv in the competition dataset, using the tokenizer you just imported
1. create the model from the model path
1. set up the trainer from the model and the tokenizer
1. run predict with the trainer! `preds = trainer.predict(test_ds).predictions.astype(float).reshape(-1)`
1. put the predictions where they belong: `submission = df_test[["id"]].copy(); submission["score"] = preds`
1. write the `submissions.csv` file that Kaggle expects to see in the output `submission.to_csv("/kaggle/working/submission.csv", index=False)`

Then, click the clear and run all cells button to make sure the output file gets created. You can download that file to 
make sure it looks like what the submission expects/matches the output you got when you ran locally.

Last, submit to the competition! There's a section literally called that in the section on the right. Clicking this might
tell you you'll need to "turn off the internet" for the notebook. That's in `settings` at the top of the page. I made sure
the notebook still ran after I did this, before I clicked submit. I was a little unsure about whether the libraries I was 
using were already installed, but most deep learning libraries are and the ones I used were there without doing anything
special. The notebook runs in a docker image; you can choose what kind of "accelerators" (read: GPUs) you want to use. I 
didn't use any for this task; and you can also pick from "environment preferences": pinned to original, or latest. I haven't
explored most of these.

Anyway... voila! The submission runs, and you get a "public" and a "private" score -- at least for the already-over competitions. 
You can select your best attempts to submit to the private leaderboard if the competition is still active, I think.

Here's [my kaggle notebook](https://www.kaggle.com/code/emmjab/patent-phrase-similarity-lesson-4) where I did all of this.

## End Notes
### Why Transformers? (image models vs text models)
For image modeling we care about recognizing features across images, even if they're not always in the same spot. For instance,
whether a dog appears in the top right corner or the bottom left, it is still a dog. This can be handled by processing 
`convolutions` (sliding windows), over small patches of each image. We didn't implement the sliding window part in the 
model-from-scratch lesson, but is baked into ResNet.

Text has different kinds of "spatial" relationships: the order of the words and the sequence of the sentences matter. Because 
these relationships are more "before/after" and "earlier/later" than "up/down" and "left/right", they're considered "temporal" 
or "sequential" or "ordered". To train a model for next word prediction, instead of e.g. inputting a randomly ordered list 
of sentence fragments from our dataset (if sentence fragment ~ image), we send ordered chunks of the document through 
the model, word by word. Unlike dirs full of dog pics and cat pics, the sentences in a document and the words in a sentence 
depend on each other. When LSTMs send tokens through sequentially, the model immediately compares its prediction with the 
actual next token, and this error is backpropagated. It's like guessing cloud and then baking in "yes that's a cloud" into 
the parameters.

This was the tension for NLP models: needing to "remember" sequential relationships between words, including grammars that 
control words quite far from each other, and pronouns whose meanings carry between sentences, without forgetting long-range 
context or overfitting. Moreover, LSTM models were slow because sequential relationships are difficult to parallelize over 
GPUs!

These sequence challenges don't exist for image classification (though they would if you were using snapshots from a video 
recording...). They are what motivated researchers to develop the transformer architecture. The LSTM model uses a recurrent 
hidden state to encode long-distance relationships so information can be carried forward from earlier tokens, since words
are processed one at a time through the document. Instead of a recurrent hidden state, the transformer model includes layers
whose parameters encode long-distance relationships by calculating how much attention a word should pay to many other words 
(e.g. every previous word) in a given context window. That "attention" to other words is called "self-attention" (I guess
because it's assigned to itself?). Perhaps obvious to note, this "self-attention" approach means the transformer model is doing
mannny manyyyy pairwise token calculations, but luckily these can be parallelized.

### Benchmark datasets and metrics for NLP
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