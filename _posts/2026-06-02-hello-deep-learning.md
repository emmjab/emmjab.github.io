---
layout: post
title: "Learning Practical Deep Learning (fastai)"
date: 2026-06-02
---

# Practically learning something deep

I'm working through the ["fastai" Practical Deep Learning for Coders course](https://course.fast.ai/)
with a study group while doing a half-batch at [The Recurse Center](https://www.recurse.com/) in May/June 2026.
[Lesson 2](https://course.fast.ai/Lessons/lesson2.html) suggested blogging the learning process.
That's my cue to set up github pages.

## What is Practical Deep Learning for Coders?

Practical Deep Learning for Coders is a free online course built by [Jeremy Howard](https://en.wikipedia.org/wiki/Jeremy_Howard_(entrepreneur))
and [Rachel Thomas](https://en.wikipedia.org/wiki/Rachel_Thomas_(academic)) "with the goal of democratizing deep learning".
It was first run in 2016 as a certificate course at the University of San Francisco, and then became a MOOC. The initial
course explored the Keras/Theano/TensorFlow libraries, but soon dropped them in favor of pytorch.

When they created the course, democratizing deep learning [meant](https://www.fast.ai/posts/2016-10-07-fastai-launch.html):
- fix the shortage of data scientists with deep learning expertise
- create highly automated tools for training deep learning models
- build software to provide deep insight into the training of, and results from, deep learning models
- develop a range of “role model” applications, in areas where deep learning is currently being poorly utilized.

In 2018, the authors published a wrapper/utility library around pytorch called "fastai" which handles 
various data pre-processing among other things. The course materials teach the principles of deep learning 
through this library and pytorch. They also peel back the layers of the libraries to build deep learning models from scratch.

In 2019, the University of San Francisco launched the Center for Applied Data Ethics (CADE) - rebranded in 2023 as 
[Center for AI & Data Ethics (CAIDE)](https://www.usfca.edu/data-institute/centers-initiatives/caide) initially directed by Rachel Thomas.

In 2020, the authors published a [paper on the fastai library](https://www.mdpi.com/2078-2489/11/2/108) 
and released the fastai book,
[*Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD*](https://course.fast.ai/Resources/book.html).

In 2022, the authors responded to transformers hitting the scene by incorporating them into the [2022 lectures](https://www.youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU)
and [course materials - course22](https://github.com/fastai/course22), recorded by Jeremy Howard at the 
University of Queensland in Brisbane, Australia. See [lesson 4 on NLP](https://course.fast.ai/Lessons/lesson4.html) 
with a link to the [patent data kaggle notebook with an intro to hugging face](https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners).

In 2023-2024, the authors responded to Stable Diffusion by making a [Part 2](https://course.fast.ai/Lessons/part2.html) to the course
with [lecture recordings](https://www.youtube.com/playlist?list=PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP), which goes through building Stable Diffusion from scratch.

It's 2026 and fastai has had [many new blog posts](https://www.fast.ai/) but no new commits to fastbook or the course materials in a bit.

## What material is covered?

The lessons are available at <https://course.fast.ai/>.

Each lesson link includes:
- a 1.5h video lecture recorded by Jeremy Howard
  - Part 1, general intro; lectures posted Jul 2022, recorded at the University of Queensland in Brisbane, Australia ([youtube](https://www.youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU)).
  - Part 2, stable diffusion; lectures posted Oct 2022 & Apr 2023 ([youtube](https://www.youtube.com/playlist?list=PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP)).
- chapter(s) to read/run from one or both course material repos (each a jupyter notebook)
  - "fastbook" - course textbook, 2020: <https://github.com/fastai/fastbook>
  - "course22" - course lecture companion, 2022: <https://github.com/fastai/course22>
- supplementary materials
  - links to kaggle competitions from the lecture
  - links to hugging face repos from the lecture
  - other random relevant links

### Table of Contents for the lessons at <https://course.fast.ai/>
#### Part 1 (lessons 1-8)
- 1: [Getting started](https://course.fast.ai/Lessons/lesson1.html)
- 2: [Deployment](https://course.fast.ai/Lessons/lesson2.html)
- 3: [Neural net foundations](https://course.fast.ai/Lessons/lesson3.html)
- 4: [Natural Language (NLP)](https://course.fast.ai/Lessons/lesson4.html)
- 5: [From-scratch model](https://course.fast.ai/Lessons/lesson5.html)
- 6: [Random forests](https://course.fast.ai/Lessons/lesson6.html)
- 7: [Collaborative filtering](https://course.fast.ai/Lessons/lesson7.html)
- 8: [Convolutions (CNNs)](https://course.fast.ai/Lessons/lesson8.html)
- Bonus: [Data ethics](https://course.fast.ai/Lessons/lesson8a.html)
- Summaries (bulleted list of notes for each lesson)

#### Part 2 (lessons 9-25)
- Part 2 [overview](https://course.fast.ai/Lessons/part2.html)
- 9: [Stable Diffusion](https://course.fast.ai/Lessons/lesson9.html)
- 10: [Diving Deeper](https://course.fast.ai/Lessons/lesson10.html)
- 11: [Matrix multiplication](https://course.fast.ai/Lessons/lesson11.html)
- 12: [Mean shift clustering](https://course.fast.ai/Lessons/lesson12.html)
- 13: [Backpropagation & MLP](https://course.fast.ai/Lessons/lesson13.html)
- 14: [Backpropagation](https://course.fast.ai/Lessons/lesson14.html)
- 15: [Autoencoders](https://course.fast.ai/Lessons/lesson15.html)
- 16: [The Learner framework](https://course.fast.ai/Lessons/lesson16.html)
- 17: [Initialization/normalization](https://course.fast.ai/Lessons/lesson17.html)
- 18: [Accelerated SGD & ResNets](https://course.fast.ai/Lessons/lesson18.html)
- 19: [DDPM and Dropout](https://course.fast.ai/Lessons/lesson19.html)
- 20: [Mixed Precision](https://course.fast.ai/Lessons/lesson20.html)
- 21: [DDIM](https://course.fast.ai/Lessons/lesson21.html)
- 22: [Karras et al (2022)](https://course.fast.ai/Lessons/lesson22.html)
- 23: [Super-resolution](https://course.fast.ai/Lessons/lesson23.html)
- 24: [Attention & transformers](https://course.fast.ai/Lessons/lesson24.html)
- 25: [Latent diffusion](https://course.fast.ai/Lessons/lesson25.html)
- Bonus: Lesson 9a (youtube video)
- Bonus: Lesson 9b (youtube video)

### Repos for the course materials
#### Course22 - part 1: these are the notebooks for each of the 2022 lectures linked on each lesson of fast.ai:
- Lesson 1:
  - fastbook ch1: <https://github.com/fastai/fastbook/blob/master/01_intro.ipynb>
  - <https://github.com/fastai/course22/blob/master/00-is-it-a-bird-creating-a-model-from-your-own-data.ipynb>
  - <https://github.com/fastai/course22/blob/master/01-jupyter-notebook-101.ipynb>
- Lesson 2:
  - fastbook ch2: <https://github.com/fastai/fastbook/blob/master/02_production.ipynb>
  - <https://github.com/fastai/course22/blob/master/02-saving-a-basic-fastai-model.ipynb>
- Lesson 3:
  - fastbook ch4: <https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb>
  - <https://github.com/fastai/course22/blob/master/03-which-image-models-are-best.ipynb>
  - <https://github.com/fastai/course22/blob/master/04-how-does-a-neural-net-really-work.ipynb>
  - titanic spreadsheet: <https://github.com/fastai/course22/blob/master/xl/titanic.xlsx>
- Lesson 4:
  - fastbook ch10: <https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb>
  - <https://www.kaggle.com/code/jhoward/getting-started-with-nlp-for-absolute-beginners>
- Lesson 5:
  - fastbook ch4: <https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb>
  - fastbook ch9: <https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb>
  - <https://github.com/fastai/course22/blob/master/05-linear-model-and-neural-net-from-scratch.ipynb>
  - <https://github.com/fastai/course22/blob/master/06-why-you-should-use-a-framework.ipynb>
  - <https://github.com/fastai/course22/blob/master/07-how-random-forests-really-work.ipynb>
- Lesson 6:
  - fastbook ch9: <https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb>
  - <https://github.com/fastai/course22/blob/master/07-how-random-forests-really-work.ipynb>
  - <https://github.com/fastai/course22/blob/master/08-first-steps-road-to-the-top-part-1.ipynb>
- Lesson 7:
  - fastbook ch8: <https://github.com/fastai/fastbook/blob/master/08_collab.ipynb>
  - <https://github.com/fastai/course22/blob/master/09-small-models-road-to-the-top-part-2.ipynb>
  - <https://github.com/fastai/course22/blob/master/10-scaling-up-road-to-the-top-part-3.ipynb>
- Lesson 8:
  - fastbook ch13: <https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb>
  - <https://github.com/fastai/course22/blob/master/xl/collab_filter.xlsx>
  - <https://github.com/fastai/course22/blob/master/xl/conv-example.xlsx>
- Bonus: Data Ethics
  - fastbook ch3: <https://github.com/fastai/fastbook/blob/master/03_ethics.ipynb>

#### Course22 - Part 2: notebook
- Notebook repo: <https://github.com/fastai/course22p2>
- Link directly to the lesson notebooks: <https://github.com/fastai/course22p2/tree/master/nbs>

#### Fastbook ("the book"): The Practical Deep Learning book chapters that accompany this course, called "Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD"
- Chapter 1, [Intro](https://github.com/fastai/fastbook/blob/master/01_intro.ipynb)
- Chapter 2, [Production](https://github.com/fastai/fastbook/blob/master/02_production.ipynb)
- Chapter 3, [Ethics](https://github.com/fastai/fastbook/blob/master/03_ethics.ipynb)
- Chapter 4, [MNIST Basics](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb)
- Chapter 5, [Pet Breeds](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)
- Chapter 6, [Multi-Category](https://github.com/fastai/fastbook/blob/master/06_multicat.ipynb)
- Chapter 7, [Sizing and TTA](https://github.com/fastai/fastbook/blob/master/07_sizing_and_tta.ipynb)
- Chapter 8, [Collab](https://github.com/fastai/fastbook/blob/master/08_collab.ipynb)
- Chapter 9, [Tabular](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb)
- Chapter 10, [NLP](https://github.com/fastai/fastbook/blob/master/10_nlp.ipynb)
- Chapter 11, [Mid-Level API](https://github.com/fastai/fastbook/blob/master/11_midlevel_data.ipynb)
- Chapter 12, [NLP Deep-Dive](https://github.com/fastai/fastbook/blob/master/12_nlp_dive.ipynb)
- Chapter 13, [Convolutions](https://github.com/fastai/fastbook/blob/master/13_convolutions.ipynb)
- Chapter 14, [Resnet](https://github.com/fastai/fastbook/blob/master/14_resnet.ipynb)
- Chapter 15, [Arch Details](https://github.com/fastai/fastbook/blob/master/15_arch_details.ipynb)
- Chapter 16, [Optimizers and Callbacks](https://github.com/fastai/fastbook/blob/master/16_accel_sgd.ipynb)
- Chapter 17, [Foundations](https://github.com/fastai/fastbook/blob/master/17_foundations.ipynb)
- Chapter 18, [GradCAM](https://github.com/fastai/fastbook/blob/master/18_CAM.ipynb)
- Chapter 19, [Learner](https://github.com/fastai/fastbook/blob/master/19_learner.ipynb)
- Chapter 20, [Conclusion](https://github.com/fastai/fastbook/blob/master/20_conclusion.ipynb)


## Why am I doing this course?

There seem to be two paths for a researcher:
- choose a particular field and spend years becoming a domain expert
- choose to become adept in using and developing techniques for processing data, and be domain agnostic

On the path:
- domain-oriented people end up learning data processing techniques used in their fields
- technique-oriented people end up learning about the data in domains in which they embed

What does that have to do with choosing this course? I'm a domain-oriented person too curious to sit still in just one domain.
This has led me to cultivate a techniques orientation so that I can move around and satisfy my curiosity.
The Practical Deep Learning course promises to teach techniques that can be applied by including datasets 
from different domains to process with each lesson. The intended audience is domain experts wanting to try 
out deep learning for their domains, and data techniques people collaborating with domain experts. 
I fall mostly into the latter category.

By following this course, I hope to gain some understanding of the kinds of problems people expect to be able 
to solve with deep learning, and intuition about the limitations of these kinds of models.
Are there domains of research that have been swiftly adopting deep learning to the detriment of other
(slower?) approaches to knowledge production that would aim at answering different but also good questions?

Moreover, the full title of the book is 
"Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD".
I am a coder, I want to learn how to apply AI, and don't have a PhD.

## Why *this* course out of all the options in 2026?

There are mannnny deep learning and other machine learning courses and materials to follow today.
Why did I choose this one?
- something I knew - heard about it and started going through the material sometime between 2016 - 2018; gave up after a few lessons because I got conceptually stuck on the ubiquitous graph of layers of nodes and fully connected edges, plus I didn't have a dataset
- saw that the material was updated over the years so it effectively fills the gap that I have for ML/AI
- someone happened to organize a group to go through this material in particular
- had to start somewhere -- will add to this as I go; I've already started supplementing my reading [with other things](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
