---
layout: post
title: auction sale price predicting in Lessons 5&6
date: 2026-07-13 22:28 -0400
---

# Intro to Random Forests with Tabular Data on Bulldozer Sales

**This post is part of my Practical Deep Learning journey.**

This is my experience with [Lesson 5](https://course.fast.ai/Lessons/lesson5.html) and 
[Lesson 6](https://course.fast.ai/Lessons/lesson6.html) of the Practical Deep Learning 
course. If you're just tuning in, I'm working through this material in a study group while 
doing a half-batch at [The Recurse Center](https://www.recurse.com/) in May/June 2026. 
See [Learning Practical Deep Learning (fastai)](https://emmjab.github.io/2026/06/02/hello-deep-learning.html) for why.

In this lesson we're exploring tabular data. One of the points of this lesson is to show that deep learning is not the 
best machine learning method for all kinds of problems. For instance, for tabular data, decision trees often perform
pretty well, and you can augment their performance by bagging them into "random forests". Moreover, decision trees are 
useful because you can visualize them to see which columns are contributing to the performance of the model.

In the first few lessons of the course, we learned how to: 
- prepare labeled image and unlabeled text datasets
- choose deep learning models architectures and training loss functions, and
- apply metrics to classification or regression tasks to evaluate the performance of the model.

In this lesson we'll:
- prepare tabular datasets by labeling "categorical" and "continuous" variables, among other things
- visualize decision trees and make random forests
- explore approaches to investigating the contribution of columns to the prediction
- compare the random forest approach with a neural net
- discuss methods for improving tabular models, e.g. "bagging" (averaging predictions) and "boosting" (summing predictions)

In these lessons I decided to do the course using my mac laptop instead of cloud GPUs. So I've also been learning
something about parameter fiddling to make the data fit on the machine and make the training loops run in reasonable 
amounts of time. Along those lines, because I followed the course advice to run the notebooks with the latest versions of 
the libraries, I also found a few places where object implementations and parameter names drifted. Nothing crazy! Modern
LLMs seem to be pretty good at suggesting up-to-date library compatibilities and letting you know when they're not up to
speed. I was sort of shocked to see a: "I'm sorry I can only make suggestions for best practices through 2025."

Fixing bugs related to library deprecations and transformations is maybe one of the more boring and time-consuming parts 
of running code. But if you don't spend the time doing it, you'll never get anywhere. Fixing things blindly (e.g. copy-pasting 
in some fix without reading what it is) is also not great. You might get your code run, but it may silently behave differently 
than you want. So I sucked it up and did it -- I can only expect that the world will keep being filled up with more and 
more aging code and messy data that supports our swiftly tilting planet. Tasks in the real world are not carefully prepared 
course notebooks.

## Bulldozer data?

In Lessons 5 & 6, we apply deep learning methods to tabular data. The technique the authors use for this type
of data is called "Random Forests", which is a method for getting better performance on decision trees. In these lessons
there's an explosion of jupyter notebooks to check out, but ultimately there are two choices of datasets to use for building,
training, and evaluating the random forest models:
- the [titanic competition](https://www.kaggle.com/competitions/titanic/data): given a list of passengers on the Titanic and a bunch of data on them (gender, age, ticket cost, etc.), predict whether they survived
- the [bulldozers competition](https://www.kaggle.com/competitions/bluebook-for-bulldozers/rules): given a list of industrial machines sold at auctions over a given period of time and a bunch of data on them (64 fields, including MachineID, ModelID, YearMade, SaleElapsed, SalePrice), predict the auction price for future sales of those machines

The Titanic dataset is too gruesome for me, so I went with the bulldozers. The thing is, the bulldozers competition is
from 13-14 years ago (!?). The data is no longer on kaggle, and you can't click the "accept the rules" button which you
usually need to do to get access to the data. With a little googling around, I found that someone else made an ML course 
using the bulldozers competition and linked [the data in their course repo](https://github.com/mrdbourke/zero-to-mastery-ml/blob/master/data/bluebook-for-bulldozers.zip).
When I went through this lesson originally, I ended up using that data and everything went mostly ok (biggest thing was 
trying to sort out the latest dependencies for a cool tree visualization library, [`dtreeviz`](https://github.com/parrt/dtreeviz),
but I figured it out and made some cool-looking graphs!).

Predicting "future" auction prices for bulldozers is interesting because in addition to being tabular like the Titanic 
passenger manifest, the data is also a timeseries. The authors pointed out a few things to keep in mind for timeseries 
data:
- instead of randomly partitioning training data into train and validation sets, the training data should be a "before" chunk, and the validation set should be "after"
- because things change over time, data from many years earlier might be totally irrelevant to contemporary predictions
- or, because inflation, we should expect to predict higher prices for auction sales of the same machine in later years

... Some of these timeseries conditions sound sort of similar to the NLP criteria from the last lesson! In particular, 
the "before" and "after" conditions, and the dependence of the later data points on the earlier ones. There are some
differences though. In the NLP case we wanted to hang on to words from much earlier for inform context - here we're told 
maybe we should drop earlier data because behavioral patterns may have shifted (but... what about cyclic patterns in the 
market??).

## What are Decision Trees

For this lesson, we start by learning about how to train a decision tree from a table of data. Next, we move from a single 
decision tree to a "random forest". Finally, we compare random forest predictions with those from a neural network.

Decision trees and random forests are part of an arsenal of classical (non-neural net) machine learning techniques that 
haven't disappeared from use. The goal of this lesson seems to be showing students that there are other model architectures,
besides variations on the neural network, that you might reach for to solve a task with a given dataset.

So how do we train a decision tree? According to the authors, "The basic steps to train a decision tree can be written down very easily":
- Loop through each column of the dataset in turn.
- For each column, loop through each possible level of that column in turn.
- Try splitting the data into two groups, based on whether they are greater than or less than that value (or if it is a categorical variable, based on whether they are equal to or not equal to that level of that categorical variable).
- Find the average sale price for each of those two groups, and see how close that is to the actual sale price of each of the items of equipment in that group. That is, treat this as a very simple "model" where our predictions are simply the average sale price of the item's group.
- After looping through all of the columns and all the possible levels for each, pick the split point that gave the best predictions using that simple model.
- We now have two different groups for our data, based on this selected split. Treat each of these as separate datasets, and find the best split for each by going back to step 1 for each group.
- Continue this process recursively, until you have reached some stopping criterion for each group—for instance, stop splitting a group further when it has only 20 items in it.

After reading this I wasn't totally sure about what "levels" meant, so I spent some time unpacking the details. I turned 
what I learned into the following example.

Let's say we have a dataset with 100 rows and four columns. The columns are "id" (unique), "age" [1,7], "type" ["bulldozer", "tractor", "wheelbarrow", 'rake'], 
and "price" (dollars). "price" is the dependent variable. To decide how best to partition the "age" column into two groups, 
we would sort from youngest to oldest. Then, we would create a list of pairs of groups by moving a partition systematically
across the values in the column. If age were [1,7], we would make 6 pairs of row-groups:
```
partition A: group 1: 1      | group 2: 234567 
partition B: group 1: 12     | group 2: 34567
partition C: group 1: 123    | group 2: 4567
...
partition F: group 1: 123456 | group 2: 7
```

For a given partition A, we would predict that the sale price for a given row in group 1 is the average sale price for all 
the rows in group 1; similarly the sale price for a given row in group 2 is the average sale price for all the rows in 
group 2. Then we would calculate the RMSE between the row's actual sale price and the prediction, for all the rows across 
both groups of partition A. The partition that produces the lowest RMSE is selected for that particular column.

Then the next column would be considered. For a categorical column like "type", we would assign each category an arbitrary 
number, e.g. bulldozer = 1, tractor = 2, wheelbarrow = 3, rake = 4, and proceed with the same partitioning scheme. Then 
this column's best partition would be compared with the best partition from the other columns. Whichever had the smallest 
sale price RMSE would be chosen as the node's first split.

To build the next node, the same procedure would be repeated with just the rows on the left side of the chosen partition. 
Simultaneously, the same procedure would be repeated on the right side of the chosen partition.

The last step in the decision tree building algorithm is the stopping criteria. If we don't impose a stopping criteria, 
we would end up creating a node for each individual row. The RMSE for the final split into individual rows would be 0, 
because the row's price would be equal to its group's average price (each side of the partition is a group with just one
element)! This would be pretty unhelpful unless you were trying to make a visualization of overfitting.

### Partitioning on categorical values
The lesson doesn't talk about this in detail, but the decision tree's partitioning algorithm that I just described is 
part of the `scikit.learn` library's implementation. It seems a little strange to consider categorical variables ordered; 
e.g. in the example above there would never be a ["bulldozer", "rake"] | ["tractor", "wheelbarrow"] partition, but there's 
no reason this partition couldn't produce the smallest RMSE. Some googling and ChatGPT-conversing unearthed "category boosting" 
libraries; these overcome the `scikit.learn` ordered categorical partitioning and usually produce better results.

What the lesson *does* talk about is how we don't need to make embedding vectors for each of the categorical values. Doing 
the same kind of partitioning of the ordinal values is good enough, meaning that [according to this 2019 paper](https://peerj.com/articles/6339/) 
the ordinal partitioning method apparently finds the same best-split as using embeddings.

## Preparing tabular data for decision trees and random forests
The jupyter notebook for this lesson, [fastbook chapter 9](https://github.com/fastai/fastbook/blob/master/09_tabular.ipynb),
was full of methods for interrogating the model and understanding the predictions by the contributions of each of the columns 
and column values. There were also more than a few places where updates to `scikit-learn`, `pandas`, and some visualization 
libraries meant I had to figure out ways to reproduce the authors' lessons with newer or alternative versions of these libraries.

For this reason, I ended up writing a lot of notes and quotes from the original lesson in my local jupyter notebook where I 
executed the cells. So, that's [here](/notebooks/bulldozers-tabular-notebook.html).

The main things to remember from this lesson are that:
- random forests are a great way to interrogate tabular data
- random forests are a great example of an ensemble methods: ways to improve model predictions by combining the results from multiple models with uncorrelated errors