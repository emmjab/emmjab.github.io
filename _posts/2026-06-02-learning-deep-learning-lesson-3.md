---
layout: post
title: 'digit classifying in Lesson 3'
date: 2026-06-04
---

# Intro to Image Models by Guessing Handwritten Numbers

**This post is part of my Practical Deep Learning journey.**

This is my experience with [Lesson 3](https://course.fast.ai/Lessons/lesson3.html) of the 
Practical Deep Learning course. If you're just tuning in, I'm working through this material 
in a study group while doing a half-batch at [The Recurse Center](https://www.recurse.com/) 
in May/June 2026. See [Learning Practical Deep Learning (fastai)](https://emmjab.github.io/2026/06/02/hello-deep-learning.html) for why.

The lessons for Practical Deep Learning are about learning by doing. *Doing* means writing code to 
create, run, and analyze the results of models. The authors thought this would be an effective way for 
students to build intuition about how models trained on real-world data perform on particular tasks, and how 
results vary when you change qualities of the model and data (architecture, batch sizes, loss function, learning rate, 
metric, methods for handling missing data, etc.).

The first two lessons use wrapper functions from the `fastai` library to abstract away the 
details of creating models like Microsoft Research's 18- and 34-layer models, [ResNet-18 and ResNet-34](https://github.com/kaiminghe/deep-residual-networks)
(published in 2015), which were trained on [ImageNet-1k](https://ieeexplore.ieee.org/document/5206848) (both originally and
[currently available in the pytorch library](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html)).
The Practical Deep Learning authors wanted to show just how good image classification models had become with ResNet's 
breakthrough with "residual" or "skip" connections in 2015, and what kind of classification accuracy you could get by fine-tuning 
general models like ResNet on particular labeled image datasets suited to your own purposes. You can try out my 
cloud classifying model (fine-tuned from ResNet-18 using clouds sourced from duck duck go) [here](https://huggingface.co/spaces/emmjab/dllearning?logs=container), 
see the code [here](https://huggingface.co/spaces/emmjab/dllearning/tree/main), and read a stream of 
consciousness blog post of me getting the model into a hugging face repo and website [here](https://emmjab.github.io/2026/06/03/cloud-classifying.html).

## Handwritten digits - MNIST dataset

For lesson 3, the walkthrough task was to create a binary classification model for images of handwritten 3s and 7s... from 
scratch. That is, instead of starting from the ResNet model and fine-tuning the latest layers to transfer learn parameters 
for a new dataset, we'll start from scratch with a particular dataset and build a few-layers-deep neural network model 
where we learn what kind of choices we have to make when setting up parameters, the model architecture, and the epoch training 
and validation methods.

The data comes from MNIST, a dataset derived from handwritten forms collected by the US National Institute of Standards (NIST) 
to prepare for digitizing the 1990 US Census forms. The original data was a mixed set of characters (A-Z, a-z, 0-9) written 
by Census Bureau field representatives and high school students from one high school in Bethesda, MD. 
From March to May 1992, the First Census OCR Systems Conference held a competition to read these character sets. 
In 1991, AT&T Bell Labs had bought National Cash Register (NCR); OCR bank check reading was just around the corner.
From 1992 through 1998, Yann LeCun and his team at AT&T Bell Labs remixed and reprocessed the digit data into the MNIST dataset.
In 1998, MNIST was released and it became a benchmark for OCR models. Most of the machine learning python libraries include 
it in a `datasets` library.

So, that's the data. The learning objective was to learn how to assemble and make choices about the parts that make up a 
deep learning model, some of which might be obfuscated by machine learning library defaults, e.g. in `fastai` and `pytorch`, 
etc. By showing off what's under the hood, the authors could make sure students don't overfit on library particulars when 
the goal is to generally learn how to apply deep learning, especially when the state-of-the-art methods are still in rapid
development. To focus on the model, the data is provided, sorted into labeled folders and separated into training, validation, 
and test sets.

To ready the images for the model, we had to load each of the `N` images as an `mxn` matrix of pixels, convert the pixels
to grayscale and normalize their values (/255), stack them into a tensor, and then convert the `mxn` matrix of pixels for each
image into a vector of length `m*n`, so we have a resulting stack of images in a tensor of shape `(N, m*n)` -- `N` rows and 
`m*n` columns. The labels are `1` for a 3 and `0` for a 7, stored as a tensor of shape `(N,)` (one value for each row).

After walking through the lesson, it's left to the reader to make modifications so that the model can classify all digits, 
0-9. I ended up explaining how to do both classification tasks, as you'll see if you keep reading. Notably, only three 
things need to change to accommodate more classes:
- the labels (changes from `1` (it's a 3) & `0` (it's a 7) to the actual number represented in the image, `(0-9)`)
- the size of the last layer of the model (and the layer's corresponding parameters) (changes from 1 class to 10 classes)
- the choice of loss function

### Just a few quick lines of code for setting up the training dataset
In the last section I wrote human language for how to turn the folders of images into training data, but let me just do
it in code here because it involves some interesting tensor manipulations.

Here's a way to make training data & label tensors for the 3s and 7s binary classification case. For each digit folder's 
files, if it's a digit we care about (3 or 7), stack its images, then concatenate the two stacks. For the labels, store 
a `1` for a handwritten three, and a `0` for a handwritten seven.
```
x_stacks = []
y_stacks = []
label_dict = {3: 1, 7: 0} # we're labeling 1 if it's a 3, and 0 if it's a 7

for folder in sorted((path/'training').ls(), key=lambda x: int(x.name)):
    num = int(folder.name)
    if num in (3,7):
    
        # images
        tensors = [tensor(Image.open(o)) for o in folder.ls().sorted()]
        x_stacks.append(torch.stack(tensors).float()/255)
        
        # labels
        y_stacks.extend([label_dict[num]] * len(folder.ls()))
        
xs = torch.cat(x_stacks).view(-1, 28*28)    # (12396, 784) - 12396 images, 6131 threes and 6265 sevens each with 784 pixels
ys = tensor(y_stacks)                       # (12396,)
```

Here's a way to make training data & label tensors for the full dataset. For each digit folder's files, stack all images, 
then concatenate the two stacks. For the labels, store a `0` if it's a zero, `1` if it's a one, ... , `9` if it's a nine.
```
x_stacks = []
y_stacks = []

for folder in sorted((path/'training').ls(), key=lambda x: int(x.name)):

    # images
    tensors = [tensor(Image.open(o)) for o in folder.ls().sorted()]
    x_stacks.append(torch.stack(tensors).float()/255)
    
    # labels
    y_stacks.extend([int(folder.name)] * len(folder.ls()))
    
xs = torch.cat(x_stacks).view(-1, 28*28)    # (60000, 784) - 60,000 images, each with 784 pixels
ys = tensor(y_stacks)                       # (60000,)
```

## How do we set up the model?

The toy "deep" neural net has two linear layers with a ReLU ("rectified linear unit") sandwiched between them.
```
def model(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
```
Unpacking the variables, we have:
- `xb` => the stack of a batch of training images as a tensor
- `w1`, `b1`; `w2`, `b2` => vectors of parameters that we initialize randomly; in the training loop they will get "trained" into different values

Unpacking the functions, we have:
- `res = xb@w1 + b1` => this is just `y = mx+b`, the equation for a line; more on the parameters `w1` & `b1` later, but the 1st dimension has to match the number of pixels in one image and the 2nd corresponds to our choice of "activations"
- `res = res.max(tensor(0.0))` => this is the ReLU - any negative numbers are clipped to 0; this is how we get the output of a series of linear transformations to be something else than a different linear function with some other choice of parameters
- `res = res@w2 + b2` => this is just `y = mx+b` again, but the 1st dimension of `w2` has to match the activation number we chose before, and the 2nd and `b2`'s only dim has to match the number of classes we want at the end, e.g. 1 if we're just guessing 3s or 7s, because `res` is now comprised of "logits" which we can convert to a probabilities of belonging to a particular class

Just highlighting what's going on here... we're kind of using matrix math to smoosh all of our pixels into classes. Pretty weird!!!

## How do we train the model?
Training a model basically involves finding fixed values for the model parameters so that when you pass in an image of a 
handwritten number 8, your model will confidently tell you "that's an 8". Finding effective parameters means looping through 
the data and updating the parameters with better values at each the end of each training loop. So what's in a training loop? 
What's the procedure for finding better values? One method is "stochastic gradient descent" (SGD). Pytorch (and fastai 
wrappings) include an implementation of this method in their `Optimizer` module, but in this lesson we're learning what's 
under the hood.

So just to keep in your head as we go through the details, the basic steps for training are, 
after you initialize the parameters:
- send the data through the model
- calculate the mean loss on the output
- calculate the gradient of the loss for each of the parameters
- update the parameters based on the gradient
- (and print the loss so the human can take a look)

### The loss function
At the heart of the training loop is the loss function. We will be happy if we calculate a smaller and smaller loss 
at the end of each training loop, because the loss for each image is smaller the closer we are to 100% certain of our 
prediction of `7` for an image actually labeled `7`.

In each run of the training loop, we execute our model on a batch of training data and parameters, and send the results (the 
tensor of raw output scores ("logits")) into a "loss function" of our choosing. We can write many different loss functions, 
but any loss function will receive logits computed from the input data from the model and labels from the training data. 
It will turn each image's computed scalar or vector of logits into probabilities that an image belongs to each class.
It then calculates the "loss" for each image based on the input tensor of labels. It does this by computing how "far" the 
predicted label was from the correct label. We can't use a function that just says "right!" or "wrong!" for a given labeled 
image because that doesn't help us figure out what direction to modify the model parameters to make better, or at least 
less wrong, predictions in the next loop.

Small loss means that we were confident and correct in our predictions; large loss means we were either not confident in 
our predictions, or confidently wrong.

The loss for a particular batch is typically one number: the mean loss over all the images.

#### Example loss functions
For the case where we're just classifying images as 3 and 7, where 3 is labeled `1` and 7 is labeled `0`. `yb` is a tensor 
that contains the label/target/class for each image. `logits` is the model output -- each image's vector or scalar representing
something that can be turned into a probability or confidence in the predicted label as part of the loss function.
```
def mnist_loss(model_logits, yb):
    probs = model_logits.sigmoid()
    return torch.where(yb==1, 1-probs, probs).mean()
    
loss = mnist_loss(logits, yb)
```
In this case, loss for all images is calculated in the following steps:
- Sigmoid `sigmoid(x) = 1 / (1 + e^(-x))` confines the one logit for each image from whatever positive or negative number it is into a value between 0 and 1 corresponding to a probability of being a 3 (closer to 1) or a 7 (closer to 0).
- The torch.where line calculates the loss for each image label: if labeled `1`, it calculates how far from 1 the prediction is; if labeled `0`, it calculates how far from 0 the prediction is
- Returns the loss as the average loss over all images in the batch

If we want to calculate loss when there are many classes (e.g. if we want to do the full dataset of [0,1,2,3,4,5,6,7,8,9]),
we can use the cross-entropy function.
```
import torch.nn.functional as F

loss = F.cross_entropy(model_logits, yb)
```
The cross entropy function:
- Uses the "softmax" function (`softmax(xᵢ) = e^xᵢ / (e^x₁ + e^x₂ + ... + e^xₙ)`) instead of sigmoid to squash each image's vector of logits into probabilities that all sum to 1.
- For each image, calculates loss as the negative log of the probability calculated for the correct label (`-log(softmax(x))`); we only care about that one
- Returns the loss as the average loss over all images in the batch

It uses `-log(p)` instead of a simple `1-p` subtraction for loss so that the optimizer has a larger range of values to work with.
When the probability decreases towards zero the loss can grow very large to indicate that a large change must be made.
When the loss is just `1-p`, it's bound between 0 and 1, which doesn't distinguish well between "slightly wrong" and "very wrong".

### Using the loss to update the model parameters
Calculating the loss just tells us how good or bad we were, on average, at predicting labels. We still need some way to learn
from this loss how to change the parameters (the matrix of weights `w1`, `w2` and vectors of biases `b1`, `b2`) for the 
next training loop.

How do we do that? Instead of using the loss value itself, we evaluate the slope of the loss function at each parameter's 
current value. The slope determines *how much* and in *which direction* we should modify the parameters to minimize the loss
for next time. We get the slope at a given point (parameter value) by taking the derivative of the function and evaluating 
it at that point. Because our loss function depends on multiple parameters (four in our case), we take partial derivatives 
with respect to each parameter. All of these partial derivatives taken together are called the gradient. Hence, the "gradient 
descent" part of "stochastic gradient descent".

The parameters have to be initialized with random values before we run any training loops, and they have to be updated 
at the end of each training loop. How do we do this in practice?

The initialization of the parameters is straightforward: 
```
def init_params(size, std=1.0): 
    return (torch.randn(size)*std).requires_grad_()
```
The most complicated part is `.requires_grad_()`. This is a pytorch method responsible for making the gradient calculation 
from the loss function possible. 

### How .requires_grad_() and .backward() work; or, how to take partial derivs of the loss function to get the gradient
When a tensor is updated with `.requires_grad_()`, it gains the attributes `.is_leaf=True` and `.grad=None`. When this 
tensor first participates in a calculation, the result is given two things:
- a `.grad_fn` attribute which is the gradient function of the last operation that produced it (e.g. `AddBackward0`, `MulBackward0`, `PowBackward0`, `SigmoidBackward0`, `ReluBackward0`, `LogSoftmaxBackward0`, `NllLossBackward0` (negative log likelihood), etc.), 
- and a `.grad_fn.next_functions` attribute which points to the leaf node (`AccumulateGrad`).

When the result participates in another calculation, the next result also gets a `.grad_fn` which is the gradient function 
of the last function that produced it, and a `.grad_fn.next_functions` which points to the gradient functions from the 
previous result rather than directly to the leaf node. And so on.

When `.backward()` is called on a result down the line, the gradients will be calculated and assembled by moving backwards
up the chain of functions, with chain rule, accumulating the results in place into the `.grad` attribute of the tensors 
originally set up with `.requires_grad_()` (our parameters). As each gradient calculation occurs, the tensor's `.grad_fn` 
stored computation buffers are released from memory. What's left is the value in `.grad` for the parameters, the same shape as the parameter itself, 
which we can use to update the weights. We'll talk about how in a bit. Once we do that we'll need to manually clear out 
the gradient with `.grad.zero_()` to prepare for the next training loop.

### The shapes of the parameter tensors
Although the parameter values are initialized randomly, the shape of each parameter is dictated by your data. In particular, 
- in the first layer of the model, `w1`'s 1st dimension is the pixels per image in your dataset
- in the last layer of the model, `w2`'s 2nd dimension is the number of output classes
- for our simple three layer model, `w1`'s 2nd dimension = `w2`'s 1st dimension so the matrix math works out. This is also the size for `b1`. This number, sometimes called the number of "activations", is something you choose.

It's easy to see in examples.

This would be param initialization when labels are either 3 or 7 (e.g. just one output class, "is 3?"):
```
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,1))
b2 = init_params(1)
```
This would be param initialization when images can be labeled any of the 10 digits from 0 to 9:
```
w1 = init_params((28*28,30))
b1 = init_params(30)
w2 = init_params((30,10))
b2 = init_params(10)
```

### Using the learning rate to update the model parameters
The last thing we haven't discussed for updating the parameters in the training loop is the "learning rate" (`lr`). We calculated 
each parameter's gradient and stored it in `.grad`, but this number just tells us a quantity and direction to *change* 
the old parameter value. That quantity is usually a lot bigger than we actually want to step the size of the parameter.
We'd likely overshoot the correct value, and have to move it all the way back in the other direction, maybe never minimizing
the loss!

Enter: the "learning rate". This is a number that we can choose, usually pretty small, like 1e-3 (.001) or 1e-4 (.0001).
We multiply the gradient by the learning rate, and subtract it from the parameter value. If we choose a learning rate that
is too small, each training loop change to our parameter values will be so small that it will take forever to train our model.
If we choose a learning rate that is too large, we might step the value of parameters too far and miss the best value.

Here we set the `.data` attribute of the parameter to update its value without adding the calculation to a gradient function graph,
and then zero out the gradient to prepare it for the next training loop. This stuff would be written into the optimizer 
class if we used one instead of doing all of this manually.
```
w1.data -= w1.grad * lr
w1.grad.zero_()
```

Ultimately we probably want to start our training loops with a higher learning rate and then systematically reduce it 
as we go through more and more training loops, because we're probably coming up on the optimal values for the parameters.
The machine learning libraries all have standard methods included in their optimizers for doing this. Fastai also includes
a method `.lr_find()` to choose an initial learning rate by running training loop with a bunch of different learning 
rates.

### Putting everything together: the training loop
Ok so, we have:
- the cleaned data divided into training, validation, and test sets
- we divided the training set into batches, and the training data into the independent (xb - images) and dependent (yb - labels) variables
- for each epoch we pass the all the batches through the training loop:
  - the data is sent through the model
  - we calculate the loss for the training loop using our chosen loss function (e.g. cross_entropy)
  - the parameter values are updated via the chosen learning rate * values calculated from the loss function via our chosen optimizer (e.g. SGD)

Here's what we've got:
```
params = [w1, b1, w2, b2] # initialized parameters

def train_epoch(model, lr, params):

    # loop through each batch of the training data in dl (DataLoaders)
    for xb,yb in dl:
    
        # run the model, calculate the loss
        model_logits = model(xb) # model uses the params we initialized
        loss = F.cross_entropy(model_logits, yb) # or mnist_loss(preds, yb)
        loss.backward() # this sets .grad in our params
        
        # update the parameters
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_() # this zeros out .grad for the param
            
    print("Training Loss: ", loss)
```

In the example notebooks for the course, the authors always print out the training loss, validation loss, and accuracy for each epoch.
We have calculated the training loss here, but what of the validation loss? And what is the "accuracy"?

## How do we validate the model?
Reiterating from the last section, we have now written code to train a model. But our only sense of how well it's performing 
is from calculating the loss on the training data with every loop. We expect that it will perform similarly on the validation 
set, because typically the validation data is just some random subset of the data that we kept aside specifically 
for validation. E.g. if we're predicting categorical labels, we expect future images of handwritten digits to have a pretty 
similar distribution to the present (e.g. people don't slowly over time start tilting their 8s until they eventually go 
sideways ∞). That's not always the case; we might want to manually make a validation set, particularly if things are changing 
over time.

Either way, we should check validation loss because it's our first proxy for how well the model will perform on real world 
data. We're only updating the model based on the training loss, so if the validation loss is not also going down, or if it's
increasing between epochs, that's a sure sign that we're overfitting on the training data. That means the model is less 
able to generalize on new data that it hasn't seen before. Usually the point of training a model is so it will do 
well on data it hasn't seen before, because we've already spent the time labeling the data we're training it on.

To do this, we can send the validation data through the model whose parameters we've just trained, and compute the loss.
This is the same as what we did for the training data, with the caveat that we don't want to build the computation graph
(save the gradient functions) and we don't want to call `loss.backward()` to compute gradients for the parameters. We can 
do this with `torch.no_grad()`. We're not going to change anything about the model from this loop, we're just checking on 
the loss.

We also don't have to bother batching up the images into random groups for this epoch because we're not looking to make 
improvements to the parameters, we just want the average loss for the total run anyway. We can still use batches though, 
if our validation dataset is too big to send into memory as one big tensor. Here we pretend we made a `valid_dl` object,
a DataLoaders for the validation data with a bunch of `xb` image tensors and `yb` label vectors.
```
def validate_epoch(model):
    # no gradient tracking
    with torch.no_grad():
        val_losses = [F.cross_entropy(model(xb), yb) for xb, yb in valid_dl] # or mnist_loss()
        val_loss_mean = torch.stack(val_losses).mean()
    print("Validation Loss: ", val_loss_mean)
```

## What about a metric?
Even more important than checking the validation loss is the metric that you care to check how your model performs in the
real world. Smaller loss will tell us that we were more confident and correct with our predicted labels, but a metric will 
tell us something more like we correctly labeled 24 out of 30 images (accuracy). It's possible to have a loss that implies 
that we were not confident with most of our labels, e.g. maybe like 51% confidence, but still find that we actually guessed 
correct labels for like 90% of the validation set.

For [Kaggle competitions](https://www.kaggle.com/competitions/), the metric is provided. It could be accuracy, F1, AUC, 
etc. Your success is based on how well your model performs as evaluated by the metric. For the MNIST problem we've been 
working on here, we'll define a batch accuracy metric for each of the classification tasks (`3` vs `7` and the full set 
of digits, 0-9).

For the binary case, `3` vs `7`:
```
def batch_accuracy_binary(model_logits, yb):
    preds = model_logits.sigmoid() # the value is clipped between 0 and 1
    correct = (preds>0.5) == yb    # If the value is >.5, it's labeled 1, which "is_3"
    return correct.float().mean()  # fraction correct in this batch
```

For the multiclass case, full set of digits, 0-9:
```
def batch_accuracy_multiclass(model_logits, yb):
    return (model_logits.argmax(dim=1) == yb).float().mean() # argmax because we guess the label with the highest probability
```

We can add the `batch_accuracy` calculation to the `validate_epoch` function:
```
def validate_epoch(model):
    with torch.no_grad():
        val_losses = [F.cross_entropy(model(xb), yb) for xb, yb in valid_dl] # or mnist_loss()
        val_loss_mean = torch.stack(val_losses).mean()
        val_accs   = [batch_accuracy_multiclass(model(xb), yb) for xb, yb in valid_dl] # or the _binary() option
        val_accs_mean = torch.stack(val_accs).mean()
    print("Validation Loss: ", val_loss_mean,
          "Validation Accuracy :", val_accs_mean)
```

And then we can wrap all of this up in a loop where we choose how many epochs (full run through all the data) we want to 
train for. The more epochs, the longer the training process will take:
```
def train_and_validate_model(model, lr, n_epochs):

    params = init_all_params() # set the w1, b1, w2, b2 with init_params() for the proper sizes (see above)
    
    for epoch in range(n_epochs):
        train_epoch(model, lr, params)  # prints training loss
        validate_epoch(model)           # prints validation loss & validation accuracy
```

Ok that was way longer and more involved than I intended to get in this post. Do you ... want to see how
this model performs on some handwritten images now?

## Saving and using the model!
Once you train your parameters, you might want to hang onto them! Those parameters, together with the model function, are 
what you use to guess the 0-9 digit in new images of handwritten digits.

If you want to do this one at a time, you can reshape your mxn-sized image (actually we've been working with square 28x28 
images this whole time) into a tensor shaped like `(1, m*n)`: an `image_tensor` with its `m*n` pixels all in a row:
```
from fastai.vision.all import *

# assuming our image is 28x28 pixels like all of our training data

# image gymnastics  -- we had to do a similar version of this at the beginning to make our huge stack of images
image = Image.open('my_digit.png').convert('L') # L = convert to grayscale
image_tensor = tensor(image).float()/255  # (28, 28); /255 for normalization
image_tensor = image_tensor.flatten().unsqueeze(0)  # (28, 28) → (784,) → (1, 784)

# image gymnastics note -- the "tensor" in the lines above (and in other places in this post) is fastai's implementation
# can also use torch's tensor together with numpy: torch.tensor(np.array(image))

# same method for getting the predicted label as in batch_accuracy, but no comparison with a label because we don't have it!
with torch.no_grad():
    model_logits = model(image_tensor)
    predicted_class = model_logits.argmax(dim=1) # index of the highest probability
    print("The digit is: ", predicted_class.item())
```