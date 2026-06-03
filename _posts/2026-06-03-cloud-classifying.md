---
layout: post
title: cloud classifying
date: 2026-06-03 20:35 -0400
---

# Classifying some clouds online

According to lesson 2, the latest and greatest way to put models on the web is no longer jupyter with binder,
but now hugging face and gradio.

Making a huggingface space with gradio was very straightforward. I followed the steps in [the blog](https://www.tanishq.ai/blog/posts/2021-11-16-gradio-huggingface.html) 
that the lesson2 video suggested.

1. make a hugging face account
2. make a hugging face space, clone it locally (it's very similar to a github clone/update/push workflow) & follow the rest of the instructions in the page that gets created when you create a space
3. including make a token on the hugging face site and use that as your password when you push your local updates
4. pushing that code loads the demo app into the space, which lets you write `$something` into a box that you can submit, and it appears in a second column as "Hello, `$something`!!" 
5. then you can delete the demo code (or comment it out if you're me and want to keep the past as notes)
6. and add in the code to load your model & run the inference (basically copy-pasting lines from the lesson 2 jupyter notebook where you learn what functions to call to classify one supplied image to the model that we trained, and copy-pasting gradio syntax from the huggingface/gradio blog post)

For the sake of time, I will move fast and not break things. Instead of adding the model training code to that repo
and then having to figure out dependencies right now, i'm just going to copy my `export.pkl` that I created in
the lesson2 jupyter notebook into the huggingface repo that I just made. If you want some example images to show up, you also have to copy-paste those in.

Um. Well that didn't work. 

`remote: -------------------------------------------------------------------------
remote: Your push was rejected because it contains files larger than 10 MiB.
remote: Please use https://git-lfs.github.com/ to store large files.
remote: See also: https://hf.co/docs/hub/repositories-getting-started#terminal
remote: 
remote: Offending files:
remote:   - export.pkl (ref: refs/heads/main)
remote: -------------------------------------------------------------------------
`

In the past I spent some time with git-lfs and the question of what to do with large files in repos so ... I really should have known better. But the tutorial said use the pickle!!!!
If the idea is to train models in this repo though, huggingface probably has something to say about it.

Oh yeah, they do: "Do you have files larger than 10MB? Those files should be tracked with git-xet, which you can initialize with:
`git xet install`". Never heard of it. Hello 2026 and the rock I've been living under for the past six years.

But it looks like they've also moved on from pickles to something called [`safetensors`](https://huggingface.co/docs/safetensors/index).
It's not like huggingface is [incompatible with fastai](https://huggingface.co/docs/hub/en/fastai).

## Ok I'm not going ot go nuts here

Just going to try the simplest thing which is to install git-lfs for now and upload the pickle.
I trained it one more time after cleaning the data and the error improved by an order of magnitude (12% to 1%).
But for some reason when I looked at the new losses:

```
interp_clean = ClassificationInterpretation.from_learner(learn_clean)
interp_clean.plot_confusion_matrix()
interp_clean.plot_top_losses(5,nrows=5)
```

...there were still images in there that looked like the ones I got rid of with the "cleaner":

```
cleaner = ImageClassifierCleaner(learn)
cleaner
for idx in cleaner.delete(): cleaner.fns[idx].unlink()
for idx,cat in cleaner.change(): shutil.move(str(cleaner.fns[idx]), path/cat)
```

After I had cleaned I made the datablock and path variables again, and results were better...
so why were the wrong images still in there? Did the clean wizard just not find all of the bad images?
I mean I could look through the images to verify this but not right now.

## After adding lfs for the pickle, is it going to just work?

No, of course not.
1. hf mentions adding in `requirements.txt` but doesn't mention what they already have installed? (e.g. they must have already installed `gradio`). Got this from the tutorial's imports (`fastai`, and `scikit-image`). I put `skimage` in the requirements.txt file first and yeah I got an error when I `git push`ed. 
2. the blog code has some issues due to "bitrot", mostly with the args for gradio's `gr.Interface()`. I just copy-pasted the failure log messages from the hf site gui into chatgpt to figure out what the changes should be. I tried to ask Claude first but it was like, "honestly, I might not be up to date with the latest..." blah blah blah ok Claude thanks for expressing your limitations.

I should have probably run this locally before pushing to hf but I didn't set up a `.venv` (and added a `.gitignore`) until a few errors in.
Luckily the hf code tree has a great interface for [checking the status & the logs](https://huggingface.co/spaces/emmjab/dllearning/tree/main)! 

Anyway to run locally you just have to start the venv and then call `python app.py` but fyi for the sake of time I didn't try that yet.

Because now my cloud classifier is live! I tried it and it works! [Check it out here!](https://huggingface.co/spaces/emmjab/dllearning)
