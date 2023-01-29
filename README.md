# Kowalski

Let Kowalski find the images of your description.

![](misc/header.png)

## Description

Finding an image from a long list of images on your device can be exhausting.
Kowalski can find the required image given a description of the image in question.
It matches the embedding of all the images with the embedding of the image description and returns the images with the best similarity score.
The app has a simple gradio interface for interactive use.


![](misc/gradio.png)

## Setting-up locally

1. Install the required dependencies

```bash
python -m requirements.txt
```
2. Run the server

```bash
python -m app
```

## Dependencies
1. 🤗 Transformers
2. Sentence-Transformers
3. Gradio
