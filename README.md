# Barks and Meows Classification using CNN

Binary audio classification task analogous to cats and dogs image classification.
This provides an example **end-to-end** audio classification model using convolutional neural networks.
By end-to-end I mean the model is directly applied upon audio samples, or PCM data, 
rather than using spectrogram-based features such as log-magnitude spectrograms or MFCCs.

This also provides a simple script for '_auralizing_' , as an audio counterpart of visualizing,
that generates the "best example" for each class - barks or meows - according to a trained model.


# Dataset

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3563990.svg)](https://doi.org/10.5281/zenodo.3563990)

BarkMeowDB, a small dataset containing about 50 wav files each of barking and meowing sounds, 
is used for training.

- [Download from zenodo.org](https://zenodo.org/record/3563990#.Xesx_JMzZ25)

# Model
The neural network model is a simple convolutional neural network (CNN) using 5 1D convolution kernels.

# Auralization

Auralization uses a technique called **gradient ascent** to render an input that maximizes the probability for a given label.
Look in the directory `auralize_examples/` for auralization results.

# Further details

For more details, visit my [blog post about barks vs meows classification](https://dansuh17.github.io/2019/12/16/bark-meow.html).
