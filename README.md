# Barks and Meows Classification using CNN
Binary audio classification task analogous to cats and dogs image classification.
This provides an example **end-to-end** audio classification model using convolutional neural networks.
By end-to-end I mean the model is directly applied upon audio samples, or PCM data, 
rather than using spectrogram-based features such as log-magnitude spectrograms or MFCCs.

This also provides a simple script for '_auralizing_' , as an audio counterpart of visualizing,
that generates the "best example" for each class - barks or meows - according to a trained model.


# Dataset
- https://www.kaggle.com/mmoreaux/environmental-sound-classification-50

cats 319 vs dogs 369

# Model
The neural network model is a simple convolutional neural network (CNN) using 1D kernels.

# Auralization
