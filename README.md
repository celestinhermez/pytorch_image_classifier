# PyTorch Image Classifier

This project exemplifies how to build an image classifier using PyTorch. The main
files include at a development notebook, with detailed steps to pre-process images,
train a network, save it and use it for predictions; and a Python app which can be run
from the command line to train a neural network on any dataset of images.

The idea is to make it easy for anyone to build their own image classifier with very 
high accuracy. This repository has two main objectives:
* provide a simple example of how to build an image classifier for flowers and use
it for predictions, all from the command line
* provide a more general command line application which can train and apply an image
classifier to any dataset of images

While the files are not clearly separated between both objectives, as a rule of thumb
the default configuration are related to the flower dataset, while customized applications
are for personal datasets.

## Installation

In order to use this Python app, simply fork and clone the repository for use on your
own computer.

## File Structure

### cat_to_name.json

This file is a mapping from category to flower name for the flower dataset.

### checkpoint.pth

A checkpoint for a classifier trained on the flower dataset. It leverages the ResNet152
architecture, dropout, ReLU activation functions as well as one hidden layer with 1,000
units in its fully connected layer.

### helper.py

A file with all the helper functions leveraged by the train and predict scripts:

* **get_input_args_train()**: a function to parse the command line arguments passed to
the train script
* **get_input_args_predict()**: a function to parse the command line arguments passed to
the predict script
* **build_model()**: a function to build a model based on a desired architecture, a number
of hidden units and the number of categories to predict
* **load_images()**: a function to load and transform images (both train and validation sets)
* **validation()**: a function to do a validation pass during the training of our model
* **train_model()**: a function to train the model, displaying some useful information
during the training (such as training loss, validation loss, accuracy, which epoch)
* **save_checkpoint()**: a function to create a checkpoint from a train model
* **load_checkpoint()**: a function to load a checkpoint and create a trained model
from it
* **get_image()**: a function to get an image from a given filepath
* **process_image()**: transform an image to a form suitable for analysis
* **imshow()**: show an image
* **predict()**: predict categories and the associated probabilities for a given image

More information on the arguments for each function can be found in the file itself,
where each function has a corresponding docstring.

### Image_Classifier_Project.ipynb

The development notebook for this project. The checkpoint comes from running this notebook.
This notebook was created and run using [Google Colab](https://colab.research.google.com/) 
in order to benefit from free GPU compute time.

### predict.py

The script which can be called from the command line to predict the classes associated
with a given image. More details on the associated command line arguments below.

### train.py

The script to train a classifier on a given training set. More details on the associated
command line arguments below.

## Usage

Describe the different command line options available, either train on the provided 
dataset or own dataset. 

Describe the checkpoint

## Caveats

Version of PyTorch
Notebook to be opened in Colab, file structure to change
File structure for train and validation dataset
If load a checkpoint, `arch` must be a key with a string with the architecture
 (either resnet152 or VGG16). This could easily be extended

## Credits