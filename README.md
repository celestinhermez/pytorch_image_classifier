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
own computer. The following libraries are necessary:
* argparse
* torch (version 0.4.1)
* json
* numpy
* os
* torchvision
* PIL
* collections

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
command line arguments below. It will build a classifier with:
* one hidden layer and the specified number of hidden units
* ReLU activation functions
* Adam gradient descent algorithm
* LogSoftMax output and NLLoss function
* dropout

## Usage

The app operates through two scripts.

### train.py

This allows to train an image classifier. Several command line arguments are available:
*  **dir (required)**: the relative path to the folder which contains the images to train on. 
This folder must follow a specific file structure: folder_name/train/class_nb/image_1.jpg for instance.
It is particularly important that both a `valid` and `train` folders be present, and
the images corresponding to the same (numbered category) be grouped together. Mimicking
the file structure of the provided flower folder is an easy way to structure for this
task. 
* **save_dir (optional)**: the directory where to save the checkpoint. By default it
is the current working directory
* **arch (optional)**: a string to choose the architecture of the classifier. This app
leverages transfer learning, and at the moment will only work with the ResNet152 and
VGG16 architectures. Use resnet152 or vgg16 to specify which arhictecture to work with,
it is resnet152 by default
* **learning_rate (optional)**: a float to specify the learning rate to start with, knowing
a scheduler is used to gradually reduce it as the network is being trained. It is 0.001 
by default
* **hidden_units (optional)**: an integer representing the number of units 
contained in the hidden layer, 1000 by default
* **epochs (optional)**: an integer to specify the number of epochs to train for
* **gpu (optional)**: if present, this argument specifies a GPU is available for training,
which will greatly speed up the calculations. By default it is absent so we assume no
GPU's are available

With all of this in mind, the simplest way of using this script (with most of the default
values for the command line arguments)

```bash
python3 train.py flowers
```
The script will require user input for the number of classes present in the dataset.
We could easily change this to being a command line argument, but since there is not
an easy default we chose to leave it as user input for now. If this script were 
automated it would have to be passed as a command line argument.

### predict.py

This script leverages a trained network to make predictions. More precisely, it
uses a saved checkpoint to re-create a classifier and then use it to predict.
Several command line arguments are associated with this script:
* **path (required)**: the path to the image whose class we want to predict
* **checkpoint (required)**: the path to the checkpoint we want to load
* **top_k (optional)**: how many predicted classes we want to return. By default
5 classes are returned
* **category_names (optional)**: a JSON mapping between class numbers and names.
By default it is empty and class numbers are returned
* **gpu (optional)**: whether a GPU is available. By default, it is absent so
no GPU is used

In order to leverage the checkpoint and mapping included in this repository
to make a prediction for one of the flowers present in the `valid` folder and
return class names:
```bash
python3 predict.py flowers/valid/1/image_06739.jpg checkpoint.pth --category_names cat_to_name.json
``` 

## Caveats

There are several caveats to keep in mind in order to properly leverage this
project:
* the code was built with version `0.4.1` of PyTorch. Bugs may occur if the version
on the local machine is more recent (a stable version `1.0.0` was released in late 2018).
In order to avoid any problems, I recommend running these scripts in a virtual environment
which has the appropriate version of PyTorch installed
* the development notebook included was created and run using Google Colab with
my personal Drive mounted on it. The file paths will need to be adapated should
this notebook be re-used
* if one is trying to use a checkpoint not created by the train script included,
please make sure that the architecture of the model is saved as a string associated
with the `arch` key, as this is parsed by the loading function and used as a way
of knowing which architecture to load


## Potential improvements

This project could be extended to provide a broader range of utilities to build an image
classifier and more flexibility for the user. Some possibilities include:
* more architectures than VGG16 and ResNet152. The challenges here are different
architectures require different number of input/output units, and all these
cases need to be included in the code
* more flexibility on the number of hidden layers: right now the classifier is
always built with one hidden layer, but we could consider extending this to any (>= 0)
number.
* provide more options in terms of the activation function: always ReLU currently
* provide more options in terms of the gradient descent algorithm, always Adam currently
* more output and loss functions
* more control over the scheduler used

## Credits

This project was inspired by Udacity's PyTorch Scholarship challenge, when I learned
a lot about neural networks (CNN's, RNN's, LSTM's) and their applications. 