# This file is meant to contain all the helper functions to augment our train.py and predict.py scripts
# We start by importing all the modules we need
import argparse
import torch
import json
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import os
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# We start by building an argument parser

def get_input_args_train():
    """
    This function parses the command line for arguments and returns them as an ArgumentParser object
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    cwd = os.getcwd()
    
    # Creates the parser 
    parser = argparse.ArgumentParser()

    # We create 7 command lines argument, to cover all our cases
    
    # The first argument is where we can find the directory where our images are. It is a required argument
    parser.add_argument('dir', type=str, default='flowers', 
                        help='data_directory where to find the images on which to train')
    
    # The second optional argument points to a directory where to save our checkpoint. By default, it is our current working directory
    parser.add_argument('--save_dir', action='store', default = cwd, help='Store the directory where to save the checkpoint')

    # The third optional argument enables choosing a model architecture. We offer the choice betwen VGG16 and ResNet152
    parser.add_argument('--arch', action='store', default = 'vgg16', help='Choose the architecture of the model, between ResNet152 (default) and VGG16')

    # The fourth optional argument enables choosing a learning rate, 0.001 by default
    parser.add_argument('--learning_rate', action='store', default = 0.001, type = float, help='Choose the learning rate of the model, 0.001 by default')
    
    # The fifth optional argument allows choosing the number of hidden units, 1000 by default. We limit ourselves to one hidden layer in this particular case
    parser.add_argument('--hidden_units', action='store', default = 1000, type = int, help='Choose the number of hidden units, 1,000 by default')
    
    # The sixth optional argument allows choosing for how many epochs we want to train our model, 5 by default
    parser.add_argument('--epochs', action='store', default = 5, type = int, help='Choose the number of epochs, 5 by default')
    
    # The last positional argument allows setting a True or False value to the GPU variable, in order to choose whether the model will be trained on a GPU or not
    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='By default, the model will not be trained on a GPU ; if this argument is present it will')
    
    return parser.parse_args()

def get_input_args_predict():
    """
    This function parses the command line for arguments and returns them as an ArgumentParser object
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creates the parser 
    parser = argparse.ArgumentParser()

    # We create 5 command lines argument, to cover all our cases
    
    # The first argument is where we can find the directory where our image is. It is a required argument
    parser.add_argument('path', type=str, 
                        help='path to the image we want to submit to our image classifier')
    
    # The second  argument points to a checkpoint we want to load
    
    parser.add_argument('checkpoint', action='store', help='Checkpoint we want to load')
    
    # The third optional argument enables choosing how many classes we want to return
    
    parser.add_argument('--top_k', action='store', default = 5, type = int, help='Choose how many predicted classes we want to return')
    
    # The fourth optional argument enables providing a mapping from the category indices to their names
    parser.add_argument('--category_names', action='store', default = '', help='The mapping from index to category names')
    
    # The fifth positional argument allows setting a True or False value to the GPU variable, in order to choose whether the model will be trained on a GPU or not
    parser.add_argument('--gpu', action='store_true', default=False, dest='gpu', help='By default, the model will not be trained on a GPU ; if this argument is present it will')
    
    return parser.parse_args()

def build_model(architecture , hidden_units, nb_categories):
    """ This function takes a model architecture, a number of hidden units and the number of categories as its inputs,
    and it returns the pre-trained model with a modified classifier
    """
    
    # We start by loading the chosen model architecture, VGG16 or DenseNet121
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    # We freeze the parameters of our convulational layers
        for param in model.parameters():
            param.requires_grad = False
    
    # We build our model classifier. We limit ourselves to one hidden layer
    
        classifier = nn.Sequential(OrderedDict([
                            ('dropout', nn.Dropout(0.2)),
                            ('fc1', nn.Linear(25088, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout', nn.Dropout(0.3)),
                            ('fc2', nn.Linear(hidden_units, nb_categories)),
                            ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.classifier = classifier
    
    if architecture == 'resnet152':
        model = models.resnet152(pretrained=True)
    
    # We freeze the parameters of our convulational layers
        for param in model.parameters():
            param.requires_grad = False
    
    # We build our model classifier. We limit ourselves to one hidden layer
    
        classifier = nn.Sequential(OrderedDict([
                            ('dropout', nn.Dropout(0.2)),
                            ('fc1', nn.Linear(2048, hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout', nn.Dropout(0.3)),
                            ('fc2', nn.Linear(hidden_units, nb_categories)),
                            ('output', nn.LogSoftmax(dim=1))
                          ]))
    
        model.fc = classifier
        
    return model

def load_images(data_dir, train_dir_ext = '/train', valid_dir_ext = '/valid'):
    """ Takes a data directory, and an extension for the train, validation and test directories
    Returns a dictionary of dataloaders for future use """
    
    # Start by setting up the transforms for our data
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])}

    train_dir = data_dir + train_dir_ext
    valid_dir = data_dir + valid_dir_ext

    dirs = {'train': train_dir,
            'valid': valid_dir}

    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                      for x in ['train', 'valid']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=16,
                                                  shuffle=True)
                   for x in ['train', 'valid']}


    return dataloaders

def validation(model, validateloader, criterion, gpu):
    """ This function is used to run a validation pass on our model"""
    # We run our validation on the GPU as well
    if gpu:
        model.to('cuda')
        
    validation_loss = 0
    accuracy = 0
    for images, labels in validateloader:
        
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
            
        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss, accuracy

def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, gpu):
  '''
  INPUT:
  model: a model instance we want to train
  criterion: the loss function
  optimizer: which optimizer we want to use
  scheduler: the scheduler to adapt our learning rate
  num_epochs: the number of epochs
  dataloaders: dataloaders with all our data
  device: whether we want to train on a CPU or GPU

  OUTPUT:
  a trained model instance
  '''

      # We train our network, including a validation pass inside
      # We print every 20 steps to have visibility somewhat regularly in our model


      # We send our model to cuda if available
      if gpu:
          model.to('cuda')

      # We initialize various values
      epochs = num_epochs
      steps = 0
      running_loss = 0
      print_every = 20
      valid_loss_min = np.Inf
      best_acc = 0.0

      for e in range(epochs):
          # We make sure we are in training mode to make sure dropout is activated
          # We send the inputs and the labels to cuda as well if available
          # We take a scheduler step to change our learning rate as we train
          if scheduler != None:
            scheduler.step()

          model.train()
          for images, labels in dataloaders['train']:
              if gpu:
                  images, labels = images.to('cuda'), labels.to('cuda')
              steps += 1

              optimizer.zero_grad()

              output = model.forward(images)
              loss = criterion(output, labels)
              loss.backward()
              optimizer.step()

              running_loss += loss.item()


              if steps % print_every == 0:
                  # We put the network in evaluation mode, to turn off dropout,
                  # otherwise we will have particularly low accuracy
                  model.eval()

                  # Turn off gradients for validation
                  with torch.no_grad():
                      validation_loss, accuracy = validation(model, dataloaders['valid'], criterion)

                  print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(validation_loss/len(dataloaders['valid'])),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])))

                  running_loss = 0

                  if validation_loss/len(dataloaders['valid']) <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    validation_loss/len(dataloaders['valid'])))
                    torch.save(model.state_dict(), 'model_flowers.pt')
                    valid_loss_min = validation_loss/len(dataloaders['valid'])
                    best_acc = accuracy/len(dataloaders['valid'])

                  # Make sure training is back on
                  model.train()
      print('Best accuracy: {:4f}'.format(best_acc))

      # load best model weights
      model.load_state_dict(torch.load('model_flowers.pt'))
      return model

def save_checkpoint(model, optimizer, learning_rate, epochs, save_dir, arch):
    """ This function allows us to save a checkpoint for later use """
    
    checkpoint = {'input_size': model.classifier.fc1.in_features,
              'output_size': model.classifier.fc2.out_features,
              'arch': arch,
              'activation_function': nn.ReLU(),
              'hidden_layer': model.classifier.fc1.out_features,
              'epochs': epochs,
              'dropout': nn.Dropout(0.2),
              'output': nn.LogSoftmax(dim = 1),
              'learning_rate': learning_rate,
              'optimizer_state': optimizer.state_dict,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def load_checkpoint(checkpoint):
    """ This function allows us to load a previously saved checkpoint 
    It takes a model architecture and the filepath to 
    """
  # We start by loading our checkpoint, which contains information on the classifier which we need to build again
    checkpoint = torch.load(checkpoint)   
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    if checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    # We start by loading the architecture of the classifier again
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(checkpoint['input_size'], checkpoint['hidden_layer'])),
                            ('relu1', checkpoint['activation_function']),
                            ('dropout', checkpoint['dropout']),
                            ('fc2', nn.Linear(checkpoint['hidden_layer'], checkpoint['output_size'])),
                            ('output', checkpoint['output'])
                          ]))
    
    # We then load some additional elements
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.epochs = checkpoint['epochs']
    model.lr = checkpoint['learning_rate']
    
    
    return model

def get_image(file_path):
    ''' Takes a string file path for an image as its input,
        returns a PIL image as its output
    '''
    image = Image.open(file_path)
    
    return image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # We resize the images to make sure there are all 224 x 224 pixels
    # We first create a thumbnail if the shortest size is 256 pixels
    if min(image.size) <= 256:
        image.thumbnail((224,224))
    
    
    # We then crop out the center 224 x 224 portion of the image
    width, height = image.size   

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))
    
    # We convert the PIL image to a NumPy array and change the color 
    
    np_image = np.array(image)/255
    
    # We normalize
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_normalized = (np_image - mean)/ std
    
    # Finally we transpose to make sure the color channel is the right dimension
    image = image_normalized.transpose((2,0,1))
    
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, checkpoint, topk, mapping, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
        The model input should be a checkpoint we want to use
    '''
    model = load_checkpoint(checkpoint)
    
    if gpu:
        model.to('cuda')
    model.eval()
    
    # We get the image thanks to our function using PIL
    image = get_image(image_path)
    
    # We process the image and convert it to a tensor
    
    processed_image = process_image(image)
    processed_image = torch.from_numpy(processed_image).float()
    
    # We calculate the probabilities for the different classes using log-softmax
    
    processed_image.unsqueeze_(0)
    with torch.no_grad():
        if gpu:
            processed_image.to('cuda')
        output = model.forward(processed_image)

    ps = torch.exp(output)
    
    # We only display the top k classes and their probabilities
    # We can first get the indices for these classes
    
    topk_probs = torch.topk(ps, topk, dim = 1)[0]
    topk_probs = topk_probs.numpy()
    topk_idx = torch.topk(ps, topk, dim = 1)[1]
    topk_idx += 1 # We need to add one to match the index to the category names
    
    # Then we use the mapping to get the category names, and convert everything to lists
    
    topk_classes = []
    topk_probs = topk_probs.tolist()
    
    # If we don't have a mapping, we simply display the top probabilities and the associated indices
    if mapping == '':
        topk_idx = topk_idx.tolist()
        print('The top k probabilities are: ' + str(topk_probs) + '\n')
        print('The top k categories indices are: ' + str(topk_idx) + '\n')
    
    else:
        with open(mapping, 'r') as f:
            cat_to_name = json.load(f)
        for i in range(topk):
            topk_classes.append(cat_to_name[str(topk_idx.numpy()[:,i].item())])
        
        print('The top k probabilities are: ' + str(topk_probs) + '\n')
        print('The top k classes are: ' + str(topk_classes) + '\n')
        
    