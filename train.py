# We start by importing all the modules we will need, as well as the helper document with all our functions
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from torch.optim import lr_scheduler
import helper

def main():
        
    # We display a short prompt
    print('Hello! This script will train a neural network with a Resnet or VGG architecture with only one layer.' +
          '\n' + 'You can consult the help for other command line arguments.')

    nb_categories = int(input('Please input the number of categories of your target variable: '))
    
    print('\n')

    args = helper.get_input_args_train()
    data_dir = args.dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    # We check the user used one of the possible architecture
    while not arch in ('vgg16', 'resnet152'):
        arch = input('You can only choose between vgg16 and resnet152 for an architecture: ')
    
    # We now create our train, validation and test loaders with our images
    # This assumes our files follow the appropriate structure
    dataloaders = helper.load_images(data_dir)

    # We create the architecture of our model

    model = helper.build_model(arch, hidden_units, nb_categories)

    # We choose our criterion and our optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Train our model
    model = helper.train_model(model, criterion, optimizer, scheduler, dataloaders, epochs, gpu)

    # Save the checkpoint
    helper.save_checkpoint(model, criterion, optimizer, learning_rate, epochs, save_dir, arch)
    
main()

