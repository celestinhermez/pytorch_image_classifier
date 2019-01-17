# We start by importing all the modules we will need, as well as the helper document with all our functions
import argparse
import torch
import json
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import helper

def main():
    
    # We display a short prompt
    print('Hello! This script will use a checkpoint to predict the top k classes of a picture of your choosing' +
          '\n'
          + 'You can choose how many classes to display.' + '\n' + 
          'You can also provide a mapping from the indices to the class names should you have it.' + '\n'
          'You can consult the help to see all the other arguments' + 
         '\n' + '\n')
    
    print('\n')
    
    # We parse the arguments from the command line
    args = helper.get_input_args_predict()
    
    image = args.path
    checkpoint = args.checkpoint
    top_k = args.top_k
    mapping = args.category_names
    gpu = args.gpu

    # We predict the categories, with their associated probabilities
    helper.predict(image, checkpoint, top_k, mapping, gpu)
    
main()