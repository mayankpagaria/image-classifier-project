#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#PROGRAMMER:Mayank pagaria
#PURPOSE: Predicts the class the image belongs to: reads the image , predicts the class and returns the
#         top 5 probability for that image
#          and outputs the class the flower belongs to.

# Imports python modules
import os
import argparse
import torch
import numpy as np
import time
import json
from torchvision import datasets,transforms,models
from torch import nn,optim
from torch.autograd import Variable
from collections import OrderedDict
from PIL import Image
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

#Main function
def main():
    in_args = get_command_line_args()
    gpu = torch.cuda.is_available() and in_args.gpu
    model = models.densenet121(pretrained = True)
    optimizer = optim.Adam(model.classifier.parameters(),lr =0.001)
    l_model=load_model(in_args.checkpoint,model,optimizer)

    probabilities,classes= predict(in_args.image,model,in_args.top_k,)
    print(probabilities)
    print(classes)
    if in_args.category_names == None:
        for item in range(in_args.top_k):
            print("Class of Flower: {} & Probability: {:.3f}".format(classes[item],probabilities[item]))
    # else the json path is valid, and we print the top "k" items with their class names and probability
    else:
        cat_to_name = translate_cat_to_name(in_args.category_names)
        for item in range(in_args.top_k):
            print("Class of Flower: {} & Probability: {:.3f}".format(cat_to_name[classes[item]], probabilities[item]))

#################################################################################################################################
#Defining functions that is called in main function.
def get_command_line_args():
    parser= argparse.ArgumentParser()

    parser.add_argument('--image',type = str,help = 'image file',default='/home/workspace/aipnd-project/flowers/test/10/image_07090.jpg')

    parser.add_argument('--checkpoint',type = str,default='/home/workspace/paind-project/densenet121_checkpoint.pth',help='saved model checkpoint')

    parser.add_argument('--top_k',type =int,default=5,help='return the top k most likely classes')
    parser.set_defaults(top_k=5)

    parser.add_argument('--category_names',type = str,default='/home/workspace/aipnd-project/cat_to_name.json',help='File of category names')
    parser.add_argument('--learning_rate',type = float,default=0.001,help='model learning_rate')
    parser.add_argument('--gpu',type = str,default=True,help='train with gpu')
    parser.set_defaults(gpu = True)

    return parser.parse_args()
###############################################################################################################################
#Loading the save model from train.py file
def load_model(save_path,model,optimizer):
    if os.path.isfile(save_path):
        checkpoint=torch.load(save_path)
        model.classifier=checkpoint['classifier']
        model.class_to_idx=checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(save_path))


    return model
###############################################################################################################################
# function of cat_to_name file
def translate_cat_to_name(path):
    # Dummy list
    cat_to_name = []

    # Open the json file and extract & store class names to the list
    with open(path, 'r') as fp:
        cat_to_name = json.load(fp)

    return cat_to_name

###############################################################################################################################
#function of processing image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    size = 224
    # TODO: Process a PIL image for use in a PyTorch model
    width,height =image.size

    if height > width:
        height = int(max(height * size / width, 1))
        width = int(size)

    else:
        width = int(max(width * size / height, 1))
        height = int(size)

    resized_image = image.resize((width,height))

    x0 = (width - size) /2
    y0 = (height - size) /2
    x1 = x0 + size
    y1 = y0 + size

    cropped_image = image.crop((x0,y0,x1,y1))
    np_image = np.array(cropped_image)/225.
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    np_image_array =(np_image - mean) /std
    np_image_array =np_image.transpose((2,0,1))
    return np_image_array
###############################################################################################################################
#function of predicting image
def predict(image_path, model,topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # TODO: Implement the code to predict the class from an image file
    model.eval()
    use_gpu =False
    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()

    else:
        model.cpu()

    image = Image.open(image_path)
    np_array = process_image(image)
    tensor = torch.from_numpy(np_array)
    if use_gpu:
        var_inputs = Variable(tensor.float().cuda(),volatile = True)
    else:
        var_inputs = Variable(tensor,volatile = True)

    var_inputs = var_inputs.unsqueeze(0)
    output = model.forward(var_inputs)
    ps = torch.exp(output).data.topk(topk)
    probabilities = ps[0].cpu() if use_gpu else ps[0]
    classes = ps[1].cpu() if use_gpu else ps[1]
    class_to_idx_inverted = {model.class_to_idx[k]:k for k in model.class_to_idx}
    mapped_classes = list()
    for label in classes.numpy()[0]:
        mapped_classes.append(class_to_idx_inverted[label])

    return probabilities.numpy()[0],mapped_classes


if __name__ == "__main__":
   main()
