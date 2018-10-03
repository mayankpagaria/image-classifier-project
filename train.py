#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#PROGRAMMER:Mayank pagaria
#PURPOSE: Trains a neural network: reads a pretrained network and creates re-train the neural netwrok   #         with differnt hyperparameters.
#          and outputs the training loss,validation loss and validation accuracy.


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


output_size =102
print_every=40

#Main function
def main():
    in_args=get_command_line_args()
    gpu = torch.cuda.is_available() and in_args.gpu
    dataloaders,class_to_idx=get_loaders(in_args.dir)
    model = get_model(in_args.arch,in_args.hidden_layers,class_to_idx)
    model,optimizer,criterion = build_network(in_args.hidden_layers,in_args.learning_rate,in_args.arch,class_to_idx)

    train_model=train(model,in_args.epochs,in_args.learning_rate,criterion,optimizer,dataloaders['training'],dataloaders['validation'],in_args.gpu)

    test_loss,accuracy = validate(model,criterion,dataloaders['testing'],in_args.gpu)

    if in_args.save_dir:
        if not os.path.exists(save_dir):
            os.mrkdir(save_dir)
            save_path = in_args.save_dir+'/'+in_args.arch+'_checkpoint.pth'
    else:
        save_path = in_args.arch+'_checkpoint.pth'

    save(in_args.arch,in_args.learning_rate,in_args.hidden_layers,in_args.epochs,model,save_path,optimizer,class_to_idx,in_args.gpu)

    print("checkpoint saved:{}".format(save_path))
    print("Data Directory:{}".format(in_args.dir))
    print("Architecture:{}".format(in_args.arch))
    print("Test Loss:{:.3f}".format(test_loss))
    print("Accuracy:{:.3f}".format(accuracy))
###############################################################################################################################
#Defining functions that is called in main function.
def get_command_line_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('--dir',type = str ,default = '/home/workspace/aipnd-project/flowers' ,help = 'Directory of flower images')
    parser.add_argument('--gpu',type = str,default=True,help='train with gpu')
    parser.set_defaults(gpu = True)
    parser.add_argument('--arch',type = str,default='densenet121',help='choose model')
    parser.add_argument('--save_dir',type = str,help='Directory to save checkpoint')
    parser.add_argument('--learning_rate',type = float,default=0.001,help='model learning_rate')
    parser.add_argument('--hidden_layers',type = int,default=512,help='Number of hidden layers')
    parser.add_argument('--epochs',type = int,default=9,help='Numbers of epochs train')

    return parser.parse_args()
#################################################################################################################################
#Defining functions that is called in main functions
#Defining function in which data of flowers is there
def get_loaders(dir):
    data_dir = dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = {
    'training': transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomRotation(30),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],                                                                         [0.229,0.224,0.225])]),

    'validation': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],                                                                        [0.229,0.224,0.225])]),

    'testing': transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485,0.456,0.406],                                                                         [0.229,0.224,0.225])])}


    # TODO: Load the datasets with ImageFolder
    image_datasets ={'training':datasets.ImageFolder(train_dir,transform=data_transforms['training']),
                    'validation':datasets.ImageFolder(valid_dir,transform=data_transforms['validation']),
                    'testing':datasets.ImageFolder(test_dir,transform=data_transforms['testing']),}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {'training':torch.utils.data.DataLoader(image_datasets['training'],batch_size=64,shuffle=True),
                   'validation':torch.utils.data.DataLoader(image_datasets['validation'],batch_size=64,shuffle=True),
                   'testing':torch.utils.data.DataLoader(image_datasets['testing'],batch_size=64,shuffle=True)}

    class_to_idx=image_datasets['training'].class_to_idx
    return dataloaders,class_to_idx
###############################################################################################################################
#defining the function of creating and loading the model and pretrained models
def get_model(arch,hidden_layers,class_to_idx):
    if arch == 'vgg13_bn':
       model = models.vgg13_bn(pretrained = True)
       input_size = model.classifier[0].in_features

    else:
        arch == 'densenet121'
        model = models.densenet121(pretrained = True)
        input_size = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False


    input_size = model.classifier.in_features
    print('input size:',input_size)
    output_size= 102

    classifier = nn.Sequential(OrderedDict([('fc0',nn.Linear(input_size,hidden_layers)),
                                           ('relu',nn.ReLU()),
                                           ('fc1',nn.Linear(hidden_layers,output_size)),
                                           ('output',nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    model.class_to_idx = class_to_idx
    return model
###############################################################################################################################
#defining the function of creating the network
def build_network(hidden_layers,learning_rate,arch,class_to_idx):
    model= get_model(arch,hidden_layers,class_to_idx)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learning_rate)
    optimizer.zero_grad()
    return model,optimizer,criterion
###############################################################################################################################
#defining the function of training the build network
def train(model,epochs,learning_rate,criterion,optimizer,training_loader,validation_loader,gpu):
    model.train()
    print_every = 40
    steps = 0
    use_gpu = False

    if torch.cuda.is_available():
        use_gpu = True
        model.cuda()

    else:
        model.cpu()


    for epoch in range(epochs):
        running_loss = 0

        for inputs,labels in iter(training_loader):
            steps += 1

            if use_gpu:
                inputs =Variable(inputs.float().cuda())
                labels = Variable(labels.long().cuda())

            else:
                inputs = Variable(inputs)
                labels = Variable(labels)

            optimizer.zero_grad()
            output = model.forward(inputs)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss,accuracy = validate(model,criterion,validation_loader,gpu)

                print("Epoch:{}/{}".format(epoch+1,epochs),
                     "Training Loss:{:.3f}".format(running_loss/print_every),
                     "Validation Loss:{:.3f}".format(validation_loss),
                     "Validation Accuracy:{:.3f}".format(accuracy))
###############################################################################################################################
#defining the function validate
def validate(model,criterion,data_loader,gpu):
    model.eval()
    accuracy = 0
    test_loss = 0

    for inputs,labels in iter(data_loader):
        if torch.cuda.is_available():
            inputs = Variable(inputs.float().cuda(),volatile=True)
            labels = Variable(labels.long().cuda(),volatile=True)

        else:
            inputs = Variable(inputs,volatile=True)
            labels = Variable(labels,volatile=True)

        output = model.forward(inputs)
        test_loss += criterion(output,labels).item()
        ps =torch.exp(output).data
        equality = (labels.data == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss/len(data_loader),accuracy/len(data_loader)
###############################################################################################################################
#defining the function save and saveing the trained model in file checkpoint
def save(arch,learning_rate,hidden_layers,epochs,model,save_path,optimizer,class_to_idx,gpu):
    state = {'arch':arch,
            'learning_rate':learning_rate,
            'hidden_layers':hidden_layers,
            'epochs':epochs,
            'state_dict':model.state_dict(),
            'optimizer':optimizer.state_dict(),
            'classifier':model.classifier,
            'class_to_idx':model.class_to_idx}

    torch.save(state,save_path)

if __name__ == "__main__":
   main()
