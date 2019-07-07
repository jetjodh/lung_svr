# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:58:10 2019

@author: jetjo
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import fastai as fai
import fastai.vision as faiv
import torchvision
from torch.utils.data import DataLoader
import torch
from torch import nn
from niftidataset import *
import torchvision.transforms as torch_tfms

data_dir = 'G:\ImageCliff\TrainingSet_1_of_2'

tfms = torch_tfms.Compose([RandomCrop2D(32, None), ToTensor()])
ds = NiftiDataset(data_dir, data_dir, tfms)

def my_collate(batch):
    x = torch.stack([torch.stack([b[0],b[0],b[0]]) for b in batch])
    y = torch.stack([b[1].contiguous().view(-1) for b in batch])
    return (x, y)

tdl = DataLoader(ds, batch_size=2)
vdl = DataLoader(ds, batch_size=2)

idb = faiv.ImageDataBunch(tdl, vdl, collate_fn=my_collate)

head = nn.Sequential(
        fai.AdaptiveConcatPool2d(), 
        fai.Flatten(),
        nn.Linear(1024,32**2),
        nn.ReLU(True),
        nn.Linear(32**2, 32**2))

loss = nn.MSELoss()
loss.__name__ = 'MSE'
learner = faiv.ConvLearner(idb, faiv.models.resnet18, 
                           custom_head=head, 
                           loss_func=loss, 
                           metrics=[loss])

print(faiv.num_features(learner.model))

for x,y in idb.train_dl:
    print(loss(learner.model(x),y))
    break
    
for x,y in idb.valid_dl:
    print(loss(learner.model(x),y))
    break

learner.lr_find();

learner.recorder.plot()

learner.fit(1)