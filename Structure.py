import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class Generator(nn.Module):
    def __init__(self, input_dim=3000):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        def cnnblock(in_channel, out_channel, kernel_size, stride=1, normalize=True):
            layers = [nn.Conv1d(in_channel, out_channel, kernel_size, stride)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # set batchnorm1d=32 because the input of generator is 256*32*739, 
        # if nn.Flatten, the input is 256*(32*739), but if not use nn.Flatten, 
        # the output of self.generator will be [N, 32, 60], so I have to use Flatten
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *cnnblock(1, 256, 5, 3, normalize=False),
            *cnnblock(256, 128, 5, 3),
            *cnnblock(128, 64, 5, 2, normalize=False),
            *cnnblock(64, 32, 5, 2),
            *cnnblock(32, 16, 3, normalize=False),
            *cnnblock(16, 8, 3),          
            nn.Flatten()
        )

        self.generator1 = nn.Sequential(
            *block(608, 500, normalize=False),
            *block(500, 400),
            *block(400, 200, normalize=False),
            *block(200, 60)
        )

        self.generator2 = nn.Sequential(
            *block(608, 500, normalize=False),
            *block(500, 400),
            *block(400, 200, normalize=False),
            *block(200, 60)
        )

    def forward(self, x):
        features = self.cnn(x)
        generator1 = self.generator1(features)
        generator2 = self.generator2(features)
        return generator1, generator2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(120, 100, normalize=False),
            *block(100, 80),
            *block(80, 60, normalize=False),
            *block(60, 30),
            *block(30, 15, normalize=False),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.model(input)
        
        return output