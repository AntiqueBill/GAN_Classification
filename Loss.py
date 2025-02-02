import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Structure import Generator, Discriminator
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
    
#复现文章《Separate In Latent Space: Unsupervised Single Image Layer Separation》的Lss,要最小化这个loss
#存在问题：采用文章中的label=generator(clean_input)还是原计划的label = cnn(clean_input)?
#这么考虑：自监督用generator(clean_input)，计算mse用cnn(clean_input)

def loss_self(output1, output2, label1, label2, lamda1=0.5, lamda2=0.5, lamda3=1.0, alpha=1.4):
    g_ab = (alpha * torch.exp(torch.tensor(alpha)) - torch.abs(output1 - output2)) / (alpha ** 2)
    distance_yz = 1 / (1 + torch.exp(g_ab))
    loss = (lamda1 * torch.abs(output1 - label1) + lamda2 * torch.abs(output2 - label2) + lamda3 * (1 - distance_yz)).mean()

    return loss

def discriminator_loss(y_hat, y):
    #采用软标签
    fake_label = 0.1 * torch.ones_like(y_hat)
    real_label = 0.9 + 0.1 * torch.rand_like(y)
    if torch.cuda.is_available():
        fake_label = fake_label.cuda()
        real_label = real_label.cuda()
    real_loss = F.binary_cross_entropy_with_logits(y, real_label)
    fake_loss = F.binary_cross_entropy_with_logits(y_hat, fake_label)
    total_loss = (real_loss + fake_loss) / 2.0

    return total_loss

#判断输出特征和label特征的对应关系，进行排序，确定最小的是哪一个，然后取对应的loss,并将相对应的label和y_label都更换
#来自文章《Permutation invariant training of deep models for speaker-invariant multi-talker speech separation》
def loss_reconst(output1, output2, label1, label2, y_label1, y_label2):
    batch_size = output1.shape[0]
    idx = []
    loss1 = []
    loss2 = []
    for i in range(batch_size):
        a, b, la, lb, y_la, y_lb = output1[i], output2[i], label1[i], label2[i], y_label1[i], y_label2[i]
        loss11 = F.mse_loss(a, la)
        loss12 = F.mse_loss(a, lb)
        loss21 = F.mse_loss(b, la)
        loss22 = F.mse_loss(b, lb)

        a = torch.argmin(torch.tensor([loss11+loss22, loss12+loss21]))
        if a == 0:
            #idx.append(0)
            loss1.append(loss11)
            loss2.append(loss22)
        elif a == 1:
            #idx.append(1)
            loss1.append(loss12)
            loss2.append(loss21)
            # can not use copy function in torch, cause "Only Tensors
            # created explicitly by the user (graph leaves) support the
            #  deepcopy protocol at the moment" problem, consider clone() function
            #c = copy.deepcopy(label1[i])
            c = label1[i].clone()
            label1[i] = label2[i]
            label2[i] = c

            c = y_label1[i].clone()
            y_label1[i] = y_label2[i]
            y_label2[i] = c
            
    loss1 = torch.mean(torch.tensor(loss1))
    loss2 = torch.mean(torch.tensor(loss2))
    loss = (loss1 + loss2) / 2.0   

    return loss, label1, label2, y_label1, y_label2

def generator_loss(fake_output):
    #考虑软标签,不过生成器貌似不需要
    #label = 0.1 * torch.ones_like(fake_output)
    labels = torch.zeros_like(fake_output)
    if torch.cuda.is_available():
        labels = labels.cuda()
    loss = F.binary_cross_entropy_with_logits(fake_output, labels)

    return loss

#第四个loss，对应于classification的loss，前三个分别是自监督loss_self（区分分离特征），重构loss_reconst, 假标签loss：generator_loss 
def loss_classification(cnn_classification, generator1, generator2, y_simple1, y_simple2):
    # can not use F.cross_entropy(cnn_classification(generator1), y_simple1) directly
    # because the output is [batch, 6] and y_simple1 is [batch, 1]
    # consider squeeze to transform [batch, 1] to [batch]
    y_simple1 = y_simple1.squeeze()
    y_simple2 = y_simple2.squeeze()
    loss1 = F.cross_entropy(cnn_classification(generator1), y_simple1)
    loss2 = F.cross_entropy(cnn_classification(generator2), y_simple2)
    loss = (loss1 + loss2) / 2.0

    return loss

