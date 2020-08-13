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
from torch.utils.data import random_split

# ##这一步需要添加引入之前的CNN提取特征模型，然后提取特征将x_simple1和x_simple2转换为特征向量x_feature1，x_feature2
# class CnnModule(pl.LightningModule):
#     def __init__(self,  *args, **kwargs):
#         super().__init__()

# cnnmodel = CnnModule.load_from_checkpoint('')

# for param in cnnmodel.parameters():
#     param.requires_grad = False

# cnn_features = cnnmodel.cnn
# cnn_classification = cnnmodel.classification

class GANDataset(torch.utils.data.Dataset):
    def __init__(self, train=True, file_dir, cnn_features):
        data = h5py.File(file_dir, 'r')
        self.train = train
        if self.train:
            train_x = np.array(data['train_x'])
            train_y = np.transpose(data['train_y'])
            #train_y -= 1
            x_simple1 =  np.array(data['x_simple1'])
            x_simple2 =  np.array(data['x_simple2'])
            y_simple1 =  np.transpose(data['y_simple1'])
            y_simple2 =  np.transpose(data['y_simple2'])
            x_pure = np.array(data['x_pure'])
            
            self.train_x = torch.from_numpy(train_x).unsqueeze(1).float()
            x_simple1 = torch.from_numpy(x_simple1).unsqueeze(1).float()
            x_simple2 = torch.from_numpy(x_simple2).unsqueeze(1).float()
            #将单个数据转化为特征
            self.x_simple1 = cnn_features(x_simple1)
            self.x_simple2 = cnn_features(x_simple2)

            self.y_simple1 = torch.from_numpy(y_simple1).long()
            self.y_simple2 = torch.from_numpy(y_simple2).long()
            self.train_y = torch.from_numpy(train_y).squeeze().long()
            #self.x_pure = torch.from_numpy(x_pure).unsqueeze(1).float()
            #self.train_y = F.one_hot(train_y)

            print("train_x.shape:", self.train_x.shape)
            print("train_y.shape:", self.train_y.shape)
            print("train_x_simple1.shape", self.x_simple1.shape)
        else:
            test_x = np.array(data['test_x']) 
            test_y = np.transpose(data['test_y'])
            #test_y -= 1 
            x_simple1 =  np.array(data['test_x_simple1'])
            x_simple2 =  np.array(data['test_x_simple2'])
            y_simple1 =  np.transpose(data['test_y_simple1'])
            y_simple2 =  np.transpose(data['test_y_simple2'])
            x_pure = np.array(data['test_x_pure'])

            self.test_x = torch.from_numpy(test_x).unsqueeze(1).float()
            x_simple1 = torch.from_numpy(x_simple1).unsqueeze(1).float()
            x_simple2 = torch.from_numpy(x_simple2).unsqueeze(1).float()
            #将单个数据转化为特征
            self.x_simple1 = cnn_features(x_simple1)
            self.x_simple2 = cnn_features(x_simple2)

            self.y_simple1 = torch.from_numpy(y_simple1).long()
            self.y_simple2 = torch.from_numpy(y_simple2).long()
            self.test_y = torch.from_numpy(test_y).squeeze().long()
            self.x_pure = torch.from_numpy(x_pure).unsqueeze(1).float()
            #self.test_y = F.one_hot(test_y)

            print("test_x.shape:", self.test_x.shape)
            print("test_y.shape:", self.test_y.shape)
            print('test_x_simple.shape', self.x_simple1.shape)

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def __getitem__(self, index):
        if self.train:
            x, y, x_simple1, x_simple2, y_simple1, y_simple2, x_pure = self.train_x[index], self.train_y[index], self.x_simple1[index],        self.x_simple2[index], self.y_simple1[index], self.y_simple2[index],self.x_pure[index] 
        else:
            x, y, x_simple1, x_simple2, y_simple1, y_simple2, x_pure = self.test_x[index], self.test_y[index], self.x_simple1[index],         self.x_simple2[index], self.y_simple1[index], self.y_simple2[index], self.x_pure[index]

        return x, y, x_simple1, x_simple2, y_simple1, y_simple2, x_pure

class GANDataLoader(pl.LightningDataModule):
    def __init__(self, batch_size, file_dir, cnn_features, valid=0.2):
        super().__init__()
        self.batch_size = batch_size
        train_dataset = GANDataset(train=True, file_dir, cnn_features)
        test_dataset = GANDataset(train=False, file_dir, cnn_features)
        valid_num = len(train_dataset) * valid
        train_num = len(train_dataset) * (1 - valid)
        train_data, valid_data = random_split(train_dataset, [train_num, valid_num])
        self.train_dataset = train_data
        self.valid_dataset = valid_data
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=256, shuffle=False)