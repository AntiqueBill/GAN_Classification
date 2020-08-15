import h5py
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torch.nn as nn 
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class AEDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        #super(self).__init__()
        data = h5py.File('./samples/dataset_CNN.mat', 'r')
        self.train = train
        if self.train:
            train_x = np.transpose(data['train_x'])
            train_y = np.transpose(data['train_y']) 
            train_y -= 1

            self.train_x = torch.from_numpy(train_x).unsqueeze(1).float()
            self.train_y = torch.from_numpy(train_y).squeeze().long()
            #self.train_y = F.one_hot(train_y)

            print("train_x.shape:", self.train_x.shape)
            print("train_y.shape:", self.train_y.shape)
        else:
            test_x = np.transpose(data['test_x']) 
            test_y = np.transpose(data['test_y'])
            test_y -= 1 

            self.test_x = torch.from_numpy(test_x).unsqueeze(1).float()
            self.test_y = torch.from_numpy(test_y).squeeze().long()
            #self.test_y = F.one_hot(test_y)

            print("test_x.shape:", self.test_x.shape)
            print("test_y.shape:", self.test_y.shape)

    def __len__(self):
        if self.train:
            return len(self.train_x)
        else:
            return len(self.test_x)

    def __getitem__(self, index):
        if self.train:
            x, y = self.train_x[index], self.train_y[index]
        else:
            x, y = self.test_x[index], self.test_y[index]

        return x, y

class MyDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        train_dataset = AEDataset(train=True)
        test_dataset = AEDataset(train=False)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=128, shuffle=True)

class CNNModel(pl.LightningModule):
    
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 30, 9, 3),
            nn.ELU(),
            nn.Conv1d(30, 25, 9, 3),
            nn.ELU(),
            nn.Conv1d(25, 20, 9, 3),
            nn.ELU(),
            nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Linear(720, 500),
            nn.ELU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 300),
            nn.ELU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.ELU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 120),
            nn.ELU(),
            nn.BatchNorm1d(120),
            nn.Linear(120, 60)
            #nn.ELU(),,
        )
        #self.features = nn.Linear(60, 40)
        self.classification = nn.Sequential(
            nn.Linear(60, 6),
            #nn.Softmax()
            #nn.Sigmoid()
        ) 
            
    def forward(self, x):
        #called with self(x)
        features = self.cnn(x)
        output = self.classification(features)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        #print('val_x.shape', x.shape)
        y_hat = self(x)
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_hat, y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #print('y_hat.shape', y_hat.shape)
        #print('y.shape', y.shape)
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(y_hat, y)
        labels_hat = torch.argmax(y_hat, dim=1)
        test_acc = accuracy(labels_hat, y)
        #test_acc = (y_hat == y).sum()
        test_output = {'test_loss': loss, 'test_acc': test_acc}
        return test_output

    def test_epoch_end(self, test_step_outputs):
        test_epoch_loss = torch.stack([x['test_loss'] for x in test_step_outputs]).mean()
        test_epoch_acc = torch.stack([x['test_acc'] for x in test_step_outputs]).mean()
        return {
            'test_loss': test_epoch_loss,
            'log':{'avg_test_loss': test_epoch_loss, 'avg_test_acc': test_epoch_acc}
        }

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters())
