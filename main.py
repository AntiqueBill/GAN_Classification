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
from argparse import Namespace
from argparse import ArgumentParser

#TODO 完善parser与主程序（超参数），涉及到Model.py的程序
parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--conda_env', type=str, default='some_name')
parser.add_argument('--notification_email', type=str, default='will@email.com')

# add model specific args
parser = LitModel.add_model_specific_args(parser)

# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = Trainer.add_argparse_args(parser)

args = parser.parse_args()
args = {
    'batch_size': 32,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'latent_dim': 3000
}

hparams = Namespace(**args)

checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

model = GAN(hparams)
trainer = pl.Trainer(checkpoint_callback=checkpoint_callback)
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=valid_dataloader)
trainer.test(test_dataloader=test_dataloader)