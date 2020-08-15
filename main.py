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
from pytorch_lightning.metrics.functional.classification import accuracy
from torch.utils.data import random_split
from argparse import Namespace, ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from Dataset import GANDataset, GANDataModule
from CNN import CNNModel
from Model import GAN

def main(hparams):

    checkpoint_callback = ModelCheckpoint(
            filepath=hparams.save_dir,
            save_top_k=1,
            verbose=True,
            monitor='val_loss',
            mode='min'
    )

    #引入cnn的模型
    cnnmodel = CNNModel.load_from_checkpoint(hparams.cnn_model_dir)
    cnn_features = cnnmodel.cnn
    cnn_classification = cnnmodel.classification
    
    datamodule = GANDataModule(hparams.batch_size, cnn_features, hparams.data_dir, hparams.valid)
    # train_dataloader = data.train_dataloader()
    # valid_dataloader = data.val_dataloader()
    # test_dataloader = data.test_dataloader()
    
    logger = TensorBoardLogger(save_dir="./lightning_logs",name='gan_logs')
    # trainer = pl.Trainer(checkpoint_callback=checkpoint_callback, logger=logger, progress_bar_refresh_rate=50,
    #                     gpus=hparams.gpus, min_epochs=hparams.min_epochs, max_epochs=hparams.max_epochs)
    # trainer = pl.Trainer(hparams, checkpoint_callback=checkpoint_callback, logger=logger, progress_bar_refresh_rate=50)
    trainer = pl.Trainer.from_argparse_args(hparams, checkpoint_callback=checkpoint_callback, logger=logger, progress_bar_refresh_rate=50)
    if hparams.train == True:    
        model = GAN(cnn_classification, hparams)
        trainer.fit(model, datamodule=datamodule)
    else:    
        test_model = GAN.load_from_checkpoint(checkpoint_path=hparams.load_dir)
        trainer.test(test_model, datamodule=datamodule)

if __name__ == "__main__":
    
    parser = ArgumentParser()

    # add PROGRAM level hparams
    parser.add_argument('--data_dir', type=str, default='./samples/dataset_GAN.mat')   
    parser.add_argument('--cnn_model_dir', type=str, default='./checkpoint/cnn.ckpt')
    parser.add_argument('--loss_self_lamda1', type=float, default=0.5)
    parser.add_argument('--loss_self_lamda2', type=float, default=0.5)
    parser.add_argument('--loss_self_lamda3', type=float, default=1.0)
    parser.add_argument('--loss_self_alpha1', type=float, default=1.4)
    parser.add_argument('--optim_Adam_lr', type=float, default=0.001)
    parser.add_argument('--optim_Adam_b1', type=float, default=0.9)
    parser.add_argument('--optim_Adam_b2', type=float, default=0.999)
    parser.add_argument('--save_dir', type=str, default='./checkpoint/gan/gan.ckpt')
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--min_epochs', type=int, default=0)
    parser.add_argument('--load_dir', type=str, default='../checkpoint/gan/gan.ckpt')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--valid', type=float, default=0.2)
    parser.add_argument('--input_dim', type=int, default=3000)

    # add model specific hparams
    #parser = GAN.add_model_specific_args(parser)
    #print('parser', parser)
    # add all the available trainer options to argparse
    # parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    #hparams = pl.Trainer.add_argparse_args(hparams)
    #hparams = Namespace(**hparams)
    main(hparams)