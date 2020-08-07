import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from Structure import Generator, Discriminator
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        
        self.generator = Generator(input_dim=hparams.input_dim)
        self.discriminator = Discriminator()


    def forward(self, x):
        generator1, generator2 = self.generator(x)

        return generator1, generator2

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser
    #复现文章《Separate In Latent Space: Unsupervised Single Image Layer Separation》的Lss,要最小化这个loss
    #存在问题：采用文章中的label=generator(clean_input)还是原计划的label = cnn(clean_input)?
    #这么考虑：自监督用generator(clean_input)，计算mse用cnn(clean_input)

    def loss_self(self, output1, output2, label1, label2, lamda1=0.5, lamda2=0.5, lamda3=1.0, alpha=1.4):
        g_ab = (alpha * torch.exp(alpha) - torch.abs(output1 - output2)) / (alpha^2) 
        distance_yz = 1 / (1 + torch.exp(g_ab))
        loss = lamda1 * torch.abs(output1 - label1) + lamda2 * torch.abs(output2 - label2) + lamda3 * (1 - distance_yz)

        return loss

    def discriminator_loss(self, y_hat, y):
        #采用软标签
        fake_label = 0.1 * torch.ones_like(y_hat)
        real_label = 0.9 + 0.1 * torch.rand(self.discrimiator_input.size(0), 1)
        if self.on_gpu:
            fake_label = fake_label.cuda()
            real_label = real_label.cuda()
        real_loss = F.binary_cross_entropy_with_logits(y, real_label)
        fake_loss = F.binary_cross_entropy_with_logits(y_hat, fake_label)
        total_loss = (real_loss + fake_loss) / 2.0

        return total_loss

    #判断输出特征和label特征的对应关系，进行排序，确定最小的是哪一个，然后取对应的loss
    def loss_reconst(self, output1, output2, label1, label2):
        loss11 = F.mse_loss(output1, label1)
        loss12 = F.mse_loss(output1, label1)
        loss21 = F.mse_loss(output1, label1)
        loss22 = F.mse_loss(output1, label1)

        a = torch.argmin([loss11, loss12, loss21, loss22])
        if a == 0 or a == 3:
            idx = 0
            loss = loss11 + loss22
            loss1 = loss11
            loss2 = loss22
        elif a == 1 or a == 2:
            idx = 1
            loss = loss12 + loss21
            loss1 = loss21
            loss2 = loss12

        return idx, loss, loss1, loss2

    #这个loss对照y_pred和y_predict, loss_generator = loss_self+loss_reconst
    def generator_loss(self, fake_output):
        #考虑软标签,不过生成器貌似不需要
        #label = 0.1 * torch.ones_like(fake_output)
        labels = torch.zeros_like(fake_output)
        loss_discrimiator = F.binary_cross_entropy_with_logits(fake_output, labels)
        loss = loss_self + loss_reconst + loss_discrimiator

        return loss

    #第四个loss，对应于classification的loss，前三个分别是自监督loss_self（区分分离特征），重构loss_reconst, 假标签loss：generator_loss 
    def loss_classification(self, cnn_classification, generator1, generator2, y_simple1, y_simple2):
        loss1 = F.cross_entropy(cnn_classification(generator1), y_simple1)
        loss2 = F.cross_entropy(cnn_classification(generator2), y_simple2)
        loss = (loss1 + loss2) / 2.0

        return loss


    def training_step(self, batch, batch_nb, optimizer_idx):
        x, y, x_simple1, x_simple2, y_simple1, y_simple2, x_pure = batch
        #self.last_imgs = imgs
        #选择最开始就将其转化为特征，这里的x_simple1和x_simple2均为特征
        # train generator
        if optimizer_idx == 0:

            # match gpu device (or keep as cpu)
            if self.on_gpu:
                x = x.cuda()

            # generate images
            self.generator1, self.generator2 = self(x)
            pure1, pure2 = self(x_pure)

            # loss
            self.discriminator_input = torch.cat((self.generator1, self.generator2), 1)
            
            #这个是软标签，用来提升性能，不知道好不好使,训练discriminator用
            # discriminator_label = 0.9 + 0.1 * torch.rand(self.discrimiator_input.size(0), 1) 
            # discriminator_label = torch.zeros(self.discrimiator_input.size(0), 1)

            if self.on_gpu:
                discriminator_label = discriminator_label.cuda()

            # loss计算
            idx, loss_reconst, loss1, loss2 = self.loss_reconst(self.generator1, self.generator2, x_simple1, x_simple2)
            #根据判定的顺序选取对应的标签拼接构成discriminator的输入
            if idx == 0:
                self.discriminator_label = torch.cat((x_simple1, x_simple2), 1)
            else:
                self.discriminator_label = torch.cat((x_simple2, x_simple1), 1)
                change = y_simple1
                y_simple1 = y_simple2
                y_simple2 = change

            #第四个loss
            loss_classification = self.loss_classification(cnn_classification, self.generator1, self.generator2, y_simple1, y_simple2)
            loss_self = self.loss_self(self.generator1, self.generator2, pure1, pure2)
            generator_loss = self.generator_loss(discriminator_input)

            g_loss = loss_self + loss_reconst + generator_loss + loss_classification

            tqdm_dict = {'g_loss': g_loss, 'loss_self': loss_self, 'loss_reconst': loss_reconst,
                     'loss_generator': generator_loss, 'loss_classification': loss_classification}
            output = OrderedDict({
                'loss': g_loss,
                #'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples
            d_loss = self.discriminator_loss(self.discriminator_input, self.discriminator_label)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
               # 'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
            
    #TODO valid和test步骤书写，目的是关注准确率
    def validation_step(self, batch, batch_idx):
        z = torch.randn(8, self.hparams.latent_dim)
        # match gpu device (or keep as cpu)
        if self.on_gpu:
            z = z.cuda(self.last_imgs.device.index)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
    
        return [opt_g, opt_d], []

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i):
        # 希望更多次更新discriminator
        if optimizer_i == 0:
            if batch_nb % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

        if optimizer_i == 1:
            if batch_nb % 2 == 0:
                optimizer.step()
                optimzer.zero_grad()

    