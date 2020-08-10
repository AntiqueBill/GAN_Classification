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
from Loss import loss_classification, loss_reconst, loss_self, generator_loss, discriminator_loss

class GAN(pl.LightningModule):

    def __init__(self, hparams, cnn_classification):
        super(GAN, self).__init__()
        self.hparams = hparams

        # networks
        
        self.generator = Generator(input_dim=hparams.input_dim)
        self.discriminator = Discriminator()
        self.cnn_classification = cnn_classification
        # 是否必须需要self？
        # self.loss_classification = loss_classification
        # self.loss_reconst = loss_reconst
        # self.loss_self = loss_self
        # self.generator_loss = generator_loss
        # self.discriminator_loss = discriminator_loss


    def forward(self, x):
        generator1, generator2 = self.generator(x)

        return generator1, generator2

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser

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
            loss_reconst, x_simple1, x_simple2, y_simple1, y_simple2 = loss_reconst(self.generator1, self.generator2,
                                                                         x_simple1, x_simple2, y_simple1, y_simple2)
            #根据判定的顺序选取对应的标签拼接构成discriminator的输入
            self.discriminator_label = torch.cat((x_simple1, x_simple2), 1)

            #第四个loss
            loss_classification = loss_classification(self.cnn_classification, self.generator1, self.generator2, y_simple1, y_simple2)
            #试图让生成pure数据时不进行反向传播
            x_pure = x_pure.detach()
            pure1, pure2 = self(x_pure)
            loss_self = loss_self(self.generator1, self.generator2, pure1, pure2)
            generator_loss = generator_loss(discriminator_input)

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
            d_loss = discriminator_loss(self.discriminator_input, self.discriminator_label)

            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
               # 'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
            
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

    #TODO test步骤的目的依旧是看4个g_loss，加上d_loss(gan后期应该没有意义)，以及最后看准确率
    # 和test步骤书写，目的是关注准确率
    # 测试阶段4个loss，loss_reconst,g_loss,loss_classification因为标签顺序不明都没了，d_loss可以有但没有意义
    # loss_self可以计算，关键是accuracy，这里考虑的是两个pred都算标签然后合起来看总的准确率，
    # 分为全对，半对，都不对三种，并计算总共预测对的数目，需要统计对的个数，所以test只有两个loss，loss_self和d_loss(没用)
    # 加上一个accuracy
    def test_step(self, batch, batch_idx):
        x, y, x_simple1, x_simple2, y_simple1, y_simple2, x_pure = batch
        generator1, generator2 = self(x)

        pred1 = self.cnn_classification(generator1)
        pred2 = self.cnn_classification(generator2)

        # discriminator_loss 这个也要顺序，所以继续无视，只剩loss_self一种
        # self.discriminator_input = torch.cat((generator1, generator2), 1)
        # self.discriminator_label = torch.cat((x_simple1, x_simple2), 1)
        # d_loss = discriminator_loss(self.discriminator_input, self.discriminator_label)
        #试图让生成pure数据时不进行反向传播
        x_pure = x_pure.detach()
        pure1, pure2 = self(x_pure)
        loss_self = loss_self(generator1, generator2, pure1, pure2)
    
        tqdm_dict = {'loss_self': loss_self}

        pred_all = torch.cat((pred1, pred2), 1)
        pred, _ = torch.sort(pred_all)
        label_all = torch.cat((y_simple1, y_simple2), 1)
        label, _ = torch.sort(label_all)
        num = torch.sum((b == d).int(), dim=1)
        output = OrderedDict({
                'loss_self': tqdm_dict,
               # 'progress_bar': tqdm_dict,
                'num': num,
                'log': tqdm_dict
            })
        
        return output

    def test_epoch_end(self, outputs):
        loss_self  = torch.stack([x['loss_self'] for x in outputs]).mean()
        all_num = torch.cat((x['num'] for x in outputs))
        N = float(all_num.shape[0])
        
        acc_2 = torch.sum(all_num.gt(1)).float() / N
        acc_1 = torch.sum(all_num.eq(1)).float() / N
        acc_0 = torch.sum(all_num.lt(1)).float() / N
        acc_all = (2.0 * acc_2 + acc_1) / 2.0
        acc_allwrong = (2.0 * acc_0 + acc_1) / 2.0

        tqdm_dict = {'acc_allright': acc_2, 'acc_oneright': acc_1, 'acc_zeroright': acc_0,
                     'acc_all': acc_all, 'acc_allwrong': acc_allwrong}
        output = OrderedDict({
                'loss_self': loss_self,
                'progress_bar': tqdm_dict,
                'accuracy': tqdm_dict,
                'log': tqdm_dict
            })

        return output

    