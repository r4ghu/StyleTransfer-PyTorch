import argparse
import os
import sys
import time
import re

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.onnx

import utils
from models.transformer_net import TransformerNet
from models.vgg import Vgg16

class Net(object):
    def __init__(self, args):
        # Args
        self.args = args

        # Setup manual seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Set the device
        # Select GPU:0 by default
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load the data
        print('Load the data...', end='')
        self._build_data_loader()
        print('DONE')

        # Load the model
        print('Load the model...', end='')
        self._build_model()
        print('DONE')

        # Setup Optimizer
        print('Build optimizer...', end='')
        self._build_optimizer()
        print('DONE')

        # Setup Loss
        print('Build loss...', end='')
        self._build_loss()
        print('DONE')

        # # Setup summary writer
        # self.writer = SummaryWriter('runs/{}'.format(self.args.data_name))

    
    def _build_model(self):
        # Load the model
        self.transformer = TransformerNet().to(self.device)
        self.vgg = Vgg16(requires_grad=False).to(self.device)

    def _build_data_loader(self):
        transform = transforms.Compose([
            transforms.Resize(self.args.image_size),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        self.train_dataset = datasets.ImageFolder(self.args.dataset, transform)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size)

        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = utils.load_image(self.args.style_image, size=self.args.style_size)
        style = style_transform(style)
        self.style = style.repeat(self.args.batch_size, 1, 1, 1).to(self.device)

        

        
    def _build_optimizer(self):
        self.optimizer = Adam(self.transformer.parameters(), self.args.lr)
        
    def _build_loss(self):
        self.mse_loss = torch.nn.MSELoss()
        
    def _train_epoch(self, epoch):
        self.transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(self.train_loader):
            n_batch = len(x)
            count += n_batch
            self.optimizer.zero_grad()

            x = x.to(self.device)
            y = self.transformer(x)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = self.vgg(y)
            features_x = self.vgg(x)

            content_loss = self.args.content_weight * self.mse_loss(features_y.relu_2_2, features_x.relu_2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, self.gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += self.mse_loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= self.args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            self.optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % self.args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), epoch + 1, count, len(self.train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if self.args.checkpoint_model_dir is not None and (batch_id + 1) % self.args.checkpoint_interval == 0:
                self.transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(self.args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(self.transformer.state_dict(), ckpt_model_path)
                self.transformer.to(self.device).train()
        
    # def test(self):
        
    def train(self):
        features_style = self.vgg(utils.normalize_batch(self.style))
        self.gram_style = [utils.gram_matrix(y) for y in features_style]
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
            self.test()
        # Save the model finally
        # if self.args.save_model:
        #     self.save_model()
        # save model
        self.transformer.eval().cpu()
        save_model_filename = "epoch_" + str(self.args.epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
            self.args.content_weight) + "_" + str(self.args.style_weight) + ".model"
        save_model_path = os.path.join(self.args.save_model_dir, save_model_filename)
        torch.save(self.transformer.state_dict(), save_model_path)
        
    # def save_model(self):
        
    # def load_model(self, iter_count=None):
        