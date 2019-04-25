# vgg.py - StyleTransfer-PyTorch
# 
# BSD 3-Clause License
# 
# Copyright (c) 2019, Sri Raghu Malireddi
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
# 	list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# 	this list of conditions and the following disclaimer in the documentation
# 	and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
# 	contributors may be used to endorse or promote products derived from
# 	this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Source: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg16_pretrained_features = models.vgg16(pretrained=True).features

        self.slice_1 = nn.Sequential()
        self.slice_2 = nn.Sequential()
        self.slice_3 = nn.Sequential()
        self.slice_4 = nn.Sequential()

        for x in range(4):
            self.slice_1.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(4, 9):
            self.slice_2.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(9, 16):
            self.slice_3.add_module(str(x), vgg16_pretrained_features[x])
        for x in range(16, 23):
            self.slice_4.add_module(str(x), vgg16_pretrained_features[x])
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.slice_1(x)
        x_relu_1_2 = x
        x = self.slice_2(x)
        x_relu_2_2 = x
        x = self.slice_3(x)
        x_relu_3_3 = x
        x = self.slice_4(x)
        x_relu_4_3 = x

        vgg16_outputs = namedtuple("VggOutputs", ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_3'])
        return vgg16_outputs(x_relu_1_2, x_relu_2_2, x_relu_3_3, x_relu_4_3)