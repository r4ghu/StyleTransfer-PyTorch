# data_loader.py - StyleTransfer-PyTorch
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

import torch
from torchvision import datasets, transforms

class DataLoader:
    train_loader, test_loader = None, None

    def __init__(self, args):
        # Set arguments for GPU
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.kwargs_gpu = {'num_workers': args.num_workers, 'pin_memory': True} if self.use_cuda else {}
        if args.data_name == 'MNIST':
            self.loadMNIST(args)
        if args.data_name == 'COCO':
            self.loadCOCO(args)
    
    def loadMNIST(self, args):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = datasets.MNIST(root=args.data_dir, train=True, 
                                  download=True, transform=transform)
        testset = datasets.MNIST(root=args.data_dir, train=False, 
                                 transform=transform)
        
        self.train_loader = torch.utils.data.DataLoader(trainset, 
                                                        batch_size=args.train_batch_size, 
                                                        shuffle=True, **self.kwargs_gpu)
        
        self.test_loader = torch.utils.data.DataLoader(testset, 
                                                        batch_size=args.test_batch_size, 
                                                        shuffle=True, **self.kwargs_gpu)
    
    def loadCIFAR10(self, args):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = datasets.CIFAR10(root=args.data_dir, train=True, 
                                    download=True, transform=transform)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, 
                                   download=True, transform=transform)
        
        self.train_loader = torch.utils.data.DataLoader(trainset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True, **self.kwargs_gpu)
        self.test_loader = torch.utils.data.DataLoader(testset,
                                                        batch_size=args.test_batch_size,
                                                        shuffle=True, **self.kwargs_gpu)
    
    def loadCOCO(self, args):
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

        trainset = datasets.ImageFolder(args.data_dir, transform)

        self.train_loader = torch.utils.data.DataLoader(trainset,
                                                        batch_size=args.train_batch_size,
                                                        shuffle=True, **self.kwargs_gpu)