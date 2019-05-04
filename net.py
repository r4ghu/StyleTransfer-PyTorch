# net.py - StyleTransfer-PyTorch
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

# Some portions of the code are taken from...
# Source: https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/
import os
import glob

from loaders import *

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

import cv2
import time
import utils
from videocapture_async import VideoCaptureAsync

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
        if torch.cuda.is_available():
            # Sanity check - Empty CUDA Cache
            torch.cuda.empty_cache()
            # Enforce CUDNN backend
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True


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

        # Setup summary writer
        self.writer = SummaryWriter('runs/{}'.format(self.args.data_name))

    
    def _build_model(self):
        # Load the model
        _model_loader = ModelLoader(self.args)
        self.model = _model_loader.model

        if self.args.phase == 'train':
            self.vgg = _model_loader.vgg
            # If continue_train, load the pre-trained model
            if self.args.continue_train:
                self.load_model()

            # If multiple GPUs are available, automatically include DataParallel
            if self.args.multi_gpu and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self.vgg = nn.DataParallel(self.vgg)
            
            self.vgg.to(self.device)
        
        self.model.to(self.device)
        

    def _build_data_loader(self):
        _data_loader = DataLoader(self.args)
        self.train_loader = _data_loader.train_loader

    def _build_optimizer(self):
        if self.args.phase == 'train':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
    
    def _build_loss(self):
        if self.args.phase == 'train':
            self.loss = nn.MSELoss()

    def _train_epoch(self, epoch):
        self.model.train()
        
        # Start Training
        for batch_idx, (data, _) in enumerate(self.train_loader):
            n_batch = len(data)
            self.args.iter_count += 1
            self.optimizer.zero_grad()

            data = data.to(self.device)
            target = self.model(data)

            target = utils.normalize_batch(target)
            data = utils.normalize_batch(data)

            features_target = self.vgg(target)
            features_data = self.vgg(data)

            content_loss = self.args.content_weight * self.loss(features_target.relu_2_2, features_data.relu_2_2)
            
            style_loss = 0.
            for ft_y, gm_s in zip(features_target, self.gram_style):
                gm_y = utils.gram_matrix(ft_y)
                style_loss += self.loss(gm_y, gm_s[:n_batch, :, :])
            style_loss *= self.args.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            self.optimizer.step()

            if batch_idx % self.args.log_interval == 0 and batch_idx>0:
                # Add the values to Summary Writer
                self.writer.add_scalar('train/style_loss', style_loss.item(), self.args.iter_count)
                self.writer.add_scalar('train/content_loss', content_loss.item(), self.args.iter_count)
                self.writer.add_scalar('train/total_loss', total_loss.item(), self.args.iter_count)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss (Content, Style, Total): {:.6f}, {:.6f}, {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), content_loss.item(), 
                    style_loss.item(), total_loss.item()))
            
            del style_loss, content_loss, total_loss
            
            if self.args.iter_count % self.args.save_frequency == 0:
                self.save_model()

    # Start with serial execution
    # Replace torch transform with numpy and opencv
    def test(self, online=True, async=True):
        # TODO: Implement Offline mode
        stream_pre = torch.cuda.Stream()
        stream_pro = torch.cuda.Stream()
        stream_post = torch.cuda.Stream()

        if online:
            # Model is already loaded
            # Keep it in eval model
            self.model.eval()

            # Initialize the camera
            if async:
                camera = VideoCaptureAsync(0)
                camera.start()
            else:
                camera = cv2.VideoCapture(0)

            with torch.no_grad():
                end = time.time()
                while True:
                    with torch.cuda.stream(stream_pre):
                        ret, frame = camera.read()
                        if not ret:
                            break
                        
                        # Process the frame
                        frame = frame.swapaxes(1, 2).swapaxes(0, 1)
                        frame = frame[np.newaxis, :, :, :]
                        content_image = torch.from_numpy(frame)
                        content_image = content_image.to(self.device)

                    with torch.cuda.stream(stream_pro):
                        torch.cuda.current_stream().wait_stream(stream_pre)
                        content_image = content_image.type(torch.cuda.FloatTensor)
                        output = self.model(content_image)
                        output = output.clamp(0,255).type(torch.cuda.ByteTensor)
                        output = output.cpu()

                    with torch.cuda.stream(stream_post):
                        torch.cuda.current_stream().wait_stream(stream_pro)
                        output = output.numpy()[0].transpose(1,2,0)

                    time_process = time.time() - end
                    
                    # Render text
                    outText = "Style Transfer time: {} sec, {} FPS".format(time_process, 1.0/time_process)
                    print(outText)

                    # Show results
                    cv2.imshow('Frame', output)

                    k = cv2.waitKey(1)
                    if k==27:
                        break
                    end = time.time()

                    # t = time.time()
                    # frame = frame.swapaxes(1, 2).swapaxes(0, 1) # 0 ms
                    # print('Swap axes =', time.time() - t)
                    # t = time.time()
                    # frame = frame[np.newaxis, :, :, :] # 0 ms
                    # print('New axes =', time.time() - t)
                    # # t = time.time()
                    # # frame = frame.astype(np.float32)
                    # # print('Float32 =', time.time() - t)
                    # t = time.time()
                    # content_image = torch.from_numpy(frame) # 0 ms
                    # print('Torch array init =', time.time() - t)
                    # t = time.time()
                    # content_image = content_image.to(self.device) # < 1 ms
                    # torch.cuda.synchronize()
                    # print('Copy to GPU =', time.time() - t)
                    # t = time.time()
                    # content_image = content_image.type(torch.cuda.FloatTensor) # < 1 ms
                    # torch.cuda.synchronize()
                    # print('Convert to Float on GPU =', time.time() - t)
                    # t = time.time()
                    # output = self.model(content_image)
                    # torch.cuda.synchronize()
                    # print('Forward pass =', time.time()-t)
                    # t = time.time()
                    # output = output.clamp(0,255)
                    # torch.cuda.synchronize()
                    # print('Clamp on GPU =', time.time()-t)
                    # # t = time.time()
                    # # output = output.type(torch.cuda.ByteTensor)
                    # # torch.cuda.synchronize()
                    # # print('Float32 to UInt8 (GPU) =', time.time()-t)
                    # t = time.time()
                    # output = output.cpu()
                    # torch.cuda.synchronize()
                    # print('Copy to CPU =', time.time()-t)
                    # t = time.time()
                    # output = output.numpy()
                    # torch.cuda.synchronize()
                    # print('Numpy array conv =', time.time()-t)
                    # t = time.time()
                    # output = output[0]
                    # print('Tensor to CHW =', time.time()-t)
                    # t = time.time()
                    # output = output.transpose(1,2,0)
                    # print('CHW to HCW =', time.time()-t)
                    # t = time.time()
                    # output = output.astype("uint8")
                    # print('Float32 to UInt8 =', time.time()-t)
                    # time_process = time.time() - end
                    
                    # # Render text
                    # outText = "Style Transfer time: {} sec, {} FPS\n\n\n".format(time_process, 1.0/time_process)
                    # print(outText)

                    # # Show results
                    # cv2.imshow('Frame', output)

                    # k = cv2.waitKey(1)
                    # if k==27:
                    #     break
                    # end = time.time()
                torch.cuda.synchronize()
            if async:
                camera.stop()
            else:
                camera.release()
            cv2.destroyAllWindows()

        return
    
    def train(self):
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = utils.load_image(self.args.style_image, size=self.args.style_size)
        style = style_transform(style)
        style = style.repeat(self.args.train_batch_size, 1, 1, 1).to(self.device)

        features_style = self.vgg(utils.normalize_batch(style))
        self.gram_style = [utils.gram_matrix(y) for y in features_style]
        
        for epoch in range(1, self.args.epochs + 1):
            self._train_epoch(epoch)
        # Save the model finally
        if self.args.save_model:
            self.save_model()

    def save_model(self):
        if not os.path.exists(self.args.checkpoint_dir):
            os.makedirs(self.args.checkpoint_dir)
        model_filename = self.args.checkpoint_dir + 'model-{}-{}.pth'.format(self.args.data_name, self.args.iter_count)
        torch.save(self.model.state_dict(), model_filename)
    
    def load_model(self, iter_count=None):
        if not os.path.exists(self.args.checkpoint_dir):
            print('Checkpoint Directory does not exist. Starting training from epoch 0.')
            return
        # Find the most recent model file
        if iter_count is None:
            model_files = glob.glob(self.args.checkpoint_dir + '*.pth')
            if len(model_files) == 0:
                print('No model checkpoint files found.')
                return
            model_prefix = self.args.checkpoint_dir + 'model-{}-'.format(self.args.data_name)
            iter_numbers = [int(x[len(model_prefix):-4]) for x in model_files]
            self.args.iter_count = max(iter_numbers)
        else:
            self.args.iter_count = iter_count
        
        model_filename = self.args.checkpoint_dir + 'model-{}-{}.pth'.format(self.args.data_name, self.args.iter_count)
        self.model.load_state_dict(torch.load(model_filename))

    def export_model_to_onnx(self):
        self.model.eval()
        with torch.no_grad():
            # Create some sample input in the shape this model expects
            dummy_input = torch.randn(1, 3, 480, 640).to(self.device)

            # It's optional to label the input and output layers
            input_names = [ "image" ]
            output_names = [ "output" ]

            # Use the exporter from torch to convert to onnx 
            # model (that has the weights and net arch)
            torch.onnx.export(self.model, dummy_input, "StyleTransfer_v1.onnx", verbose=True, input_names=input_names, output_names=output_names)