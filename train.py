import os
from os.path import join
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import DataLoaderHelper

from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import G, D, weights_init
from util import load_image, save_image
from skimage.measure import compare_ssim as ssim

# for debug
import torchvision
import matplotlib.pyplot as plt
import numpy as np

class GanGlobalIllumination():
    def __init__(self, opt, root_dir):
        self.opt = opt
        print('=> Loading datasets')
        
        train_dir = join(root_dir + self.opt.dataset, "train")
        test_dir = join(root_dir + self.opt.dataset, "val")

        train_set = DataLoaderHelper(train_dir)
        val_set = DataLoaderHelper(test_dir)

        self.batch_size = self.opt.train_batch_size
        self.n_epoch = self.opt.n_epoch
        self.lastEpoch = 0

        # workers == 0, run main process, default 0
        self.train_data = DataLoader(dataset=train_set, num_workers=self.opt.workers, batch_size=self.opt.train_batch_size, shuffle=True)
        self.val_data = DataLoader(dataset=val_set, num_workers=self.opt.workers, batch_size=self.opt.test_batch_size, shuffle=False)

        # debug
        """
        for (i, images) in enumerate(self.train_data):
            (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (images[0], images[1], images[2], images[3], images[4])
            # debug (1,3,256,256)
            debug_albedo = albedo_cpu[0]
            debug_albedo = debug_albedo.add_(1).div_(2)
            debug_albedo = debug_albedo.numpy()
            print(debug_albedo.shape)
            debug_albedo *= 255.0
            debug_albedo = debug_albedo.clip(0, 255)
            debug_albedo = np.transpose(debug_albedo, (1, 2, 0)) # (width, height, channels)
            debug_albedo = debug_albedo.astype(np.uint8)
            plt.imshow(debug_albedo)
            plt.show()
            print(debug_albedo.shape)
        """        
        print('=> Building model')

        self.netG = G(self.opt.n_channel_input*4, self.opt.n_channel_output, self.opt.n_generator_filters)
        self.netG.apply(weights_init)
        self.netD = D(self.opt.n_channel_input*4, self.opt.n_channel_output, self.opt.n_discriminator_filters)
        self.netD.apply(weights_init)

        self.criterion = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()

        self.label = torch.FloatTensor(self.opt.train_batch_size)
        self.real_label = 1
        self.fake_label = 0

        self.albedo = torch.FloatTensor(self.opt.train_batch_size, self.opt.n_channel_input, 256, 256)
        self.direct = torch.FloatTensor(self.opt.train_batch_size, self.opt.n_channel_input, 256, 256)
        self.normal = torch.FloatTensor(self.opt.train_batch_size, self.opt.n_channel_input, 256, 256)
        self.depth = torch.FloatTensor(self.opt.train_batch_size, self.opt.n_channel_input, 256, 256)
        self.gt = torch.FloatTensor(self.opt.train_batch_size, self.opt.n_channel_output, 256, 256)

        # GPU
        self.netD = self.netD.cuda()
        self.netG = self.netG.cuda()
        self.criterion = self.criterion.cuda()
        self.criterion_l1 = self.criterion_l1.cuda()

        self.albedo = self.albedo.cuda()
        self.direct = self.direct.cuda()
        self.normal = self.normal.cuda()
        self.depth = self.depth.cuda()
        self.gt = self.gt.cuda()
        self.label = self.label.cuda()

        # Derivative
        self.albedo = Variable(self.albedo)
        self.direct = Variable(self.direct)
        self.normal = Variable(self.normal)
        self.depth = Variable(self.depth)
        self.gt = Variable(self.gt)
        self.label = Variable(self.label)

        # Optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        
        if self.opt.resume_G:
            if os.path.isfile(self.opt.resume_G):
                print("=> loading generator checkpoint '{}'".format(self.opt.resume_G))
                checkpoint = torch.load(self.opt.resume_G)
                self.lastEpoch = checkpoint['epoch']
                self.n_epoch = self.n_epoch - self.lastEpoch
                self.netG.load_state_dict(checkpoint['state_dict_G'])
                self.optimizerG.load_state_dict(checkpoint['optimizer_G'])
                print("=> loaded generator checkpoint '{}' (epoch {})".format(self.opt.resume_G, checkpoint['epoch']))

            else:
                print("=> no checkpoint found")

        if self.opt.resume_D:
            if os.path.isfile(self.opt.resume_D):
                print("=> loading discriminator checkpoint '{}'".format(self.opt.resume_D))
                checkpoint = torch.load(self.opt.resume_D)
                self.netD.load_state_dict(checkpoint['state_dict_D'])
                self.optimizerD.load_state_dict(checkpoint['optimizer_D'])
                print("=> loaded discriminator checkpoint '{}'".format(self.opt.resume_D))

    def train(self, epoch):
        for (i, images) in enumerate(self.train_data):
            self.netD.zero_grad()
            (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (images[0], images[1], images[2], images[3], images[4])
            self.albedo.data.resize_(albedo_cpu.size()).copy_(albedo_cpu)
            self.direct.data.resize_(direct_cpu.size()).copy_(direct_cpu)
            self.normal.data.resize_(normal_cpu.size()).copy_(normal_cpu)
            self.depth.data.resize_(depth_cpu.size()).copy_(depth_cpu)
            self.gt.data.resize_(gt_cpu.size()).copy_(gt_cpu)
            output = self.netD(torch.cat((self.albedo, self.direct, self.normal, self.depth, self.gt), 1))
            self.label.data.resize_(output.size()).fill_(self.real_label)
            err_d_real = self.criterion(output, self.label)
            err_d_real.backward()
            d_x_y = output.data.mean()
            fake_B = self.netG(torch.cat((self.albedo, self.direct, self.normal, self.depth), 1))
            output = self.netD(torch.cat((self.albedo, self.direct, self.normal, self.depth, fake_B.detach()), 1))
            self.label.data.resize_(output.size()).fill_(self.fake_label)
            err_d_fake = self.criterion(output, self.label)
            err_d_fake.backward()
            d_x_gx = output.data.mean()
            err_d = (err_d_real + err_d_fake) * 0.5
            self.optimizerD.step()

            self.netG.zero_grad()
            output = self.netD(torch.cat((self.albedo, self.direct, self.normal, self.depth, fake_B), 1))
            self.label.data.resize_(output.size()).fill_(self.real_label)
            err_g = self.criterion(output, self.label) + self.opt.lamda \
                * self.criterion_l1(fake_B, self.gt) 
            err_g.backward()
            d_x_gx_2 = output.data.mean()
            self.optimizerG.step()
            print ('=> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}/{:.4f}'.format(
                epoch,
                i,
                len(self.train_data),            
                err_d,
                err_g,
                d_x_y,
                d_x_gx,
                d_x_gx_2,
                ))

    def save_checkpoint(self, epoch):
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        if not os.path.exists(os.path.join("checkpoint", self.opt.dataset)):
            os.mkdir(os.path.join("checkpoint", self.opt.dataset))
        net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(self.opt.dataset, epoch)
        net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth".format(self.opt.dataset, epoch)
        torch.save({'epoch':epoch+1, 'state_dict_G': self.netG.state_dict(), 'optimizer_G':self.optimizerG.state_dict()}, net_g_model_out_path)
        torch.save({'state_dict_D': self.netD.state_dict(), 'optimizer_D':self.optimizerD.state_dict()}, net_d_model_out_path)
        print("Checkpoint saved to {}".format("checkpoint" + self.opt.dataset))

        if not os.path.exists("validation"):
            os.mkdir("validation")
        if not os.path.exists(os.path.join("validation", self.opt.dataset)):
            os.mkdir(os.path.join("validation", self.opt.dataset))

        for index, images in enumerate(self.val_data):
            (albedo_cpu, direct_cpu, normal_cpu, depth_cpu, gt_cpu) = (images[0], images[1], images[2], images[3], images[4])
            self.albedo.data.resize_(albedo_cpu.size()).copy_(albedo_cpu)
            self.direct.data.resize_(direct_cpu.size()).copy_(direct_cpu)
            self.normal.data.resize_(normal_cpu.size()).copy_(normal_cpu)
            self.depth.data.resize_(depth_cpu.size()).copy_(depth_cpu)
            out = self.netG(torch.cat((self.albedo, self.direct, self.normal, self.depth), 1))
            out = out.cpu()
            out_img = out.data[0]
            save_image(out_img,"validation/{}/{}_Fake.png".format(self.opt.dataset, index))
            save_image(gt_cpu[0],"validation/{}/{}_Real.png".format(self.opt.dataset, index))
            save_image(direct_cpu[0],"validation/{}/{}_Direct.png".format(self.opt.dataset, index))

    def run(self):
        for epoch in range(self.n_epoch):
            self.train(epoch + self.lastEpoch)
            if epoch % 1 == 0:
                self.save_checkpoint(epoch + self.lastEpoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepRendering-implemention')
    parser.add_argument('--dataset', required=True, help='output from unity')
    parser.add_argument('--train_batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size for testing')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of iterations')
    parser.add_argument('--n_channel_input', type=int, default=3, help='number of input channels')
    parser.add_argument('--n_channel_output', type=int, default=3, help='number of output channels')
    parser.add_argument('--n_generator_filters', type=int, default=64, help='number of initial generator filters')
    parser.add_argument('--n_discriminator_filters', type=int, default=64, help='number of initial discriminator filters')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--cuda', action='store_true', help='cuda')
    parser.add_argument('--resume_G', help='resume G')
    parser.add_argument('--resume_D', help='resume D')
    parser.add_argument('--workers', type=int, default=4, help='number of threads for data loader')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--lamda', type=int, default=100, help='L1 regularization factor')
    
    opt = parser.parse_args()
    cudnn.benchmark = True
    torch.cuda.manual_seed(opt.seed)
    
    root_dir = "dataset/"
    gan_gi = GanGlobalIllumination(opt, root_dir)
    gan_gi.run()