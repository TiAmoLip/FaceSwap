#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: fs_model_fix_idnorm_donggp_saveoptim copy.py
# Created Date: Wednesday January 12th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Saturday, 13th May 2023 9:56:35 am
# Modified By: Chen Xuanhong
# Copyright (c) 2022 Shanghai Jiao Tong University
#############################################################


import torch
import torch.nn as nn
from torch.autograd import Variable
from .base_model import BaseModel
from .fs_networks_fix import Generator_Adain_Upsample
from .custom_network import DeformConvGenerator,DancerGenerator
from pg_modules.projected_discriminator import ProjectedDiscriminator

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

class fsModel(BaseModel):
    def name(self):
        return 'fsModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        # if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM
        self.isTrain = opt.isTrain

        if opt.model_name=="simswap":
            model_k = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=opt.n_blocks, deep=opt.Gdeep)
        elif opt.model_name=="simswap+=+":
            self.netG = DeformConvGenerator(opt.n_layers, opt.n_layers, latent_size=512, n_blocks=opt.n_blocks)
        elif opt.model_name=="dancer":
            self.netG = DancerGenerator(opt.n_layers, opt.n_layers, latent_size=512, n_blocks=opt.n_blocks, kernel_type=opt.kernel_type)
        else:
            self.netG = None
        
        #     model_k = DeformConvGenerator
        # elif opt.model_name=="dancer":
        #     model_k = DancerGenerator
        # else:
        #     model_k = None
        # # Generator network
        # if opt.model_name!="dancer":
        #     self.netG = model_k(input_nc=3, output_nc=3, latent_size=512, n_blocks=opt.n_blocks, deep=opt.Gdeep)
        # else:
        #     self.netG = model_k(input_nc=3, output_nc=3, latent_size=512, n_blocks=opt.n_blocks, n_layers=opt.n_layers, deep=opt.Gdeep,kernel_type=opt.kernel_type)
        self.netG.cuda()

        # Id network
        netArc_checkpoint = opt.Arc_path
        netArc_checkpoint = torch.load(netArc_checkpoint, map_location=torch.device("cpu"))
        self.netArc = netArc_checkpoint
        self.netArc = self.netArc.cuda()
        self.netArc.eval()
        self.netArc.requires_grad_(False)
        if not self.isTrain:
            pretrained_path =  opt.checkpoints_dir
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            return
        self.netD = ProjectedDiscriminator(diffaug=False, interp224=False, **{})
        # self.netD.feature_network.requires_grad_(False)
        self.netD.cuda()


        if self.isTrain:
            # define loss functions
            self.criterionFeat  = nn.L1Loss()
            self.criterionRec   = nn.L1Loss()


           # initialize optimizers

            # optimizer G
            params = list(self.netG.parameters())
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

            # optimizer D
            params = list(self.netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.99),eps=1e-8)

        # load networks
        if opt.continue_train:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print (pretrained_path)
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_G, 'G', opt.which_epoch, pretrained_path)
            self.load_optim(self.optimizer_D, 'D', opt.which_epoch, pretrained_path)
        torch.cuda.empty_cache()

    def cosin_metric(self, x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))



    def save(self, which_epoch, overlap = False):
        self.save_network(self.netG, 'G', which_epoch if not overlap else 0)
        self.save_network(self.netD, 'D', which_epoch if not overlap else 0)
        self.save_optim(self.optimizer_G, 'G', which_epoch if not overlap else 0)
        self.save_optim(self.optimizer_D, 'D', which_epoch if not overlap else 0)
        '''if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)'''

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
    
    def gradinet_penalty_D(self, netD, img_att, img_fake):
        # interpolate sample
        bs = img_fake.shape[0]
        alpha = torch.rand(bs, 1, 1, 1).expand_as(img_fake).cuda()
        interpolated = Variable(alpha * img_att + (1 - alpha) * img_fake, requires_grad=True)
        pred_interpolated = netD.forward(interpolated,None)
        pred_interpolated = pred_interpolated[0]

        # compute gradients
        grad = torch.autograd.grad(outputs=pred_interpolated,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(pred_interpolated.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        loss_d_gp = torch.mean((grad_l2norm - 1) ** 2)

        return loss_d_gp

