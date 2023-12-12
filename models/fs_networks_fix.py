"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x




class Generator_Adain_Upsample(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size, n_blocks=6, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        
        super(Generator_Adain_Upsample, self).__init__()
        assert (n_blocks >= 0)
        activation = nn.ReLU(True)
        
        self.deep = deep
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
                                   
                                   

        if self.deep:
            self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                       norm_layer(512), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents):
        x = input  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)
            bot.append(x)

        if self.deep:
            x = self.up4(x)
            features.append(x)
        x = self.up3(x)
        features.append(x)
        x = self.up2(x)
        features.append(x)
        x = self.up1(x)
        features.append(x)
        x = self.last_layer(x)
        # x = (x + 1) / 2

        # return x, bot, features, dlatents
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class AADBlock(nn.Module):
    def __init__(self,h_inchannel, z_inchannel, z_id_size=512) -> None:
        
        """
        This block will not change the size of the input, but h_in.shape[-2:] must be equal to z_att.shape[-2:], and the output channels is h_inchannel
        z_inchannel is the number of channels of the input z_att
        """
        
        super().__init__()
        self.norm = nn.InstanceNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)
        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self,h_in, z_att, z_id):
        h_bar = self.norm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))
        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)# [batch_size, h_inchannel, 1, 1]->[batch_size, h_inchannel, h, w]
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        i = r_id*h_bar + beta_id
        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        a = r_att * h_bar + beta_att
        h_out = (1-m)*a + m*i
        return h_out
    
class AADResBlock(nn.Module):
    def __init__(self,h_inchannel, z_inchannel, latent_size=512) -> None:
        super().__init__()
        
        self.add1 = AADBlock(h_inchannel, z_inchannel, z_id_size=latent_size)
        self.act = nn.LeakyReLU(0.2,True)
        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.add2 = AADBlock(h_inchannel, z_inchannel, z_id_size=latent_size)
        self.conv2 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.norm = nn.InstanceNorm2d(h_inchannel)
        
    def forward(self, h_in, z_att, z_id):
        x1 = self.add1(h_in, z_att, z_id)
        x1 = self.act(x1)
        x1 = self.conv1(x1)
        x1 = self.norm(x1)
        x1 = self.add2(x1, z_att, z_id)
        x1 = self.act(x1)
        x1 = self.conv2(x1)
        return x1 + h_in
    
    
# def make_layer(h_inchannel,z_inchannel,latent_size,init_AAD_layer=False):
#     if init_AAD_layer:
#         return AADResBlock(h_inchannel,z_inchannel,latent_size=latent_size)
        
class simplifiedGenerator(nn.Module):
    def __init__(self,input_layers = 5, output_layers= 5, latent_size=512, n_blocks=6,
                norm_layer=nn.InstanceNorm2d,deep=True,
                padding_type='reflect',init_channels=32) -> None:
        super().__init__()
        assert (n_blocks >= 0)
        # assert (input_layers <=5)
        self.deep = deep
        activation = nn.LeakyReLU(0.2,True)
        
        self.n_iteration = n_blocks//(input_layers-2) if input_layers>2 else 0
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(3, init_channels, kernel_size=7, padding=0),
                                        norm_layer(init_channels), activation)
        
        self.down = nn.ModuleDict()
        for i in range(3):
            self.down[f'layer_{i}'] = nn.Sequential(nn.Conv2d(init_channels*(2**i), init_channels*(2**(i+1)), kernel_size=3, stride=2, padding=1),
                                    norm_layer(init_channels*(2**(i+1))), activation)
        if deep:
            self.down[f'layer_{3}'] = nn.Sequential(nn.Conv2d(init_channels*(2**3), init_channels*(2**3), kernel_size=3, stride=2, padding=1),
                                    norm_layer(init_channels*(2**3)), activation)
        else:
            self.down[f'layer_{3}'] = nn.Identity()
        # 没有想好这块stride=1还是2
        for i in range(3+self.deep,input_layers+self.deep):
            self.down[f'layer_{i}'] = nn.Sequential(nn.Conv2d(init_channels*(2**3), init_channels*(2**3), kernel_size=3, stride=1, padding=1),
                                    norm_layer(init_channels*(2**3)), activation)
        
        # BN = []
        # for i in range(n_blocks):
        #     BN += [
        #         ResnetBlock_Adain(256, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        # self.BottleNeck = nn.Sequential(*BN)
        self.aads = nn.ModuleDict()
        for i in range(self.n_iteration*(input_layers-2)):
            #
            self.aads.add_module(f'layer_{i}',AADResBlock(init_channels*(2**3),init_channels*(2**3),latent_size=latent_size))
        
        
        self.up = nn.ModuleDict()
        # for i in range(4,output_layers):
        #     self.up[f'layer_{i}'] = nn.Sequential(
        #         # nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
        #         nn.Conv2d(init_channels*(2**3),init_channels*(2**3) , kernel_size=3, stride=1, padding=1),
        #         nn.InstanceNorm2d(init_channels*(2**3)), activation
        #     )
        if deep:
            self.up[f'layer_{3}'] = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(init_channels*(2**3),init_channels*(2**3) , kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(init_channels*(2**3)), activation
            )
        for i in range(3):
            self.up[f'layer_{i}'] = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(init_channels*(2**(i+1)), init_channels*(2**i), kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(init_channels*(2**i)), activation
            )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(init_channels, 3, kernel_size=7, padding=0))
        self.input_layers = input_layers
        self.output_layers = output_layers
    def forward(self, x, latents):
        x = self.first_layer(x)
        x_attrs = []
        for i in range(len(self.down)):
            if i>=self.input_layers-2 and i>=3+self.deep:
                x_attrs.append(x)
            x = self.down[f'layer_{i}'](x)
        # for i in range(len(self.BottleNeck)):
        #     x = self.BottleNeck[i](x, latents)
        
        for i in range(self.n_iteration):
            for j in range(self.input_layers-2):
                x = self.aads[f'layer_{i*(self.input_layers-2)+j}'](x,x_attrs[i],latents)
        
        
        for i in reversed(range(4 if self.deep else 3)):
            x = self.up[f'layer_{i}'](x)
        x = self.last_layer(x)
        return x

# model = AADBlock(512,256,512)
# t = torch.randn(1,512,4,4)
# x = torch.randn(1,256,4,4)
# l = torch.randn(1,512)
# print(model(t,x,l).shape)
# model = simplifiedGenerator(5,5,512,deep=False)
# t = torch.randn(1,3,224,224)
# l = torch.randn(1,512)
# print(model(t,l).shape)