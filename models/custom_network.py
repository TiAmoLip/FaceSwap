import torchvision
import torch
import torch.nn as nn
from .fs_networks_fix import InstanceNorm,ResnetBlock_Adain
from .fs_networks_fix import ApplyStyle as AdaIn

def count_parameters(model:nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DeformConv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(kernel_size, int) or isinstance(kernel_size, tuple)
        assert isinstance(stride, int) or isinstance(stride, tuple)
        assert isinstance(padding, int) or isinstance(padding, tuple)
        assert isinstance(bias, bool)
        assert isinstance(input_channels, int)
        assert isinstance(output_channels, int)
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias
        self.conv_offset = nn.Conv2d(input_channels, 2*self.kernel_size[0]*self.kernel_size[1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)
        self.deform_conv = torchvision.ops.deform_conv.DeformConv2d(input_channels, output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)


    def forward(self, input):
        offsets = self.conv_offset(input)
        return self.deform_conv(input, offsets)
class IdDeformConv(nn.Module):
    def __init__(self, latent_size, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=False) -> None:
        super(IdDeformConv,self).__init__()
        # self.latent_size = latent_size
        self.dconv = DeformConv(input_channels, output_channels, kernel_size, stride, padding, bias)
        self.latent_injection = nn.Linear(latent_size, output_channels)
        # self.res = 
    def forward(self, input, latent=None):
        if latent == None:
            return self.dconv(input)
        latent = self.latent_injection(latent)
        return self.dconv(input) + latent.view(latent.size(0), latent.size(1), 1, 1)
    

class DeformConvDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DeformConvDownSample, self).__init__()
        self.dconv = DeformConv(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x = self.dconv(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class DeformConvUpSample(nn.Module):
    def __init__(self, scaleFactor,latent_size, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.upsample = nn.Upsample(scale_factor=scaleFactor, mode='bilinear',align_corners=False)
        self.IdDeformConv = IdDeformConv(latent_size,in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.rl = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, x, latent_id):
        x = self.upsample(x)
        x = self.IdDeformConv(x, latent_id)
        x = self.conv(x)
        x = self.norm(x)
        x = self.rl(x)
        return x
    
class AFFAModule(nn.Module):
    """
    perfrom attention fusion feature aggregation
    
    input_shape: (batch_size, in_channels, height, width)
    
    output_shape: (batch_size, in_channels, height, width)
    
    forward_args:
        h: hidden feature from last layer
        z: feature map in forward process
    theoretically, h and z should have the same shape
    """
    def __init__(self, in_channels, kernel_type = "ordinary") -> None:
        super().__init__()
        assert kernel_type in ["ordinary","deform"], "kernel type must be ordinary or deform"
        # self.linproj = nn.Linear(latent_size, in_channels)
        if kernel_type == "ordinary":
            self.conv1 = nn.Conv2d(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv1 = DeformConv(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            self.conv2 = DeformConv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.norm = InstanceNorm()
        self.act = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, h, z):

        x = torch.concat([h, z], dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        
        return x * h + (1 - x) * z


class AFFA_RB(nn.Module):
    """
    This is the adaptive feature fusion attention residual block in FaceDancer.
    
    input_shape: (batch_size, in_channels, height, width)
    
    output_shape: (batch_size, output_channels, height, width)
    
    The output_channels can be 1/2 or 2 times of in_channels depending on sample method.
    
    forward args:
        h: hidden feature from last layer
        z: feature map in encoder process
        w: id projection vector
    """
    def __init__(self, latent_size, in_channels, out_channels, kernel_type="ordinary", sample_method="down", upsample_method="ordinary") -> None:
        super().__init__()
        assert sample_method in ["down","up","none"], "sample method must be down, none or up"
        assert kernel_type in ["ordinary","deform"], "convolution kernel type must be ordinary or deform"
        assert upsample_method in ["ordinary","convolution"], "sample kernel type must be ordinary or convolution"
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1) if kernel_type == "ordinary" else DeformConv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False) if sample_method == "up" else nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        if sample_method == "none":
            self.sample = nn.Identity()
        if upsample_method == "convolution" and sample_method == "up":
            self.sample = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,stride=2,padding=1,output_padding=1)
        
        self.affa = AFFAModule(in_channels, kernel_type)
        self.adain = AdaIn(latent_size, in_channels)
        self.act = nn.LeakyReLU(0.2,inplace=True)
    def forward(self,h,z,w):
        h = self.affa(h,z)
        x = self.adain(h,w)
        x = self.act(x)
        x = self.conv1(x)
        x = self.sample(x)
        h = self.conv2(h)
        h = self.sample(h)
        return x + h
        
        
class DancerGeneratorEncoder(nn.Module):
    def __init__(self, input_nc=3) -> None:
        super().__init__()
        initial_channels = 64
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, initial_channels, kernel_size=7, padding=0))
        # self.down = nn.ModuleList()
        # self.down1 = DeformConvDownSample(initial_channels, initial_channels*2, kernel_size=3, stride=2, padding=1)
        # self.down2 = DeformConvDownSample(initial_channels*2, initial_channels*4, kernel_size=3, stride=2, padding=1)
        # self.down3 = DeformConvDownSample(initial_channels*4, initial_channels*8, kernel_size=3, stride=2, padding=1)
        self.down = nn.ModuleList()
        for i in range(3):
            self.down.append(DeformConvDownSample(initial_channels*(2**i),initial_channels*(2**(i+1)),kernel_size=3,stride=2,padding=1))
    def forward(self,x):
        features = []
        x = self.first_layer(x)
        for i,layer in enumerate(self.down):
            x = layer(x)
            features.append(x)
        features.reverse()# ensure that the first element is the output last layer
        return x
    
class DancerGeneratorDecoder(nn.Module):
    def __init__(self,latent_size, kernel_type, upsample_method) -> None:
        super().__init__()
        final_channels = 512
        self.up = nn.ModuleList()
        for i in range(3):
            self.up.append(AFFA_RB(latent_size=latent_size,in_channels=final_channels//(2**i),out_channels=final_channels//(2**(i+1)),sample_method="up",upsample_method=upsample_method,kernel_type=kernel_type))
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(final_channels//8, 3, kernel_size=7, padding=0))

    def forward(self,x,hidden_list,z_id):
        for hidden, module in zip(hidden_list,self.up):
            x = module(x,hidden,z_id)
        x = self.last_layer(x)
        return x

class DancerGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, latent_size=512, n_blocks=6, deep=False,norm_layer=nn.BatchNorm2d,padding_type='reflect',
                 kernel_type="ordinary",upsample_method="ordinary") -> None:
        assert (n_blocks >= 0)
        super(DancerGenerator, self).__init__()
        self.deep = deep
        
        self.enc = DancerGeneratorEncoder(input_nc)
        
        BN = []
        activation = nn.LeakyReLU(0.2,True)
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)
        
        self.transition = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        
        # self.affa3 = AFFA_RB(latent_size = 512,in_channels = 512,out_channels= 256, sample_method="up",kernel_type=kernel_type,upsample_method=upsample_method)
        # self.affa2 = AFFA_RB(latent_size= 512,in_channels= 256,out_channels= 128, sample_method="up",kernel_type=kernel_type,upsample_method=upsample_method)
        # self.affa1 = AFFA_RB(latent_size = 512,in_channels= 128,out_channels= 64, sample_method="up",kernel_type=kernel_type,upsample_method=upsample_method)
        self.dec = DancerGeneratorDecoder(latent_size,kernel_type,upsample_method)
        
        self.norm = InstanceNorm()
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(64, output_nc, kernel_size=7, padding=0))
    def forward(self, x, latent):
        # x: (batch_size, 3, 224, 224)
        # x = self.first_layer(x)# (batch_size, 64, 224, 224)
        
        # skip1 = self.down1(skip) # (batch_size, 128, 112, 112)
        # skip2 = self.down2(skip1) # (batch_size, 256, 56, 56)
        # skip3 = self.down3(skip2) # (batch_size, 512, 28, 28)
        
        x, features = self.enc(x) # (batch_size, 512, 28, 28)
        
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, latent)
        
        x = self.transition(x) # (batch_size, 512, 14, 14) if self.deep else (batch_size, 512, 28, 28)
        
        
        x = self.dec(x,features,latent)
        
        # x,m = self.affa3(x,skip3,latent)# (batch_size, 256, 56, 56)
        # x,m = self.affa2(x,skip2,latent) # (batch_size, 128, 112, 112)
        # x,m = self.affa1(x,skip1,latent) # (batch_size, 64, 224, 224)
        # x = self.last_layer(x) # (batch_size, 3, 224, 224)
        return x

class DeformConvGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, latent_size=512, n_blocks=6, deep=False,norm_layer=nn.BatchNorm2d,padding_type='reflect') -> None:
        assert (n_blocks >= 0)
        super(DeformConvGenerator, self).__init__()
        self.deep = deep
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), nn.ReLU(True))
        ### downsample
        self.down1 = DeformConvDownSample(latent_size, 64, 128, kernel_size=3, stride=2, padding=1)
        self.down2 = DeformConvDownSample(latent_size, 128, 256, kernel_size=3, stride=2, padding=1)
        self.down3 = DeformConvDownSample(latent_size, 256, 512, kernel_size=3, stride=2, padding=1)

        if self.deep:
            self.down4 = DeformConvDownSample(latent_size, 512, 512, kernel_size=3, stride=2, padding=1)

        ### resnet blocks
        BN = []
        activation = nn.LeakyReLU(0.2,True)
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = DeformConvUpSample(2,latent_size, 512, 512, kernel_size=3, stride=1, padding=1)
        self.up3 = DeformConvUpSample(2,latent_size, 512, 256, kernel_size=3, stride=1, padding=1)
        self.up2 = DeformConvUpSample(2,latent_size, 256, 128, kernel_size=3, stride=1, padding=1)
        self.up1 = DeformConvUpSample(2,latent_size, 128, 64, kernel_size=3, stride=1, padding=1)

        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(64, output_nc, kernel_size=7, padding=0))

    def forward(self, input, dlatents):
        x = input  # 3*224*224

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1,dlatents)
        skip3 = self.down2(skip2,dlatents)
        if self.deep:
            skip4 = self.down3(skip3,None)
            x = self.down4(skip4,None)
        else:
            x = self.down3(skip3,None)
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, dlatents)

        if self.deep:
            x = self.up4(x,None)    
        x = self.up3(x,None)
        x = self.up2(x,dlatents)
        x = self.up1(x,dlatents)
        x = self.last_layer(x)
        return x

