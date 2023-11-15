import torchvision
import torch
import torch.nn as nn
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
    def forward(self, input, latent):
        latent = self.latent_injection(latent)
        return self.dconv(input) + latent.view(latent.size(0), latent.size(1), 1, 1)
    

class DeformConvDownSample(nn.Module):
    def __init__(self, latent_size, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DeformConvDownSample, self).__init__()
        self.dconv = IdDeformConv(latent_size,in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x, latent_id):
        x = self.dconv(x, latent_id)
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
        