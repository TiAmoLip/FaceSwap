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
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.bias = bias
        self.conv_offset = nn.Conv2d(input_channels, 2*self.kernel_size[0]*self.kernel_size[1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)
        self.deform_conv = torchvision.ops.deform_conv.DeformConv2d(input_channels, output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=bias)


    def forward(self, input):
        offsets = self.conv_offset(input)
        return self.deform_conv(input, offsets)
# class IdDeformConv(nn.Module):
#     def __init__(self, latent_size, input_channels, output_channels, kernel_size, stride=1, padding=0, bias=False) -> None:
#         super(IdDeformConv,self).__init__()
#         # self.latent_size = latent_size
#         self.dconv = DeformConv(input_channels, output_channels, kernel_size, stride, padding, bias)
#         self.latent_injection = nn.Linear(latent_size, output_channels)
#         # self.res = 
#     def forward(self, input, latent=None):
#         if latent == None:
#             return self.dconv(input)
#         latent = self.latent_injection(latent)
#         return self.dconv(input) + latent.view(latent.size(0), latent.size(1), 1, 1)
    

class DeformConvDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, kernel_type="ordinary"):
        super(DeformConvDownSample, self).__init__()
        assert kernel_type in ["ordinary","deform"], "kernel type must be ordinary or deform"
        Conv = DeformConv if kernel_type == "deform" else nn.Conv2d
        self.dconv = Conv(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2,inplace=True)

    def forward(self, x):
        x = self.dconv(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x
    
class DeformConvUpSample(nn.Module):
    def __init__(self, scaleFactor, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False,kernel_type='ordinary') -> None:
        super().__init__()
        assert kernel_type in ["ordinary","deform"], "kernel type must be ordinary or deform"
        self.upsample = nn.Upsample(scale_factor=scaleFactor, mode='bilinear',align_corners=False) if scaleFactor > 1 else nn.Identity()
        Conv = DeformConv if kernel_type == "deform" else nn.Conv2d
        self.DeformConv = Conv(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.rl = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, x):
        x = self.upsample(x)
        x = self.DeformConv(x)
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
        Conv = DeformConv if kernel_type == "deform" else nn.Conv2d
        
        
        self.conv1 = Conv(2*in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
    
        self.norm = InstanceNorm()
        self.act = nn.LeakyReLU(0.2,inplace=True)
    def forward(self, h, z):

        x = torch.concat([h, z], dim=1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.conv2(x)
        # return x
        x = torch.sigmoid(x)
        
        return (1 + x)/2 * h + (1 - x)/2 * z


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
    def __init__(self, latent_size, in_channels, out_channels, kernel_type="ordinary", sample_method="down") -> None:
        super().__init__()
        assert sample_method in ["down","up","none"], "sample method must be down, none or up"
        assert kernel_type in ["ordinary","deform"], "convolution kernel type must be ordinary or deform"
        
        Conv = DeformConv if kernel_type == "deform" else nn.Conv2d        
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.sample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False) if sample_method == "up" else nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        
        if sample_method == "none":
            self.sample = nn.Identity()

        
        self.affa = AFFAModule(in_channels, kernel_type)
        self.adain = AdaIn(latent_size, in_channels)
        self.act = nn.LeakyReLU(0.2,inplace=True)
    def forward(self,h,z,w):
        t = self.affa(h,z)
        x = self.adain(t,w)
        
        x = self.act(x)
        x = x + h
        x = self.conv1(x)
        x = self.sample(x)
        # h = self.conv2(h)
        # h = self.sample(h)
        return x
        
        
class DancerGeneratorEncoder(nn.Module):
    def __init__(self, input_nc=3,n_layers=3) -> None:
        super().__init__()
        initial_channels = 64
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, initial_channels, kernel_size=7, padding=0))
        self.down = nn.ModuleList()
        for i in range(3):
            self.down.append(DeformConvDownSample(initial_channels*(2**i),initial_channels*(2**(i+1)),kernel_size=3,stride=2,padding=1))
        for i in range(n_layers-3):
            self.down.append(DeformConvDownSample(512,512,kernel_size=3,stride=1,padding=1))
    def forward(self,x):
        features = []
        x = self.first_layer(x)
        for i in range(len(self.down)):
            x = self.down[i](x)
            features.append(x)
        # for i in range()
        features.reverse()# ensure that the first element is the output last layer
        return x,features
    
class DancerGeneratorDecoder(nn.Module):
    def __init__(self,latent_size, n_layers, kernel_type) -> None:
        super().__init__()
        final_channels = 512
        self.up = nn.ModuleList()
        for i in range(n_layers-3):
            self.up.append(AFFA_RB(latent_size=latent_size,in_channels=512,out_channels=512,sample_method="none",kernel_type=kernel_type))
        for i in range(3):
            self.up.append(AFFA_RB(latent_size=latent_size,in_channels=final_channels//(2**i),out_channels=final_channels//(2**(i+1)),sample_method="up",kernel_type=kernel_type))
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(final_channels//8, 3, kernel_size=7, padding=0))

    def forward(self,x,hidden_list,z_id):
        for hidden, module in zip(hidden_list,self.up):
            x = module(x,hidden,z_id)
        x = self.last_layer(x)
        return x

class DancerGenerator(nn.Module):
    def __init__(self, enc_layers, dec_layers, latent_size=512, n_blocks=6, norm_layer=InstanceNorm(),padding_type='reflect',
                 kernel_type="ordinary") -> None:
        assert (n_blocks >= 0)
        super(DancerGenerator, self).__init__()

        
        self.latent_project = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, latent_size),
            nn.LeakyReLU(True),
        )
        
        self.enc = DancerGeneratorEncoder(3, enc_layers)
        self.enc_norm = norm_layer
        BN = []
        activation = nn.LeakyReLU(0.2,True)
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)
        
        # self.transition = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.dec_norm = norm_layer
        
        self.dec = DancerGeneratorDecoder(latent_size, dec_layers ,kernel_type)
        
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(64, 3, kernel_size=7, padding=0))
    def forward(self, x, latent):
        # x: (batch_size, 3, 224, 224)
        # x = self.first_layer(x)# (batch_size, 64, 224, 224)
        
        x, features = self.enc(x) # (batch_size, 512, 28, 28)
        x = self.enc_norm.forward(x)
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, latent)
        
        # latent = self.latent_project(latent)
        
        x = self.dec_norm.forward(x)
        x = self.dec(x,features,latent)
        
        return x
    
class FeatureFusion(nn.Module):
    def __init__(self, input_channels = 512, feature_layers = 3) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels*feature_layers, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(input_channels*feature_layers, input_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(input_channels*feature_layers, input_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.norm = InstanceNorm()
        self.act = nn.LeakyReLU(0.2,inplace=True)
        self.final_conv = nn.Conv2d(input_channels*3, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.fl = feature_layers
        
    def forward(self, base_feature:torch.Tensor, advance_feature:list[torch.Tensor]):
        if self.fl == 1:
            return base_feature
        features = torch.cat([base_feature] + advance_feature, dim=1)
        f1 = self.conv1(features)
        f2 = self.conv2(features)
        f3 = self.conv3(features)
        features = torch.cat([f1,f2,f3], dim=1)
        features = self.norm(features)
        features = self.act(features)
        features = self.final_conv(features)
        return features


class DeformConvGenerator(nn.Module):
    def __init__(self, enc_layers, dec_layers, latent_size=512, n_blocks=3,norm_layer=InstanceNorm,padding_type='reflect',kernel_type = "ordinary") -> None:
        assert (n_blocks >= 0)
        super(DeformConvGenerator, self).__init__()
        initial_channels = 64
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(3, 64, kernel_size=7, padding=0),
                                         norm_layer(), nn.ReLU(True))
        ### downsample
        
        self.down = nn.ModuleList()
        for i in range(3):
            self.down.append(DeformConvDownSample(initial_channels*(2**i),initial_channels*(2**(i+1)),kernel_size=3,stride=2,padding=1))
        for i in range(enc_layers-3):
            self.down.append(DeformConvDownSample(512,512,kernel_size=3,stride=1,padding=1))
        # 由于我前面输入参数的时候就让enc_layers和dec_layers
        # 所以有了这个:
        
        self.baseBN = nn.ModuleList()
        activation = nn.LeakyReLU(0.2,True)
        for i in range(n_blocks):
            self.baseBN.append(ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation))
        
        self.furtherBN = nn.ModuleDict()
        for i in range(enc_layers-3):
            self.furtherBN[f"{i}"] = nn.ModuleList()
            for j in range(n_blocks):
                self.furtherBN[f"{i}"].append(ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation))
        

        self.fusion = FeatureFusion(512, enc_layers-2)


        # 我原本意思是，他这个每个层都输出了特征，每个层过自己的id block，直接融合，decoder不整花活了。
        # self.up = nn.ModuleDict()

        up = []
        for i in range(dec_layers-3):
            up.append(DeformConvUpSample(in_channels=512,out_channels=512,scaleFactor=1,kernel_size=3,stride=1,padding=1))

        for i in range(3):
            up.append(DeformConvUpSample(in_channels=512//(2**i),out_channels=512//(2**(i+1)),scaleFactor=2,kernel_size=3,stride=1,padding=1))
        self.up = nn.Sequential(*up)
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), DeformConv(64, 3, kernel_size=7, padding=0))

    def forward(self, input:torch.Tensor, dlatents):
        x = input  # 3*224*224
        x = self.first_layer(x)
        # x = self.down(x)
        for i in range(len(self.down)):
            x = self.down[i](x)


        for i in range(len(self.baseBN)):
            x = self.baseBN[i](x, dlatents)
        base_feature = x.clone()
        
        advance_feature = []
        
        for i in range(len(self.furtherBN)):
            t = x.clone()
            for j in range(len(self.furtherBN[f"{i}"])):
                t = self.furtherBN[f"{i}"][j].forward(t, dlatents)
            advance_feature.append(t)

        x = self.fusion(base_feature, advance_feature)

        x = self.up(x)
        x = self.last_layer(x)
        return x

