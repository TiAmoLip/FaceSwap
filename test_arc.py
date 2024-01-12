import torch
import torch.nn.functional as F
from options.test_options import TestOptions
import os
from PIL import Image
from torch.utils.data import TensorDataset,DataLoader
from torchvision import transforms as T
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    opt = TestOptions().parse()
    cos_loss     = torch.nn.CosineSimilarity()

    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean= torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)

    ArcFace      = torch.load('./arcface_model/arcface_checkpoint.tar', map_location=torch.device("cpu"))
    ArcFace.eval()
    ArcFace.requires_grad_(False)

    c_transforms = []
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    #simgs = []
    #rimgs = []
    imgs = []
    for i in range(20):
        id = str(i)
        s = 'source_' + id + '.jpg'
        r = 'result_' + id + '.jpg'
        t = 'target_' + id + '.jpg'
        rr='r_result_'+ id + '.jpg'
        #simgs.append(c_transforms(Image.open(os.path.join(opt.output_path,s)).convert('RGB')))
        #rimgs.append(c_transforms(Image.open(os.path.join(opt.output_path,r)).convert('RGB')))
        imgs.append((c_transforms(Image.open(os.path.join(opt.output_path,s)).convert('RGB'))\
                    ,c_transforms(Image.open(os.path.join(opt.output_path,r)).convert('RGB'))\
                    ,c_transforms(Image.open(os.path.join(opt.output_path,t)).convert('RGB'))
                    ,c_transforms(Image.open(os.path.join(opt.output_path,rr)).convert('RGB'))))

    cosdis = []
    rcosdis = []
    for source_img,result_img,target_img,r_result_img in imgs:
        source_img   = (source_img - imagenet_mean)/imagenet_std                 # source_img~[0,1]
        source_img   = F.interpolate(torch.unsqueeze(source_img,0), size=(112,112), mode='bicubic')
        source_id    = ArcFace(source_img)                                       # ArcFace model
        source_id    = F.normalize(source_id, p=2, dim=1)

        result_img   = (result_img - imagenet_mean)/imagenet_std                 # result_img~[0,1]
        result_img   = F.interpolate(torch.unsqueeze(result_img,0), size=(112,112), mode='bicubic')
        result_id    = ArcFace(result_img)
        result_id    = F.normalize(result_id, p=2, dim=1)

        cos_dis      = 1 -  cos_loss(source_id, result_id)                       # cosine distance between source image and result image
        cosdis.append(cos_dis.numpy())
        source_img,result_img = target_img,r_result_img
        source_img   = (source_img - imagenet_mean)/imagenet_std                 # source_img~[0,1]
        source_img   = F.interpolate(torch.unsqueeze(source_img,0), size=(112,112), mode='bicubic')
        source_id    = ArcFace(source_img)                                       # ArcFace model
        source_id    = F.normalize(source_id, p=2, dim=1)

        result_img   = (result_img - imagenet_mean)/imagenet_std                 # result_img~[0,1]
        result_img   = F.interpolate(torch.unsqueeze(result_img,0), size=(112,112), mode='bicubic')
        result_id    = ArcFace(result_img)
        result_id    = F.normalize(result_id, p=2, dim=1)

        cos_dis      = 1 -  cos_loss(source_id, result_id)                       # cosine distance between source image and result image
        rcosdis.append(cos_dis.numpy())
    cosdis = np.array(cosdis)
    rcosdis= np.array(rcosdis)
    print(np.mean(cosdis))
    print(np.mean(rcosdis))
    print(np.mean(rcosdis+cosdis)/2)

    plt.plot(cosdis,c='r')
    plt.plot(rcosdis,c='b')
    plt.show()

    #不切：0.24381888