import os
import cv2
import torch
import fractions
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from test_data import GetLoader
from util.plot import plot_batch
from insightface_func.face_detect_crop_single import Face_detect_crop
#import pynvml
#pynvml.nvmlInit()

if __name__ == '__main__':
    #handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    opt = TestOptions().parse()
    start_epoch, epoch_iter = 1, 0
    imagenet_std    = torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1)
    imagenet_mean   = torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1)
    #torch.nn.Module.dump_patches = True
    model = create_model(opt)
    model.eval()
    model.netG.eval()
    opt.batchSize = 1
    train_loader    = GetLoader(opt.test_root,opt.batchSize,0,1234,False)
    for o in range(len(train_loader)):
        #if o :
        #    del src_image1, src_image2, idx, arcface_112, id_vector_src1, img_fake, x
        #torch.cuda.empty_cache()
        #if o in [0,1,2,3]:
        #    continue
        #memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        #print("Total memory:", memory_info.total)
        #print(f"pic {o} Free memory:", memory_info.free)
        #print(f"pic {o} Used memory:", memory_info.used)
        src_image1, src_image2, idx  = train_loader.next()
        with torch.no_grad():
            #imgs        = list()
            #zero_img    = (torch.zeros_like(src_image1[0,...]))
            #imgs.append(zero_img.cpu().numpy())
            #save_img    = ((src_image1.cpu())* imagenet_std + imagenet_mean).numpy()
            #for r in range(idx.size()[0]):
            #    x = save_img[r].transpose(1,2,0)
            #    x = np.clip(255 * x, 0, 255)
            #    x = np.cast[np.uint8](x)
            #    Image.fromarray(x).save(os.path.join(opt.output_path, 'source_'+str((idx[r].numpy()))+'.jpg'))
            #save_img    = ((src_image2.cpu())* imagenet_std + imagenet_mean).numpy()
            #for r in range(idx.size()[0]):
            #    x = save_img[r].transpose(1,2,0)
            #    x = np.clip(255 * x, 0, 255)
            #    x = np.cast[np.uint8](x)
            #    Image.fromarray(x).save(os.path.join(opt.output_path, 'target_'+str((idx[r].numpy()))+'.jpg'))
            ##print(src_image2.size())
            arcface_112     = F.interpolate(src_image2,size=(112,112), mode='bicubic')
            id_vector_src1  = model.netArc(arcface_112)
            id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)
            for i in range(idx.size()[0]):
                
                #imgs.append(save_img[i,...])
                image_infer = src_image1[i, ...].repeat(idx.size()[0], 1, 1, 1)
                #print(f"pic {o} Free memory:", memory_info.free)
                #print(f"pic {o} Used memory:", memory_info.used)
                img_fake    = model.netG(image_infer, id_vector_src1).cpu()
                #print(f"pic {o} Free memory:", memory_info.free)
                #print(f"pic {o} Used memory:", memory_info.used)
                #img_fake    = F.interpolate(img_fake,size=imgs[0].shape[1:], mode='bicubic')
                img_fake    = img_fake * imagenet_std
                img_fake    = img_fake + imagenet_mean
                img_fake    = img_fake.numpy()
                for j in range(idx.size()[0]):
                    #imgs.append(img_fake[j,...])
            #print("Save test data")
            #print(imgs[1].shape,imgs[0].shape)
            #for img in imgs:
            #    print(img.shape)
            #imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
            #plot_batch(imgs, os.path.join(opt.output_path, 'result'+'.jpg'))
            #cv2.imwrite(opt.output_path + 'result.jpg', img_fake[0].transpose(1,2,0)*255)
            #print(len(imgs))
                    x = img_fake[j].transpose(1,2,0)
                    x = np.clip(255 * x, 0, 255)
                    x = np.cast[np.uint8](x)
                    Image.fromarray(x).save(os.path.join(opt.output_path, 'result_'+str((idx[j].numpy()))+'.jpg'))
            arcface_112     = F.interpolate(src_image1,size=(112,112), mode='bicubic')
            id_vector_src1  = model.netArc(arcface_112)
            id_vector_src1  = F.normalize(id_vector_src1, p=2, dim=1)
            for i in range(idx.size()[0]):
                
                #imgs.append(save_img[i,...])
                image_infer = src_image2[i, ...].repeat(idx.size()[0], 1, 1, 1)
                #print(f"pic {o} Free memory:", memory_info.free)
                #print(f"pic {o} Used memory:", memory_info.used)
                img_fake    = model.netG(image_infer, id_vector_src1).cpu()
                #print(f"pic {o} Free memory:", memory_info.free)
                #print(f"pic {o} Used memory:", memory_info.used)
                #img_fake    = F.interpolate(img_fake,size=imgs[0].shape[1:], mode='bicubic')
                img_fake    = img_fake * imagenet_std
                img_fake    = img_fake + imagenet_mean
                img_fake    = img_fake.numpy()
                for j in range(idx.size()[0]):
                    #imgs.append(img_fake[j,...])
            #print("Save test data")
            #print(imgs[1].shape,imgs[0].shape)
            #for img in imgs:
            #    print(img.shape)
            #imgs = np.stack(imgs, axis = 0).transpose(0,2,3,1)
            #plot_batch(imgs, os.path.join(opt.output_path, 'result'+'.jpg'))
            #cv2.imwrite(opt.output_path + 'result.jpg', img_fake[0].transpose(1,2,0)*255)
            #print(len(imgs))
                    x = img_fake[j].transpose(1,2,0)
                    x = np.clip(255 * x, 0, 255)
                    x = np.cast[np.uint8](x)
                    Image.fromarray(x).save(os.path.join(opt.output_path, 'r_result_'+str((idx[j].numpy()))+'.jpg'))