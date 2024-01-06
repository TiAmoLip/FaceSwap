import os
import glob
import torch
import random
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
import torch.nn.functional as F
from insightface_func.face_detect_crop_single import Face_detect_crop
import numpy as np

class data_prefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.num_images = len(loader)
        self.preload()

    def preload(self):
        try:
            self.src_image1, self.src_image2, self.idx = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.src_image1, self.src_image2, self.idx = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.src_image1  = self.src_image1.cuda(non_blocking=True)
            self.src_image1  = self.src_image1.sub_(self.mean).div_(self.std)
            self.src_image2  = self.src_image2.cuda(non_blocking=True)
            self.src_image2  = self.src_image2.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src_image1  = self.src_image1
        src_image2  = self.src_image2
        idx = self.idx
        self.preload()
        return src_image1, src_image2, idx
    
    def __len__(self):
        """Return the number of images."""
        return self.num_images

class SwappingDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""

    def __init__(self,
                    dataroot,
                    img_transform,
                    subffix='jpg',
                    random_seed=1234):
        """Initialize and preprocess the Swapping dataset."""
        self.dataroot = dataroot
        self.img_transform  = img_transform   
        self.subffix        = subffix
        #self.dataset        = []
        for _,_,files in os.walk(dataroot):
            self.files = files
        self.random_seed    = random_seed
        self.crop_size = 512
        #self.preprocess()
        #self.num_images = len(self.dataset)
        #self.app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        #self.app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640),mode='ffhq')
    def preprocess(self):
        """Preprocess the Swapping dataset."""
        print("processing Swapping dataset images...")

        temp_path   = os.path.join(self.image_dir,'*/')
        pathes      = glob.glob(temp_path)
        self.dataset = []
        for dir_item in pathes:
            join_path = glob.glob(os.path.join(dir_item,'*.jpg'))
            print("processing %s"%dir_item,end='\r')
            temp_list = []
            for item in join_path:
                temp_list.append(item)
            self.dataset.append(temp_list)
        random.seed(self.random_seed)
        random.shuffle(self.dataset)
        print('Finished preprocessing the Swapping dataset, total dirs number: %d...'%len(self.dataset))
             
    def __getitem__(self, index):
        """Return two src domain images and two dst domain images."""
        for file in self.files:
            id = str(index) if index >= 10 else '0' + str(index)
            if file.find(id) != -1:
                if file.find('source') != -1:
                    self.image_dir1 = os.path.join(self.dataroot,file)
                else:
                    self.image_dir = os.path.join(self.dataroot,file)
        #image1,_ = self.app.get(np.asarray(Image.open(self.image_dir).convert('RGB')),self.crop_size)
        image1 = Image.open(self.image_dir).convert('RGB')
        image1      = self.img_transform(image1)
        #image1 =    F.interpolate(image1.view(1,image1.size()[0],image1.size()[1],image1.size()[2]),size=(224,224), mode='bicubic').view(image1.size()[0],224,224)
        #image2,_ = self.app.get(np.asarray(Image.open(self.image_dir1).convert('RGB')),self.crop_size)
        image2 = Image.open(self.image_dir1).convert('RGB')
        image2      = self.img_transform(image2)
        #image2 =    F.interpolate(image2.view(1,image2.size()[0],image2.size()[1],image2.size()[2]),size=(224,224), mode='bicubic').view(image2.size()[0],224,224)
        return image1, image2, index
    
    def __len__(self):
        """Return the number of images."""
        return len(self.files)//2

def GetLoader(  dataroot,
                batch_size=16,
                dataloader_workers=8,
                random_seed = 1234
                ):
    """Build and return a data loader."""
        
    num_workers         = dataloader_workers
    random_seed         = random_seed
    
    c_transforms = []
    
    c_transforms.append(T.ToTensor())
    c_transforms = T.Compose(c_transforms)

    content_dataset = SwappingDataset(
                            dataroot,
                            c_transforms,
                            "jpg",
                            random_seed)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=False,shuffle=False,num_workers=num_workers,pin_memory=False)
    prefetcher = data_prefetcher(content_data_loader)
    return prefetcher