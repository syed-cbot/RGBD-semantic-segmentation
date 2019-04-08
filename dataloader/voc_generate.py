import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from os import listdir
import cv2
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.io as sio
import h5py
import gc
import os
import torch
import scipy.misc as misc
import numpy as np
from torch.utils import data
import cv2
import os
import glob
#IMG_SCALE  = 1./255
IMG_MEAN = torch.from_numpy(np.array([73.16, 82.91, 72.39]).reshape((1, 1, 3))).float()/255
IMG_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))).float()

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class VOC(Dataset):
    def __init__(self,  phase='train'):
        root = '/home/gzx/RGBD/'
        filename = root+'VOC2012/ImageSets/Segmentation/%s.txt'%phase
        with open(filename, 'r') as f:
            lines = f.readlines()
        data_list = [c.strip() for c in lines]
        #print (lines,data_list)
        self.label_list = [root+'VOC2012/SegmentationClass/'+input+'.png' for input in data_list]
        self.image_list = [root+'VOC2012/JPEGImages/'+input+'.jpg' for input in data_list]
        self.depth_list = [root+'depth/'+input+'.jpg' for input  in data_list]

        def base_transform(t='img'):
            if t=='label':
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR
            return {
                'train': transforms.Compose([
                    transforms.Resize((960,960), interpolation=interpolation),
                    transforms.RandomRotation(15, resample=interpolation),
                    transforms.RandomResizedCrop(640, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation=interpolation),
                    #transforms.RandomCrop((480,640), padding=0),
                    transforms.RandomHorizontalFlip(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((640,640), interpolation=interpolation),
                ]),
                'test': transforms.Compose([
                    transforms.Resize(720, interpolation=interpolation),
                    transforms.Resize((640,640), interpolation=interpolation),
                ]),
            }[phase]
        img_transform = {
            'train': transforms.Compose([
                base_transform(),
                #transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                #transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            ]),
            'val': transforms.Compose([
                base_transform(),
                transforms.ToTensor(),
            ]),
            'test': transforms.Compose([
                transforms.TenCrop(640),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            ]),
        }[phase]

        def image_transform(image, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            image = img_transform(image).permute(1,2,0)
            image = (image  - IMG_MEAN) / IMG_STD
            image = image.permute(2,0,1)
            if image.size(0)==1:
                image = image.repeat(3, 1, 1)
            return image

        self.image_transform = image_transform

        def depth_transform(depth, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            depth = np.array(base_transform('label')(depth))[np.newaxis,:,:]
            return torch.from_numpy(depth.astype(np.float32))
        self.depth_transform = depth_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(base_transform('label')(label)).astype('uint8')
            #print (label)
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform

    def __getitem__(self, idx):

        rand_seed = np.random.randint(9223372036854775808)
        #print (self.image_list[idx],self.label_list[idx],self.depth_list[idx])
        image = Image.open(self.image_list[idx]).convert("RGB")
        image = self.image_transform(image, rand_seed)
        #depth = Image.open(self.depth_list[idx])
        depth = -np.array(Image.open(self.depth_list[idx]))
        depth = (depth-depth.min())*255.0/(depth.max()-depth.min())
        depth = Image.fromarray(depth.astype('uint8'))
        depth = self.depth_transform(depth, rand_seed)
        label = Image.open(self.label_list[idx])
        label = self.label_transform(label, rand_seed)
        sample = {
            'image': image, 'label': label,'depth': depth
        }

        return sample

    def __len__(self):
        return len(self.label_list)

if __name__ == '__main__':

    from imagereader import imagefile
    from torch.utils.data import DataLoader
    
    import numpy as np
    cmap = np.load('./utils/cmap.npy')
    data_dataset = VOC(phase = 'val')
    batch_size = 1
    num_workers = 1
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #import time
    #t = time.time()
    from tqdm import tqdm
    #all_0 = np.zeros((640,640))
    #all_37 = np.ones((640,640))*37
    #cv2.imwrite("0.jpg", (cmap[all_0.astype(np.uint8)]));
    #cv2.imwrite("37.jpg", (cmap[all_37.astype(np.uint8)]));
    for i, batch in enumerate(tqdm(data_loader)):
        image, label, depth = batch['image'], batch['label'], batch['depth']
        print(i, image.shape, label.shape,depth.shape,((label>19)*(label<255)).max(),label.min(),label.max())
        #print (depth[0].numpy())
        #cv2.imwrite("orid_%s.jpg"%i, depth[0][0].numpy());
        #cv2.imwrite("orirgb_%s.jpg"%i, (image[0].numpy().transpose(1, 2, 0)*255*IMG_STD.numpy()+IMG_MEAN.numpy()));
        #cv2.imwrite("label_%s.jpg"%i, (cmap[label[0].numpy().astype(np.uint8)]));
        #iou,acc,mean = cal_metric(label,label,19)
        #print (iou,acc,mean)
        if i==10:
            break

