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

#IMG_SCALE  = 1./255
IMG_MEAN = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))).float()
IMG_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))).float()

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RGBD(Dataset):
    def __init__(self, dataset, phase='train'):
        if dataset == 'NYU':
            data_dir = '/home/gzx/RGBD/NYU-DV2/nyudv2_data_label_extractor/'
            #print (sio.loadmat(data_dir+'splits.mat')['trainNdxs'][:,0].tolist(),sio.loadmat(data_dir+'splits.mat')['testNdxs'][:200][:,0].tolist())
            input_list = sio.loadmat(data_dir+'splits.mat')['trainNdxs'][:,0].tolist() + sio.loadmat(data_dir+'splits.mat')['testNdxs'][::5][:,0].tolist() if phase=='train' else sio.loadmat(data_dir+'splits.mat')['testNdxs']
            self.image_list = [data_dir+'data/images/'+'img_%d.png'%(input+5000) for input in input_list]
            self.label_list = [data_dir+'data/label40/'+ 'img_%d.png'%(input+5000) for input in input_list]
            self.depth_list = [data_dir+'data/depth/'+ 'img_%d.png'%(input+5000) for input in input_list]
        elif dataset == 'SUN':
            sun_phase = phase if phase=='train' else 'test'
            data_dir = '/home/gzx/RGBD/'
            input_list = range(1,5286) if phase=='train' else range(1,5051)
            self.image_list = [data_dir+'SUN/%s_image/'%sun_phase+'img-%06d.jpg'%input for input in input_list]
            self.label_list = [data_dir+'SUN/sunrgbd-meta-data/labels/%s/'%sun_phase+ 'img-%06d.png'%(input if sun_phase=='test' else input+5050) for input in input_list]
            self.depth_list = [data_dir+'SUN/sunrgbd_%s_depth/'%sun_phase+ '%d.png'%input for input in input_list]
        else:
            assert (1<0)
        self.dataset = dataset
        def base_transform(t='img'):
            if t=='label':
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR
            return {
                'train': transforms.Compose([
                    transforms.Resize((720,960), interpolation=interpolation),
                    transforms.RandomRotation(15, resample=interpolation),
                    transforms.RandomResizedCrop(640, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation=interpolation),
                    transforms.RandomCrop((480,640), padding=0),
                    transforms.RandomHorizontalFlip(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((480,640), interpolation=interpolation),
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
            label = np.array(base_transform('label')(label)).astype('uint8')-1
            #print (label)
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform

    def __getitem__(self, idx):

        rand_seed = np.random.randint(9223372036854775808)

        image = Image.open(self.image_list[idx]).convert("RGB")
        image = self.image_transform(image, rand_seed)
        #depth = Image.open(self.depth_list[idx])
        depth = np.array(Image.open(self.depth_list[idx]))
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


class NYU(Dataset):
    def __init__(self, data_dir = '/home/guzhangxuan/dataset/NYU-DV2/nyudv2_data_label_extractor/', phase='train'):
        
        self.input_list = sio.loadmat(data_dir+'splits.mat')['trainNdxs'] if phase=='train' else sio.loadmat(data_dir+'splits.mat')['testNdxs']
        self.data_dir = data_dir
        lab_transform = {
            'train': transforms.Compose([
                transforms.Resize((640,640),interpolation = Image.NEAREST),
                #transforms.RandomRotation(10),
                #transforms.RandomResizedCrop(560, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation = Image.NEAREST),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((640,640),interpolation = Image.NEAREST),
                #transforms.ToTensor(),
            ]),
        }[phase]
        img_transform = {
            'train': transforms.Compose([
                transforms.Resize((640,640)),
                #transforms.RandomRotation(10),
                #transforms.RandomResizedCrop(560, scale=(0.8, 1.5), ratio=(1.2, 0.85)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((640,640)),
                transforms.ToTensor(),
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
            depth = np.array(lab_transform(depth))
            return torch.from_numpy(depth.astype(np.int64))
        self.depth_transform = depth_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(lab_transform(label))-1
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform
        self.phase = 'train' if phase=='train' else 'test'

    def __getitem__(self, idx):
        #rgb: img-000001.jpg  1-5285         1-5050     
        #d: 4247.png   1-5285      1-5050  
        #label: img-005051.png  5151- 10335  1-5050
        rand_seed = np.random.randint(9223372036854775808)

        rgbPath = self.data_dir+'data/images/'+'img_%d.png'%(self.input_list[idx]+5000)
        
        depthPath = self.data_dir+'data/depth/'+ 'img_%d.png'%(self.input_list[idx]+5000) 

        labelPath =self.data_dir+'data/label40/'+ 'img_%d.png'%(self.input_list[idx]+5000) 

        image = Image.open(rgbPath).convert("RGB")
        image = self.image_transform(image, rand_seed)
        depth = np.array(Image.open(depthPath))
        #print (np.array(Image.open(depthPath)))
        #print (depth.min(),depth.max())
        depth = (depth-depth.min())*255.0/(depth.max()-depth.min())
        #print(np.array(depth).shape)
        #cv2.imwrite("orid_.jpg", depth);
        depth = Image.fromarray(depth.astype('uint8'))
        #depth = transforms.ToPILImage()(depth[:,:,np.newaxis].astype(np.uint8))
        #print(np.array(depth).shape)
        #cv2.imwrite("orid_.jpg", np.array(depth));
        depth = self.depth_transform(depth, rand_seed)
        label = Image.open(labelPath)
        label = self.label_transform(label, rand_seed)
        #print (label.shape)
        sample = {
            'image': image, 'label': label,'depth': depth
        }

        return sample

    def __len__(self):
        return len(self.input_list)

class SUN(Dataset):
    def __init__(self, data_dir = '/home/guzhangxuan/dataset/', phase='train'):
        #rgb: img-000001.jpg  1-5285         1-5050     
        #d: 4247.png   1-5285      1-5050  
        #label: img-005051.png  5151- 10335  1-5050
        self.input_list = range(1,5286) if phase=='train' else range(1,5051)
        self.data_dir = data_dir
        lab_transform = {
            'train': transforms.Compose([
                transforms.Resize((640,640),interpolation = Image.NEAREST),
                #transforms.RandomRotation(10),
                #transforms.RandomResizedCrop(560, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation = Image.NEAREST),
                #transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((640,640),interpolation = Image.NEAREST),
                #transforms.ToTensor(),
            ]),
        }[phase]
        img_transform = {
            'train': transforms.Compose([
                transforms.Resize((640,640)),
                #transforms.RandomRotation(10),
                #transforms.RandomResizedCrop(560, scale=(0.8, 1.5), ratio=(1.2, 0.85)),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((640,640)),
                transforms.ToTensor(),
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
            depth = np.array(lab_transform(depth))
            return torch.from_numpy(depth.astype(np.int64))
        self.depth_transform = depth_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(lab_transform(label))-1
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform
        self.phase = 'train' if phase=='train' else 'test'

    def __getitem__(self, idx):
        #rgb: img-000001.jpg  1-5285         1-5050     
        #d: 4247.png   1-5285      1-5050  
        #label: img-005051.png  5151- 10335  1-5050
        rand_seed = np.random.randint(9223372036854775808)

        rgbPath = self.data_dir+'SUN/%s_image/'%self.phase+'img-%06d.jpg'%self.input_list[idx]
        
        depthPath = self.data_dir+'SUN/sunrgbd_%s_depth/'%self.phase+ '%d.png'%self.input_list[idx] 

        labelPath = self.data_dir+'SUN/sunrgbd-meta-data/labels/%s/'%self.phase+ 'img-%06d.png'%(self.input_list[idx] if self.phase=='test' else self.input_list[idx]+5050)

        image = Image.open(rgbPath).convert("RGB")
        image = self.image_transform(image, rand_seed)
        depth = np.array(Image.open(depthPath))
        #print (np.array(Image.open(depthPath)))
        #print (depth.min(),depth.max())
        depth = (depth-depth.min())*255.0/(depth.max()-depth.min())
        #print(np.array(depth).shape)
        #cv2.imwrite("orid_.jpg", depth);
        depth = Image.fromarray(depth.astype('uint8'))
        #depth = transforms.ToPILImage()(depth[:,:,np.newaxis].astype(np.uint8))
        #print(np.array(depth).shape)
        #cv2.imwrite("orid_.jpg", np.array(depth));
        depth = self.depth_transform(depth, rand_seed)
        label = Image.open(labelPath)
        label = self.label_transform(label, rand_seed)
        #print (label.shape)
        sample = {
            'image': image, 'label': label,'depth': depth
        }

        return sample

    def __len__(self):
        return len(self.input_list)


if __name__ == '__main__':

    from imagereader import imagefile
    from torch.utils.data import DataLoader
    
    import numpy as np
    cmap = np.load('./utils/cmap.npy')
    data_dataset = RGBD(dataset = 'NYU', phase = 'val')
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
        print(i, image.shape, label.shape, depth.shape)
        #cv2.imwrite("orid.jpg", depth[0].numpy());
        #cv2.imwrite("orirgb.jpg", image[0].numpy().transpose(1, 2, 0)*255);
        #cv2.imwrite("label.jpg", (cmap[label[0].numpy().astype(np.uint8)+1]));
        #print (depth[0],label[0])
        bs, ncrops, c, h, w = image.size()
        print (bs, ncrops, c, h, w)
        if i==0:

            break
    #print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)
