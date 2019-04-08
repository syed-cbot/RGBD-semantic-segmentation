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
#IMG_SCALE  = 1./255
IMG_MEAN = torch.from_numpy(np.array([73.16, 82.91, 72.39]).reshape((1, 1, 3))).float()/255
#IMG_STD = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))).float()

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]

class CityscapesLoader(data.Dataset):
    """
    CityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]]

    label_colours = dict(zip(range(19), colors))

    def __init__(self, root = '.', split="train", gt="gtFine"):
        """
        :param root:         (str)  Path to the data sets root
        :param split:        (str)  Data set split -- 'train' 'train_extra' or 'val'
        :param gt:           (str)  Type of ground truth label -- 'gtFine' or 'gtCoarse'
        """
        self.root = root
        self.gt = gt
        self.split = split
        self.n_classes = 19
        self.mean = np.array([73.16, 82.91, 72.39])
        self.files = {}
        self.depths = {}
        self.images_base = os.path.join(self.root, 'leftImg8bit', self.split)
        self.annotations_base = os.path.join(self.root, gt, self.split)
        self.depth_base = os.path.join(self.root, 'depth', self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix='.png')

        self.depths[split] = recursive_glob(rootdir=self.depth_base, suffix='.png')
        #print (self.depths[split][0])
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                            'motorcycle', 'bicycle']

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("> No files for split=[%s] found in %s" % (split, self.images_base))

        print("> Found %d %s images..." % (len(self.files[split]), split))

        #assert (len(self.files[self.split])==len(self.depths[self.split]))
        def base_transform(t='img'):
            if t=='label':
                interpolation = Image.NEAREST
            else:
                interpolation = Image.BILINEAR
            return {
                'train': transforms.Compose([
                    transforms.Resize((1024,2048), interpolation=interpolation),
                    transforms.RandomRotation(15, resample=interpolation),
                    transforms.RandomResizedCrop(1024, scale=(0.8, 1.5), ratio=(1.2, 0.85), interpolation=interpolation),
                    transforms.RandomCrop((512,1024), padding=0),
                    transforms.RandomHorizontalFlip(),
                ]),
                'val': transforms.Compose([
                    transforms.Resize((512,1024), interpolation=interpolation),
                ]),
                'test': transforms.Compose([
                    transforms.Resize((512,1024), interpolation=interpolation),
                ]),
            }[split]
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
                base_transform(),
                transforms.ToTensor(),
            ]),
        }[split]

        def image_transform(image, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            image = img_transform(image).permute(1,2,0)
            image = (image  - IMG_MEAN)
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
            label = self.encode_segmap(np.array(base_transform('label')(label), dtype=np.uint8))
            #print (label)
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform
    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        rand_seed = np.random.randint(9223372036854775808)
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + '{}_labelIds.png'.format(self.gt))
        depth_path = self.depths[self.split][index].rstrip()

        if not os.path.isfile(img_path) or not os.path.exists(img_path):
            raise Exception("{} is not a file, can not open with imread.".format(img_path))
        if not os.path.isfile(lbl_path) or not os.path.exists(lbl_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))
        if not os.path.isfile(depth_path) or not os.path.exists(depth_path):
            raise Exception("{} is not a file, can not open with imread.".format(lbl_path))  
        '''      
        img = misc.imread(img_path)
        img = np.array(img, dtype=np.uint8)
        # img = misc.imresize(img, (self.img_size[0], self.img_size[1], "bilinear"))
        lbl = misc.imread(lbl_path)
        # lbl = misc.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode='F')
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl'''

        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image, rand_seed)
        #depth = Image.open(self.depth_list[idx])
        depth = np.array(Image.open(depth_path))
        depth = 255.0-(depth-depth.min())*255.0/(depth.max()-depth.min())
        depth = Image.fromarray(depth.astype('uint8'))
        depth = self.depth_transform(depth, rand_seed)
        label = Image.open(lbl_path)
        label = self.label_transform(label, rand_seed)
        sample = {
            'image': image, 'label': label,'depth': depth
        }

        return sample

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        img = img[:, :, ::-1]         # From RGB to BGR
        img = img.astype(float)
        img -= self.mean
        img /= 255.0
        img = img.transpose(2, 0, 1)  # From H*W*C to C*H*W

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            raise ValueError("> Segmentation map contained invalid class values.")

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask
def cal_metric(heatmaps, prediction, num_class):
    total_iou = 0
    total_acc = 0
    acc = 0
    mean_acc = 0
    prediction = prediction
    max_label = num_class
    for i in range(len(prediction)):
        pred = prediction[i]
        gt = heatmaps[i]
        #pred = pred [(gt > 0)]
        #print (pred.shape,gt.shape)
        #acc += torch.sum((pred == gt)).item()/(pred.shape[0]*pred.shape[1]-torch.sum((gt==255)).item()-torch.sum((gt==-1)).item())  

        intersect = [0]*max_label
        union = [0]*max_label
        gt_label = [0]*max_label
        #acc_label = [0]*max_label
        ac_mean = []
        unique_label = np.unique(gt.data.cpu().numpy())
        iou = []
        uni_gtlabel = []
        #print ('=====new_image==========')
        #print ('unique:', unique_label)
        for j in range(max_label):          
            p_acc = (pred==j)
            g_acc = (gt==j)
            match = p_acc +  g_acc
            it = torch.sum(match==2).item()
            un = torch.sum(match>0).item()
            intersect[j] += it
            union[j] += un
            #acc_label[j] += torch.sum((pred==j)).item()
            gt_label[j] += torch.sum((gt==j)).item()

        for k in range(max_label):
            if k in unique_label:
                ac_mean.append(intersect[k]*1.0/gt_label[k])
                iou.append(intersect[k]*1.0/union[k])
                uni_gtlabel.append(gt_label[k])
        #print ('uni_gtlabel:',uni_gtlabel)
        #print ('iou:',iou)
        acc = (sum(intersect)/sum(gt_label))
        Aiou = (sum(iou)/len(iou))
        Amean = (sum(ac_mean)/len(ac_mean))
        total_iou  += Aiou
        total_acc += acc
        mean_acc += Amean
    return total_iou / len(prediction), total_acc/ len(prediction), mean_acc/len(prediction)

if __name__ == '__main__':

    from imagereader import imagefile
    from torch.utils.data import DataLoader
    
    import numpy as np
    cmap = np.load('./utils/cmap.npy')
    data_dataset = CityscapesLoader(split = 'val')
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
        print(i, image.shape, label.shape,((label>18)*(label<250)).max(),label.min())
        #print (depth[0].numpy())
        cv2.imwrite("orid_%s.jpg"%i, depth[0].numpy());
        cv2.imwrite("orirgb_%s.jpg"%i, (image[0].numpy().transpose(1, 2, 0)+IMG_MEAN.numpy())*255);
        cv2.imwrite("label_%s.jpg"%i, (cmap[label[0].numpy().astype(np.uint8)+1]));
        iou,acc,mean = cal_metric(label,label,19)
        print (iou,acc,mean)
        if i==1:
            break

