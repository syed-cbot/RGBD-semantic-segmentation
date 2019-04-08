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
from sklearn.model_selection import train_test_split
import os




class DataGenerator(Dataset):
    def __init__(self, data, phase='train'):
        if type(data) is list:
            self.image, self.label = [], []
            for d in data:
                self.image.extend(d.image)
                self.label.extend(d.label)
        else:
            self.image = data.image
            self.label = data.label
        base_transform = {
            'train': transforms.Compose([
                transforms.Resize(640),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(512, scale=(0.8, 1.5), ratio=(1.2, 0.85)),
                transforms.RandomHorizontalFlip(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((480, 640)),
            ]),
        }[phase]
        img_transform = {
            'train': transforms.Compose([
                base_transform,
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            ]),
            'val': transforms.Compose([
                base_transform,
                transforms.ToTensor(),
            ]),
        }[phase]

        def image_transform(image, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            image = img_transform(image)
            if image.size(0)==1:
                image = image.repeat(3, 1, 1)
            return image

        self.image_transform = image_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(base_transform(label))
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy((label>0).astype(np.int64))
        self.label_transform = label_transform

    def __getitem__(self, idx):

        rand_seed = np.random.randint(9223372036854775808)

        image = Image.open(self.image[idx]).convert("RGB")
        image = self.image_transform(image, rand_seed)

        label = Image.open(self.label[idx])
        label = self.label_transform(label, rand_seed)

        sample = {
            'image': image, 'label': label,
        }

        return sample

    def __len__(self):
        return len(self.label)



class SUN(Dataset):
    def __init__(self, data_dir = '/home/guzhangxuan/dataset/', phase='train'):
                #with h5py.File(data_dir+'SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat') as f:
                #            segment2d = [f[element][:] for element in f['SUNRGBD2Dseg']['seglabel']]
        #f = h5py.File(data_dir+'SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBD2Dseg.mat')
        #segment2d = f['SUNRGBD2Dseg']['seglabel']
        #segment2d_list = []
        #for i in range(len(segment2d)):
        #    segment2d_list.append(f[segment2d[0][0]].value)

        traintest_list = sio.loadmat(data_dir+'SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')['alltrain'][0] if phase=='train' else sio.loadmat(data_dir+'SUNRGBD/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')['alltest'][0]
        traintest_list = [key[0] for key in traintest_list]
        #data_list = sio.loadmat(data_dir+'SUNRGBD/SUNRGBDtoolbox/Metadata/SUNRGBDMeta.mat')['SUNRGBDMeta'][0]
        #all_list = []
        #for i in range(len(data_list)):
        #    all_list.append(data_list[i][0][0])


        self.input_list = traintest_list
        #self.input_label = segment2d_list
        #print (len(self.input_list),len(self.input_label))
        #print (self.input_list[0],self.input_label[0])
        #print (segment2d_list[0].shape,segment2d_list[1].shape,len(traintest_list),len(all_list))
        #print (segment2d_list[0].max(),traintest_list[0],all_list[0])
        #print (traintest_list[2000][0],all_list[2000][0])
        #self.input_list = []
        #self.input_label = []
        #for i in range(len(all_list)):
        #    if ('/n/fs/sun3d/data/'+all_list[i]) in traintest_list:
       #         self.input_list.append(all_list[i])
       #         self.input_label.append(segment2d_list[i])
       #     else:
        #        print ('/n/fs/sun3d/data/'+all_list[i])
        #        break
        #print (self.input_list[0],self.input_label[0].shape)
        #print (len(self.input_list),len(self.input_label))
        #del segment2d_list,all_list#traintest_list,data_list,all_list
        #gc.collect()
        string2label = sio.loadmat(data_dir+'SUNRGBD/SUNRGBDtoolbox/Metadata/seg37list.mat')['seg37list'][0].tolist()
        #print (string2label)
        self.label_dict = dict()
        self.label_dict['0'] = 0
        self.label_dict['ottoman'] = 0
        self.label_dict['light'] = 0
        self.phase = phase



        for i in range(len(string2label)):
            #print (string2label[i].tolist()[0])
            self.label_dict[string2label[i].tolist()[0]] = i+1
        #self.label_dict = {string2label[i]:i+1 for i in range(len(string2label))}
        #self.label_dict[0] = 0
        #print (self.label_dict)
        if not os.path.exists('./label_%s.mat'%phase):
            for idx in range(len(traintest_list)):
                print (idx)
                labelPath = data_dir+self.input_list[idx][17:] + '/seg.mat'
                label_old = sio.loadmat(labelPath)['seglabel']
                names = sio.loadmat(labelPath)['names'][0]
                #print (label_old.shape,names)
                label = np.zeros((label_old).shape)
                for i in range(len(label_old)):
                    for j in range(len(label_old[i])):
                        if label_old[i][j]!=0:
                            if names[label_old[i][j]-1][0] not in self.label_dict.keys():
                                self.label_dict[names[label_old[i][j]-1][0]] = 0
                                #print (label_old[i][j],names[label_old[i][j]-1][0],self.input_list[idx][17:])
                                label[i][j] = self.label_dict[names[label_old[i][j]-1][0]]
            scio.savemat('./label_%s.mat'%phase, {'label':label})  
        self.data_dir = data_dir
        base_transform = {
            'train': transforms.Compose([
                transforms.Resize(720),
                transforms.RandomRotation(15),
                transforms.RandomResizedCrop(640, scale=(0.8, 1.5), ratio=(1.2, 0.85)),
                transforms.RandomHorizontalFlip(),
            ]),
            'val': transforms.Compose([
                transforms.Resize((640, 640)),
            ]),
        }[phase]
        img_transform = {
            'train': transforms.Compose([
                base_transform,
                transforms.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.05),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.02),
            ]),
            'val': transforms.Compose([
                base_transform,
                transforms.ToTensor(),
            ]),
        }[phase]

        def image_transform(image, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            image = img_transform(image)
            if image.size(0)==1:
                image = image.repeat(3, 1, 1)
            return image

        self.image_transform = image_transform

        def label_transform(label, rand_seed=None):
            if rand_seed:
                random.seed(rand_seed)
            label = np.array(base_transform(label))
            if label.ndim==3:
                label = label[:, :, 0]
            return torch.from_numpy(label.astype(np.int64))
        self.label_transform = label_transform

    def __getitem__(self, idx):

        rand_seed = np.random.randint(9223372036854775808)
        rgbPath = self.data_dir+self.input_list[idx][17:] + '/image/'
        rgbPath += listdir(rgbPath)[0]
        depthPath = self.data_dir+self.input_list[idx][17:]  + "/depth_bfx/" 
        depthPath += listdir(depthPath)[0]
        labelPath = './label_%s.mat'%self.phase 

        image = Image.open(rgbPath).convert("RGB")
        #cv2.imwrite("orirgb_.jpg", np.array(image));
        image = self.image_transform(image, rand_seed)
        #print (image.shape)

        depth = np.array(Image.open(depthPath))
        depth = depth/depth.max()*255
        #print(np.array(depth).shape)
        depth = transforms.ToPILImage()(depth[:,:,np.newaxis].astype(np.uint8))
        #print(np.array(depth).shape)
        #cv2.imwrite("orid_.jpg", np.array(depth));
        depth = self.image_transform(depth, rand_seed)
        #print (depth.shape)
        #label_old = sio.loadmat(labelPath)['seglabel']
        #names = sio.loadmat(labelPath)['names'][0]
        #print (label_old.shape,names)
        #label = np.zeros((label_old).shape)
        #for i in range(len(label_old)):
        #    for j in range(len(label_old[i])):
        #        if label_old[i][j]!=0:
        #            if names[label_old[i][j]-1][0] not in self.label_dict.keys():
        #                self.label_dict[names[label_old[i][j]-1][0]] = 0
        #            #print (label_old[i][j],names[label_old[i][j]-1][0],self.input_list[idx][17:])
        #            label[i][j] = self.label_dict[names[label_old[i][j]-1][0]]
        label = sio.loadmat(labelPath)['label']
        label = transforms.ToPILImage()(label[:,:,np.newaxis].astype(np.uint8))
        #print (self.input_label[idx].astype(np.uint8).shape,np.array(label).shape)
        #cv2.imwrite("label_.jpg", np.array(label));
        #print (np.array(label))
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
    

    data_dataset = SUN(phase = 'train')
    batch_size = 1
    num_workers = 12
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #import time
    #t = time.time()
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(data_loader)):
        image, label, depth = batch['image'], batch['label'], batch['depth']
        #print(i, image.shape, label.shape, depth.shape)
        #cv2.imwrite("orid.jpg", depth[0].numpy().transpose(1, 2, 0)*255);
        #cv2.imwrite("orirgb.jpg", image[0].numpy().transpose(1, 2, 0)*255);
        #cv2.imwrite("label.jpg", (label[0].numpy()*1.0/label[0].numpy().max()*255));
        #print (depth[0],label[0])
        if np.array(label[0]).max()>36:
            #print (i,label[0])
            cv2.imwrite("label.jpg", (label[0].numpy()*1.0/label[0].numpy().max()*255));
            file = open('file_name.txt','w');

            file.write(str(label[0].numpy().tolist()));

            file.close()
            #print (data_dataset.input_list[i][17:])
            break
    #print(time.time() - t, 'seconds', 'batch_size', batch_size, 'num_workers', num_workers)
    data_dataset = SUN(phase = 'val')
    batch_size = 1
    num_workers = 12
    data_loader = DataLoader(
        data_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #import time
    #t = time.time()
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(data_loader)):
        image, label, depth = batch['image'], batch['label'], batch['depth']
        #print(i, image.shape, label.shape, depth.shape)
        #cv2.imwrite("orid.jpg", depth[0].numpy().transpose(1, 2, 0)*255);
        #cv2.imwrite("orirgb.jpg", image[0].numpy().transpose(1, 2, 0)*255);
        #cv2.imwrite("label.jpg", (label[0].numpy()*1.0/label[0].numpy().max()*255));
        #print (depth[0],label[0])
        if np.array(label[0]).max()>36:
            #print (i,label[0])
            cv2.imwrite("label.jpg", (label[0].numpy()*1.0/label[0].numpy().max()*255));
            file = open('file_name.txt','w');

            file.write(str(label[0].numpy().tolist()));

            file.close()
            #print (data_dataset.input_list[i][17:])
            break