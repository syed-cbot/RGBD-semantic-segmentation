import os
import time
import argparse
import numpy as np
import cv2
import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
from dataloader.voc_generate_test import VOC
#from dataloader.generator import RGBD
from dataloader.imagereader import imagefile
from utils.lr_scheduler import LRScheduler
from utils import helper
import torchvision.models as models
#================ load your net here ===================
from net.refinenet_1 import RefineNet4CascadePoolingImproved as network
#from net.refinenet_nolz import RefineNet4CascadePoolingImproved as network
#from net.refinenet import RefineNet4CascadePoolingImprovedDepth as network
#from net.refinenet import RefineNet4CascadePoolingImproved as network
import torch.nn.functional as F



def validate(data_loader,data_loader_1,data_loader_2,data_loader_3, net, loss, epoch, num_class):
    root = '/home/gzx/RGBD/'
    filename = root+'VOC2012/ImageSets/Segmentation/test.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()
    data_list = [c.strip() for c in lines]
    net.eval()
    i = 0
    for batch,batch_1,batch_2,batch_3 in zip(data_loader,data_loader_1,data_loader_2,data_loader_3):
        data, depth, shape = batch['image'], batch['depth'], batch['shape']
        data = data.cuda(async=True)
        depth = depth.cuda(async=True)
        #prediction = net(data)
        #prediction = net(data,depth)
        prediction = net(x = data,epoch = epoch,depth = depth.squeeze(1),label = None, train=False) 
        prediction = F.interpolate(prediction, scale_factor=1, mode='bilinear', align_corners=False).cpu()
 

        data,  depth = batch_1['image'],  batch_1['depth']
        data = data.cuda(async=True)
        depth = depth.cuda(async=True)
        #prediction_1 = net(data)
        #prediction_1 = net(data,depth)
        prediction_1 = net(x = data,epoch = epoch,depth = depth.squeeze(1), label = None,train=False) 
        prediction_1 = F.interpolate(prediction_1, scale_factor=512/1024, mode='bilinear', align_corners=False).cpu()    
        
        data,  depth = batch_2['image'], batch_2['depth']
        data = data.cuda(async=True)
        depth = depth.cuda(async=True)
        #prediction_2 = net(data)
        #prediction_2 = net(data,depth)
        prediction_2 = net(x = data,epoch = epoch,depth = depth.squeeze(1), label = None,train=False) 
        prediction_2 = F.interpolate(prediction_2, scale_factor=512/1536, mode='bilinear', align_corners=False).cpu()  
		
        data,  depth = batch_3['image'], batch_3['depth']
        data = data.cuda(async=True)
        depth = depth.cuda(async=True)
        #prediction_3 = net(data)
        #prediction_3 = net(data,depth)
        prediction_3 = net(x = data,epoch = epoch,depth = depth.squeeze(1),label = None,train=False) 
        prediction_3 = F.interpolate(prediction_3, scale_factor=512/256, mode='bilinear', align_corners=False).cpu()

        
        pred = torch.max((prediction+prediction_1+prediction_2+prediction_3)/4,1)[1].squeeze(0).numpy()
        size = (shape[1], shape[0])  
        pred = cv2.resize(pred, size, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite('results/VOC2012/Segmentation/comp5_test_cls/%s.png'%(data_list[i]),pred)
        i+=1


if __name__ == '__main__':

    workers = 4
    batch_size = 1
    
    base_lr = 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    #parser.add_argument('-d', '--dataset', default='NYU', help='NYU or SUN', type=str)
    args = parser.parse_args()
    #save_dir = './%s_base/'%args.dataset
    #if not os.path.exists(save_dir):
    #    os.mkdir(save_dir)
        
    #epochs = {'SUN':60,'NYU':300}[args.dataset]

    val_dataset = VOC(512,'test')
    val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_dataset_2048 = VOC(1024,'test')
    val_loader_2048 = DataLoader(
            val_dataset_2048, batch_size=batch_size, shuffle=False, num_workers=workers)

    val_dataset_480 = VOC(1536,'test')
    val_loader_480 = DataLoader(
            val_dataset_480, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_dataset_840 = VOC(256,'test')
    val_loader_840 = DataLoader(
            val_dataset_840, batch_size=batch_size, shuffle=False, num_workers=workers)
    num_class = 21
    ignore_label = 255
    loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
    #patience = {'SUN':15,'NYU':60}[args.dataset]


    print('Val sample number: %d' % len(val_dataset))
    ############################################################

	

    #net = network(640,num_classes = num_class,resnet_factory = models.resnet152, freeze_resnet=False)
    net = network((3,640),num_classes = num_class,resnet_factory = models.resnet152, freeze_resnet=False)
    
    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    #print(net.named_parameter())
    net = DataParallel(net)
    if os.path.exists(args.resume):
        print('loading checkpoint %s'%(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    

    optimizer = torch.optim.SGD(
        net.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    #lrs = LRScheduler(
    #    lr, patience=patience, factor=0.5, min_lr=0.5*0.5*0.5 * lr, best_loss=best_val_loss)
    with torch.no_grad():
        
        validate(val_loader,val_loader_2048,val_loader_480,val_loader_840, net, loss, 0, num_class)

    
