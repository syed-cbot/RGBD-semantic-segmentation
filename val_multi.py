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
from dataloader.generator_val import RGBD
#from dataloader.generator import RGBD
from dataloader.imagereader import imagefile
from utils.lr_scheduler import LRScheduler
from utils import helper
import torchvision.models as models
#================ load your net here ===================
from net.refinenet_1 import RefineNet4CascadePoolingImproved as network
import torch.nn.functional as F



def validate(data_loader,data_loader_1,data_loader_2,data_loader_3, net, loss, epoch, num_class):
    start_time = time.time()
    net.eval()
    metrics = []
    iou = 0
    acc = 0
    mean_acc = 0



    for batch,batch_1,batch_2,batch_3 in zip(data_loader,data_loader_1,data_loader_2,data_loader_3):
        data, heatmaps, depth = batch['image'], batch['label'], batch['depth']
        data = data.cuda(async=True)
        heatmaps_0 = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction = F.interpolate(prediction, scale_factor=1, mode='bilinear', align_corners=False).cpu()
 

        data, heatmaps, depth = batch_1['image'], batch_1['label'], batch_1['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_1,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_1 = F.interpolate(prediction_1, scale_factor=5/12, mode='bilinear', align_corners=False).cpu()    
        
        data, heatmaps, depth = batch_2['image'], batch_2['label'], batch_2['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_2,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_2 = F.interpolate(prediction_2, scale_factor=1.25, mode='bilinear', align_corners=False).cpu()  
		
        data, heatmaps, depth = batch_3['image'], batch_3['label'], batch_3['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_3,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_3 = F.interpolate(prediction_3, scale_factor=0.625, mode='bilinear', align_corners=False).cpu()

        
        pred = (prediction+prediction_1+prediction_2+prediction_3)/4
        iou_, acc_, mean_acc_ = helper.cal_metric(heatmaps_0.cpu(), pred, num_class)
        #iou_1, acc_1, mean_acc_1 = helper.cal_metric(label_list[i], pred_list_1[i], num_class)
        #iou_2, acc_2, mean_acc_2 = helper.cal_metric(label_list[i], pred_list_2[i], num_class)
        iou += iou_
        #iou1 += iou_1
        #iou2 += iou_2
        acc += acc_
        mean_acc += mean_acc_




    iou /= len(data_loader)
    #iou1 /= len(data_loader)
    #iou2 /= len(data_loader)
    acc /= len(data_loader)
    mean_acc /= len(data_loader)
    #img = helper.make_validation_img(batch['image'].numpy(),
    #                                batch['label'].numpy(),
    #                                prediction.cpu().numpy())
    #cv2.imwrite('%s/validate_%d_%.4f.png'%(save_dir, epoch, iou),
    #            img[:, :, ::-1])

    end_time = time.time()

    return  end_time - start_time, iou, acc, mean_acc


if __name__ == '__main__':

    workers = 8
    batch_size = 4
    
    base_lr = 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    parser.add_argument('-d', '--dataset', default='NYU', help='NYU or SUN', type=str)
    args = parser.parse_args()
    save_dir = './%s_base/'%args.dataset
    #if not os.path.exists(save_dir):
    #    os.mkdir(save_dir)
        
    epochs = {'SUN':60,'NYU':300}[args.dataset]

    val_dataset = RGBD(args.dataset,640,'val')
    val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_dataset_2048 = RGBD(args.dataset,1536,'val')
    val_loader_2048 = DataLoader(
            val_dataset_2048, batch_size=batch_size, shuffle=False, num_workers=workers)

    val_dataset_480 = RGBD(args.dataset,512,'val')
    val_loader_480 = DataLoader(
            val_dataset_480, batch_size=batch_size, shuffle=False, num_workers=workers)
    val_dataset_840 = RGBD(args.dataset,1024,'val')
    val_loader_840 = DataLoader(
            val_dataset_840, batch_size=batch_size, shuffle=False, num_workers=workers)
    num_class = {'SUN':37,'NYU':40}[args.dataset]
    ignore_label = 255
    loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
    patience = {'SUN':15,'NYU':60}[args.dataset]


    print('Val sample number: %d' % len(val_dataset))
    ############################################################

	


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
    lrs = LRScheduler(
        lr, patience=patience, factor=0.5, min_lr=0.5*0.5*0.5 * lr, best_loss=best_val_loss)
    with torch.no_grad():
        #val_time, val_iou, val_acc, val_mean_acc = validate(val_loader, net, loss, 0, num_class)
        #val_time, val_iou_480, val_acc, val_mean_acc = validate(val_loader_480, net, loss, 0, num_class)
        #val_time, val_iou_840, val_acc, val_mean_acc = validate(val_loader_840, net, loss, 0, num_class)
        val_time, val_iou, val_acc, val_mean_acc = validate(val_loader,val_loader_2048,val_loader_480,val_loader_840, net, loss, 0, num_class)
    print (val_iou, val_acc, val_mean_acc)
    