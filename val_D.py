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
from dataloader.generator import SUN,NYU, RGBD
from dataloader.imagereader import imagefile
from utils.lr_scheduler import LRScheduler
from utils import helper
import torchvision.models as models
#================ load your net here ===================
#from net.refinenet import RefineNet4CascadePoolingImproved as network
#from net.refinenet import RefineNet4Cascade as network
from net.refinenet import RefineNet4CascadePoolingImprovedDepth as network








def validate(data_loader, net, loss, epoch, num_class):
    start_time = time.time()
    net.eval()
    metrics = []
    iou = 0
    acc = 0
    mean_acc = 0
    d_iou = 0
    d_acc = 0
    d_mean_acc = 0
    k = 0


    for i, batch in enumerate(tqdm(data_loader)):
        data, heatmaps, depth = batch['image'], batch['label'], batch['depth']
        data = data.cuda(async=True)
        depth = depth.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        #prediction_d = net(data,depth)
        prediction_d = net(data,torch.zeros(depth.size()))
        #print(prediction_d.shape,prediction.shape)
        #loss_output = loss(prediction, heatmaps)
        #iou_, acc_, mean_acc_ = helper.cal_metric(heatmaps, prediction, num_class)
        iou_d, acc_d, mean_acc_d = helper.cal_metric(heatmaps, prediction_d, num_class)
        #iou += iou_
        #acc += acc_
        #mean_acc += mean_acc_
        d_iou += iou_d
        d_acc += acc_d
        d_mean_acc += mean_acc_d
        k += 1
        if k==(len(data_loader)-2):
        	output_img = batch['image']
        	out_label = batch['label']
        	out_prediction_d = prediction_d
        	#out_prediction = prediction

        #metrics.append(loss_output.item())
    assert k==len(data_loader)
    #iou /= len(data_loader)
    #acc /= len(data_loader)
    #mean_acc /= len(data_loader)
    d_iou /= len(data_loader)
    d_acc /= len(data_loader)
    d_mean_acc /= len(data_loader)
    #acc, mean_acc, iou, fwavacc = evaluate(predictions_all, gts_all, voc.num_classes)
    img = helper.make_validation_img(output_img.numpy(),
                                    out_label.numpy(),
                                    out_prediction_d.cpu().numpy())
    cv2.imwrite('%s/test_withdepth_%.4f.png'%(save_dir, iou),
                img[:, :, ::-1])


    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time, d_iou, d_acc, d_mean_acc


if __name__ == '__main__':
    val_path = 'SUN_RDFnet/ckpt_030.ckpt'
    workers = 4
    batch_size = 15
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    parser.add_argument('-d', '--dataset', default='NYU', help='NYU or SUN', type=str)
    args = parser.parse_args()
    save_dir = './%s_RDFnet/'%args.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    val_dataset = RGBD(args.dataset,'val')
    val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    num_class = {'SUN':37,'NYU':40}[args.dataset]
    ignore_label = {'SUN':255,'NYU':-1}[args.dataset]
    loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
    patience = {'SUN':10,'NYU':40}[args.dataset]
    print('Val sample number: %d' % len(val_dataset))
    ############################################################
    net = network(640,num_classes = num_class,resnet_factory = models.resnet152, freeze_resnet=False)
    loss = nn.CrossEntropyLoss(ignore_index=ignore_label)

   

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    net = DataParallel(net)



    print('loading checkpoint ')
    checkpoint = torch.load(val_path)
    net.load_state_dict(checkpoint['state_dict'])


    with torch.no_grad():
        val_metrics, val_time,val_iou_d, val_acc_d, val_mean_acc_d = validate(val_loader, net, loss, 0, num_class)
    print (val_iou_d, val_acc_d, val_mean_acc_d)