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
#from net.refinenet import RefineNet4CascadePoolingImprovedDepth as network
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs

def print_log(epoch,
              lr,
              train_metrics,
              train_time,
              val_metrics=None,
              val_time=None,
              val_iou=None,
              val_acc = None, 
              val_mean_acc = None,
              save_dir=None,
              log_mode=None):
    if epoch > 1:
        log_mode = 'a'
    train_metrics = np.mean(train_metrics, axis=0)
    str0 = 'Epoch %03d (lr %.7f)' % (epoch, lr)
    str0 += ', Train: time %3.2f loss: %2.4f' \
           % (train_time, train_metrics)
    f = open(save_dir + 'train_log.txt', log_mode)
    if val_time is not None:
        val_metrics = np.mean(val_metrics, axis=0)
        str0 += ', Validation: time %3.2f loss: %2.4f iou: %.4f acc: %2.4f mean_acc: %2.4f' \
               % (val_time, val_metrics, val_iou,  val_acc, val_mean_acc)
    print(str0)
    f.write(str0)
    f.write('\n')
    f.close()


def train(data_loader, net, loss, optimizer, lr):
    start_time = time.time()

    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    #net.module.freeze_bn()
    metrics = []
    for i, batch in enumerate(tqdm(data_loader)):
        data, heatmaps, depth = batch['image'], batch['label'], batch['depth']
        #print (data.size(), heatmaps.size(), depth.size())
        data = data.cuda(async=True)
        #depth = depth.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        prediction = net(data)
        loss_output = loss(prediction, heatmaps)
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()
        metrics.append(loss_output.item())
    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time


def validate(data_loader, net, loss, epoch, num_class):
    start_time = time.time()
    net.eval()
    metrics = []
    iou = 0
    acc = 0
    mean_acc = 0
    for i, batch in enumerate(tqdm(data_loader)):
        data, heatmaps, depth = batch['image'], batch['label'], batch['depth']
        data = data.cuda(async=True)
        #depth = depth.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        prediction = net(data)
        loss_output = loss(prediction, heatmaps)
        iou_, acc_, mean_acc_ = helper.cal_metric(heatmaps, prediction, num_class)
        iou += iou_
        acc += acc_
        mean_acc += mean_acc_
        metrics.append(loss_output.item())
        if i==(len(data_loader)-2):
        	output_img = batch['image']
        	out_label = batch['label']
        	out_prediction = prediction
    iou /= len(data_loader)
    acc /= len(data_loader)
    mean_acc /= len(data_loader)
    img = helper.make_validation_img(output_img.numpy(),
                                    out_label.numpy(),
                                    out_prediction.cpu().numpy())
    cv2.imwrite('%s/validate_%d_%.4f.png'%(save_dir, epoch, iou),
                img[:, :, ::-1])

    end_time = time.time()
    metrics = np.asarray(metrics, np.float32)
    return metrics, end_time - start_time, iou, acc, mean_acc


if __name__ == '__main__':

    workers = 8
    batch_size = 4
    
    base_lr = 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    parser.add_argument('-d', '--dataset', default='NYU', help='NYU or SUN', type=str)
    args = parser.parse_args()
    save_dir = './%s_fcn/'%args.dataset
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    epochs = {'SUN':80,'NYU':200}[args.dataset]
    train_dataset = RGBD(args.dataset,'train')
    train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers)
    val_dataset = RGBD(args.dataset,'val')
    val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    num_class = {'SUN':37,'NYU':40}[args.dataset]
    ignore_label = {'SUN':255,'NYU':255}[args.dataset]
    loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
    patience = {'SUN':30,'NYU':50}[args.dataset]
    
    print('Train sample number: %d' % len(train_dataset))
    print('Val sample number: %d' % len(val_dataset))
    ############################################################

    #net = network(640,num_classes = num_class,resnet_factory = models.resnet152, freeze_resnet=False)
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    #vgg_model_d = VGGNet(requires_grad=True, remove_fc=True)
    net = FCN8s(pretrained_net=vgg_model, n_class=num_class)
    
    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'
    if os.path.exists(args.resume):
        print('loading checkpoint %s'%(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        best_val_loss = checkpoint['best_val_loss']
        net.load_state_dict(checkpoint['state_dict'])
        log_mode = 'a'

    net = net.cuda()
    loss = loss.cuda()
    cudnn.benchmark = True
    #print(net.named_parameter())
    net = DataParallel(net)


    
    #for p in module.parameters():
        #p.requires_grad = True
    optimizer = torch.optim.SGD(
        net.parameters(), lr, momentum=0.9, weight_decay=1e-4)
    lrs = LRScheduler(
        lr, patience=patience, factor=0.5, min_lr=0.5*0.5*0.5 * lr, best_loss=best_val_loss)
    #with torch.no_grad():
    #    val_metrics, val_time, val_iou, val_acc, val_mean_acc = validate(val_loader, net, loss, 0, num_class)
    #print (val_iou, val_acc, val_mean_acc)
    for epoch in range(start_epoch, epochs + 1):
        train_metrics, train_time = train(train_loader, net, loss, optimizer,
                                          lr)
        with torch.no_grad():
            val_metrics, val_time, val_iou,  val_acc, val_mean_acc = validate(val_loader, net, loss, epoch, num_class)

        print_log(
            epoch,
            lr,
            train_metrics,
            train_time,
            val_metrics,
            val_time,
            val_iou,  
            val_acc, 
            val_mean_acc, 
            save_dir=save_dir,
            log_mode=log_mode)

        val_loss = np.mean(val_metrics)
        lr = lrs.update_by_rule(val_loss)
        if val_loss < best_val_loss or epoch % 10 == 0 or lr is None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
            torch.save({
                'epoch': epoch,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'lr': lr,
                'best_val_loss': best_val_loss
            }, os.path.join(save_dir, 'ckpt_%03d.ckpt' % epoch))

        if lr is None:
            print('Training is early-stopped')
            break
