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
from utils import helper_fig as helper
import torchvision.models as models
#================ load your net here ===================
from net.refinenet import RefineNet4CascadePoolingImproved as network_base
from net.refinenet_1_1 import RefineNet4CascadePoolingImproved as network
import torch.nn.functional as F
from net.pspnet import PSPNet
from net.deeplab_resnet import DeepLabv3_plus as network_deeplab
from net.refinenetP2T1 import RefineNet4CascadePoolingImproved as network_rdfnet
cmap = np.load('./utils/cmap.npy')
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
models_p = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(n_classes=num_class,sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

def build_network(snapshot = None, backend='resnet152'):
    epoch = 0
    backend = backend.lower()
    net = models_p[backend]()
    #net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    #net = net.cuda()
    return net, epoch

good_list = [26,56,109,223,468,489]
bad_list = [27,277,482,507,534,611]
test_list = [545]
h = 200
w = 0
t = 200
def validate(data_loader,data_loader_1,data_loader_2,data_loader_3, net,base,psp,deeplab,rdfnet, epoch, num_class):

    j = 0

    data_list = []
    label_list= []
    pred_list= []
    base_list= []
    psp_list= []
    deeplab_list= []
    rdfnet_list= []

    for batch,batch_1,batch_2,batch_3 in zip(data_loader,data_loader_1,data_loader_2,data_loader_3):
        if j not in test_list:
            j += 1
            continue
        if not os.path.exists('./draws/%s'%j):
            os.mkdir('./draws/%s'%j)
        data, heatmaps, depth = batch['image'], batch['label'], batch['depth']
        data_0 = data.cuda(async=True)
        heatmaps_0 = heatmaps.cuda(async=True)
        #depth = depth.cuda(async=True)
        prediction, pred_depth,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data_0,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        #prediction = F.interpolate(prediction, scale_factor=1, mode='bilinear', align_corners=False).cpu()
        #pred= prediction.cpu().numpy()
        pred_depth = pred_depth.squeeze().cpu().numpy()*255
        pre = base(data_0).cpu().numpy()
        #print (depth.squeeze().size())
        depth_0 = depth.squeeze().numpy()
        #depth = depth.transpose(1, 2, 0)


        #prediction_psp,_ = psp(data_0)
        #prediction_psp = prediction_psp.cpu()

        #prediction_deeplab = deeplab(data_0).cpu()

        #prediction_rdfnet,loss_output, loss_1,loss_2,loss_3,loss_4 = rdfnet(data_0,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        #prediction_rdfnet = prediction_rdfnet.cpu()
#######################################################


        
        data, heatmaps, depth = batch_1['image'], batch_1['label'], batch_1['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_1,_,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_1 = F.interpolate(prediction_1, scale_factor=5/12, mode='bilinear', align_corners=False).cpu()    
        
        data, heatmaps, depth = batch_2['image'], batch_2['label'], batch_2['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_2,_,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_2 = F.interpolate(prediction_2, scale_factor=1.25, mode='bilinear', align_corners=False).cpu()  
    
        data, heatmaps, depth = batch_3['image'], batch_3['label'], batch_3['depth']
        data = data.cuda(async=True)
        heatmaps = heatmaps.cuda(async=True)
        depth = depth.cuda(async=True)
        prediction_3,_,loss_output, loss_1,loss_2,loss_3,loss_4 = net(data,epoch,depth.squeeze(1),label=heatmaps, train=False) 
        prediction_3 = F.interpolate(prediction_3, scale_factor=0.625, mode='bilinear', align_corners=False).cpu()

        
        pred = (prediction.cpu()+prediction_1+prediction_2+prediction_3)/4
        '''
        iou_, acc_, mean_acc_ = helper.cal_metric(heatmaps_0.cpu(), pred, num_class)
        
        if iou_>0.5:
            img = helper.make_validation_img(data_0.cpu().numpy(),
                                    heatmaps_0.cpu().numpy(),
                                    pred.numpy(),
                                    prediction_base.numpy(),
                                    prediction_psp.numpy(),
                                    prediction_deeplab.numpy(),
                                    prediction_rdfnet.numpy())
            cv2.imwrite('%s/validate_%d_%.4f.png'%(save_dir,i, iou_),
                img[:, :, ::-1])
        '''
        '''
        data_list.append(data_0[0].cpu().numpy())
        label_list.append(heatmaps_0[0].cpu().numpy())
        pred_list.append(pred[0].numpy())
        base_list.append(prediction_base[0].numpy())
        psp_list.append(prediction_psp[0].numpy())
        deeplab_list.append(prediction_deeplab[0].numpy())
        rdfnet_list.append(prediction_rdfnet[0].numpy())
        '''
        prediction_1 = np.array([i.argmax(axis=0) for i in prediction_1.numpy()])
        prediction_1 = np.concatenate(prediction_1)
        prediction_1 = np.array([cmap[i.astype(np.uint8)+1] for i in prediction_1])

        prediction_2 = np.array([i.argmax(axis=0) for i in prediction_2.numpy()])
        prediction_2 = np.concatenate(prediction_2)
        prediction_2 = np.array([cmap[i.astype(np.uint8)+1] for i in prediction_2])


        prediction_3 = np.array([i.argmax(axis=0) for i in prediction_3.numpy()])
        prediction_3 = np.concatenate(prediction_3)
        prediction_3 = np.array([cmap[i.astype(np.uint8)+1] for i in prediction_3])



        img_0 = data_0.cpu().numpy()
        label_0 = heatmaps_0.cpu().numpy()
        print (label_0[0],label_0[0][180])
        pred = np.array([i.argmax(axis=0) for i in pred.numpy()])
        pred = np.concatenate(pred)
        pred = np.array([cmap[i.astype(np.uint8)+1] for i in pred])
        pre = np.array([i.argmax(axis=0) for i in pre])
        pre = np.concatenate(pre)
        pre = np.array([cmap[i.astype(np.uint8)+1] for i in pre])
        img = np.array([i*IMG_STD+IMG_MEAN for i in img_0]) 
        img *= 255
        img = img.astype(np.uint8)
    #print (img.shape)
        img = np.concatenate(img, axis=1)
    #print (img.shape)
    # label
        lab = np.concatenate(label_0)
    #cmap[segm.argmax(axis=2).astype(np.uint8)]
        lab = np.array([cmap[i.astype(np.uint8)+1] for i in lab])
        img = img.transpose(1, 2, 0)

        cv2.imwrite('./draws/%s/image%s%s.png'%(j,h,w),img[h:h+t, w:w+t, ::-1])
        cv2.imwrite('./draws/%s/label%s%s.png'%(j,h,w),lab[h:h+t, w:w+t, ::-1])
        cv2.imwrite('./draws/%s/pred_base%s%s.png'%(j,h,w),pre[h:h+t, w:w+t, ::-1])
        cv2.imwrite('./draws/%s/pred%s%s.png'%(j,h,w),pred[h:h+t, w:w+t, ::-1])
        #cv2.imwrite('./draws/%s/pred_1_%s%s.png'%(j,h,w),prediction_1[h:h+t, w:w+t, ::-1])
        #cv2.imwrite('./draws/%s/pred_2_%s%s.png'%(j,h,w),prediction_2[h:h+t, w:w+t, ::-1])
        #cv2.imwrite('./draws/%s/pred_3_%s%s.png'%(j,h,w),prediction_3[h:h+t, w:w+t, ::-1])
        cv2.imwrite('./draws/%s/depth%s%s.png'%(j,h,w),depth_0[h:h+t, w:w+t,])
        cv2.imwrite('./draws/%s/pred_depth%s%s.png'%(j,h,w),pred_depth[h:h+t, w:w+t,])
        j += 1



if __name__ == '__main__':

    workers = 8
    batch_size = 1
    
    base_lr = 1e-3
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resume', default='', help='resume training from checkpoint ...', type=str)
    parser.add_argument('-d', '--dataset', default='NYU', help='NYU or SUN', type=str)
    args = parser.parse_args()
    save_dir = './draws/'

        
    epochs = {'SUN':60,'NYU':t}[args.dataset]

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

    patience = {'SUN':15,'NYU':60}[args.dataset]


    print('Val sample number: %d' % len(val_dataset))
    ############################################################

   
    net = network((3,640),num_classes = num_class,resnet_factory = models.resnet152, freeze_resnet=False)
    net = net.cuda()
    net = DataParallel(net)
    net.eval()
    #checkpoint = torch.load('NYU.ckpt')
    #checkpoint = torch.load('NYU_best.ckpt')
    checkpoint = torch.load('NYU_LPI_7/ckpt_best.ckpt')
    net.load_state_dict(checkpoint['state_dict'])


    base = network_base((3,640),num_classes = num_class,resnet_factory = models.resnet101, freeze_resnet=False)
    base = base.cuda()
    base = DataParallel(base)
    base.eval()
    checkpoint = torch.load('NYU_base101/ckpt_best.ckpt')
    base.load_state_dict(checkpoint['state_dict'])
    
    psp, _ = build_network()
    #psp = psp.cuda()
    #psp = DataParallel(psp)
    #psp.eval()
    #checkpoint = torch.load('RGBD/NYU_pspnet/ckpt_120.ckpt')
    #psp.load_state_dict(checkpoint['state_dict'])


    deeplab = network_deeplab(nInputChannels=3, n_classes=num_class, os=16, pretrained=True)
    #deeplab = deeplab.cuda()
    #deeplab = DataParallel(deeplab)
    #deeplab.eval()
    #checkpoint = torch.load('RGBD/NYU_deeplabv3/ckpt_190.ckpt')
    #deeplab.load_state_dict(checkpoint['state_dict'])  


    rdfnet = network_rdfnet((3,640),num_classes = num_class,ignore_label = ignore_label,resnet_factory = models.resnet152, freeze_resnet=False)
    #rdfnet = rdfnet.cuda()
    #rdfnet = DataParallel(rdfnet)
    #rdfnet.eval()
    #checkpoint = torch.load('NYU_P2T1/ckpt_best.ckpt')
    #rdfnet.load_state_dict(checkpoint['state_dict'])  
    ##############################################################
    
    start_epoch = 1
    lr = base_lr
    best_val_loss = float('inf')
    log_mode = 'w'


    cudnn.benchmark = True




    optimizer = torch.optim.SGD(
        net.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    lrs = LRScheduler(
        lr, patience=patience, factor=0.5, min_lr=0.5*0.5*0.5 * lr, best_loss=best_val_loss)
    with torch.no_grad():
        validate(val_loader,val_loader_2048,val_loader_480,val_loader_840, net,base,psp,deeplab,rdfnet, 0, num_class)

    