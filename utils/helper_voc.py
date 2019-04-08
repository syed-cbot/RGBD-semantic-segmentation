import numpy as np
import torch
cmap = np.load('./utils/cmap.npy')
IMG_MEAN = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
IMG_STD = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
from utils.evalmetric import pixel_accuracy,mean_accuracy,mean_IU
def cal_iou(heatmaps, prediction):
    pred_heat = (prediction[:, 1:].max(axis=1) > prediction[:, 0]).long()
    equal = (pred_heat == heatmaps).long()
    inter = (equal * heatmaps).sum().item()
    union = heatmaps.size(0) * heatmaps.size(1) * heatmaps.size(2) - \
        (equal * (1 - heatmaps)).sum().item()
    if union==0:
        union = 1
    return inter / union

def evaluate(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def eval_metric(heatmaps, prediction, num_class):
    total_iou = 0
    acc = 0
    mean_acc = 0
    prediction = torch.max(prediction,1)[1]
    max_label = num_class
    for i in range(len(prediction)):
        pred = prediction[i]
        gt = heatmaps[i]
        acc_ = pixel_accuracy(pred.cpu().numpy(),gt.cpu().numpy())
        mean_acc_ = mean_accuracy(pred.cpu().numpy(),gt.cpu().numpy())
        iou_ = mean_IU(pred.cpu().numpy(),gt.cpu().numpy())
        acc += acc_
        total_iou  += iou_
        mean_acc += mean_acc_
    return total_iou / len(prediction), acc/ len(prediction), mean_acc/len(prediction)


def cal_metric(heatmaps, prediction, num_class):
    total_iou = 0
    total_acc = 0
    acc = 0
    mean_acc = 0
    prediction = torch.max(prediction,1)[1]
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
        for j in range(1,max_label):          
            p_acc = (pred==j)
            g_acc = (gt==j)
            match = p_acc +  g_acc
            it = torch.sum(match==2).item()
            un = torch.sum(match>0).item()
            intersect[j] += it
            union[j] += un
            #acc_label[j] += torch.sum((pred==j)).item()
            gt_label[j] += torch.sum((gt==j)).item()

        for k in range(1,max_label):
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

def make_validation_img(img_, lab, pre):
    # img (image): (np.array) batchsize * 3 * H * W
    # lab (label): (np.array) batchsize * H * W
    # pre (predition): (np.array) batchsize * H * W

    # img
    
    
    img = np.array([i*IMG_STD+IMG_MEAN for i in img_]) 
    img *= 255
    img = img.astype(np.uint8)
    #print (img.shape)
    img = np.concatenate(img, axis=1)
    #print (img.shape)
    # label
    lab = np.concatenate(lab)
    #cmap[segm.argmax(axis=2).astype(np.uint8)]
    lab = np.array([cmap[i.astype(np.uint8)+1] for i in lab])
    #lab = np.array([i*1.0/i.max()*255 for i in lab]).transpose(1, 2, 0)
    # predict
    #pre = pre[:, 1:].max(axis=1) > pre[:, 0]
    pre = np.array([i.argmax(axis=0) for i in pre])
    #print (pre.shape)
    pre = np.concatenate(pre)
    #print (pre.shape)
    #pre = np.array([(np.argmax(i.transpose(1,2,0),axis = 2))/(np.argmax(i.transpose(1,2,0),axis = 2).max())*255 for i in pre]).transpose(1, 2, 0)
    pre = np.array([cmap[i.astype(np.uint8)+1] for i in pre])

    img = img.transpose(1, 2, 0)
    #print (img.shape,lab.shape,pre.shape)
    return np.concatenate([img, lab, pre], 1)
