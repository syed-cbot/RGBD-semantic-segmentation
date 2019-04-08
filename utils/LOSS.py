import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2
cmap = np.load('./utils/cmap.npy')
class depthloss(nn.Module):
    def __init__(self,num_class,ignore_label):
        super(depthloss, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
    def forward(self,prediction,label,depth,pred_depth):
        length = len(label)
        pred = torch.max(prediction,1)[1].view(length,-1) 
        label = label.view(length,-1) 
        #print (pred.size(),label.size(),depth.size(),pred_depth.size())
        for j in range(length):
            for i in range(self.num_class):
                if ((label[j]==i).float().sum())>0:
                    pred_depth[j][(pred[j]==i)] += depth[j].mul((label[j]==i).float()).sum()/((label[j]==i).float().sum())

        mask = 1- ((pred==label) | (label==self.ignore_label) | (pred_depth==1)).float()
        log_prediction_d = torch.log(pred_depth+1)
        #print (log_prediction_d.min(),log_prediction_d.max())
        log_gt = torch.log(depth+1)
        #print (depth.min(),depth.max())
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        
        log_d_diff = torch.mul(log_d_diff, mask)
        #print ('log_d_diff',log_d_diff)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N 
        s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)  
        data_loss = s1 - s2
        return data_loss




class L_Z(nn.Module):
    def __init__(self,ignore_label,num_class,m=8,d=10):
        super(L_Z, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.m = m
        self.d = d
    def forward(self,prediction, heatmaps,depth,loss_1):
        b,w,h = depth.size() 
        pred = torch.max(prediction,1)[1]
        times = .0
        loss = .0
        for i in range(self.m):
            for j in range(self.m):
                mask_select = torch.zeros(depth.size()).to(depth.device)
                mask_select[:,i*w//self.m:i*w//self.m+w//self.m-1,j*h//self.m:j*h//self.m+h//self.m-1] = 1
                mask_select = mask_select*(heatmaps!=self.ignore_label).float()
                mask_label = torch.masked_select(heatmaps, mask_select.byte())
                mask_depth = torch.masked_select(depth, mask_select.byte())
                mask_loss = torch.masked_select(loss_1, mask_select.byte())
                mask_pred = torch.masked_select(pred, mask_select.byte())
                for k in range(255//self.d+1):
                    if ((mask_depth>k*self.d) & (mask_depth<((k+1)*self.d-1))).float().sum()>0:
                        mask = ((mask_depth>k*self.d) & (mask_depth<((k+1)*self.d-1)))
                        patch_label = torch.masked_select(mask_label,mask.byte())
                        patch_pred = torch.masked_select(mask_pred,mask.byte())
                        patch_loss = torch.masked_select(mask_loss,mask.byte())
                        y = torch.zeros((self.num_class)).to(depth.device)
                        y[patch_label] = 1
                        one_hot_label = y
                        y = torch.zeros((self.num_class)).to(depth.device)
                        y[patch_pred] = 1
                        one_hot_pred = y
                        #print (torch.abs(one_hot_label-one_hot_pred).sum(),patch_loss.mean())
                        loss += (torch.abs(one_hot_label-one_hot_pred).sum()*(patch_loss)).mean()
                        times += 1
                    else:
                        continue
        return loss/times



def Data_Loss(self, prediction, heatmaps, gt, mask, num_class, ignore_label):




    log_prediction_d = torch.log(prediction_d)
    log_gt = torch.log(gt)
    N = torch.sum(mask)
    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)
    s1 = torch.sum( torch.pow(log_d_diff,2) )/N 
    s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)  
    data_loss = s1 - s2
        
    return data_loss

class weightentropy(nn.Module):
    def __init__(self,num_class,ignore_label):
        super(depthloss, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
    def forward(self,prediction,label,depth,pred_depth):
        length = len(label)
        pred = torch.max(prediction,1)[1].view(length,-1) 
        label = label.view(length,-1) 
        #print (pred.size(),label.size(),depth.size(),pred_depth.size())
        for j in range(length):
            for i in range(self.num_class):
                if ((label[j]==i).float().sum())>0:
                    pred_depth[j][(pred[j]==i)] += depth[j].mul((label[j]==i).float()).sum()/((label[j]==i).float().sum())
        mask = 1- ((pred==label) | (label==self.ignore_label) | (pred_depth==0)).float()
        log_prediction_d = torch.log(pred_depth)
        log_gt = torch.log(depth)
        N = torch.sum(mask)
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N 
        s2 = torch.pow(torch.sum(log_d_diff),2)/(N*N)  
        data_loss = s1 - s2
        return data_loss




def calculatedepth(num_class,prediction,label,depth,pred_depth,output,epoch): 
    length = len(label)
    pred = torch.max(prediction,1)[1]
    for j in range(length):
        for i in range(num_class):
            if ((label[j]==i).float().sum())>0:
                pred_depth[j][(pred[j]==i)] += depth[j].mul((label[j]==i).float()).sum()/((label[j]==i).float().sum())
                #depth[j][(label[j]==i)] = depth[j].mul((label[j]==i).float()).sum()/((label[j]==i).float().sum())
    mask = 1- (pred_depth==1).float()
    
    log_prediction_d = torch.log(pred_depth)
    log_gt = torch.log(depth)
    log_d_diff = log_prediction_d - log_gt
    log_d_diff = torch.mul(log_d_diff, mask)
    output += (torch.abs(log_d_diff))
    #cv2.imwrite("depth.jpg", depth[0].cpu().numpy());
    #cv2.imwrite("pred_depth.jpg", pred_depth[0].cpu().numpy());
    #cv2.imwrite("label.jpg", (cmap[label[0].cpu().numpy().astype(np.uint8)+1]));
    #if epoch>20:
    #	output -= 1
    return output



if __name__=='__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    depth_loss = depthloss(5,-1).cuda()
    prediction = torch.from_numpy(np.array([0,0,1,2,3,4,1,2])).cuda(async=True)
    label = torch.from_numpy(np.array([-1,0,2,1,4,-1,0,-1])).cuda(async=True)
    depth = torch.from_numpy(np.array([10,9,5,2,6,4,2,1])).cuda(async=True)
    pred_depth = torch.zeros(depth.size()).cuda(async=True)

    loss = depth_loss(prediction,label,depth,pred_depth)
    print (loss)

    #x = torch.rand(1, 3, 512, 512, device=device)
    #y = net(x)
    #print(y.shape)
    # torch.onnx.export(net, x, "temp.onnx", verbose=False)
