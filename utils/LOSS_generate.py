import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import cv2
cmap = np.load('./utils/cmap.npy')
class depthloss(nn.Module):
    def __init__(self,ignore_label, C = 2):
        super(depthloss, self).__init__()
        self.ignore_label = ignore_label
        self.C = C
    def forward(self,label,depth,pred_depth): 
        #print (label.size(),depth.size(),pred_depth.size())     
        #mask = torch.ones((depth.size()))
        
        log_prediction_d = torch.log(pred_depth*255+1)
        log_gt = torch.log(depth+1)
        N = depth.size(0)*depth.size(1)*depth.size(2)
        log_d_diff = log_prediction_d - log_gt
        #log_d_diff = torch.mul(log_d_diff, mask)
        #print ('log_d_diff',log_d_diff)
        s1 = torch.sum( torch.pow(log_d_diff,2) )/N 
        s2 = torch.pow(torch.sum(log_d_diff)/N,2)  
        #print (mask,log_prediction_d,log_gt,log_d_diff,s1,s2)
        data_loss = s1 - s2
        if self.C == 1:
            return data_loss
        else:
            return data_loss,log_d_diff




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