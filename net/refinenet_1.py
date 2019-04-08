import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .blocks import (RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling, MMF)
import torch
#from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from .smooth_cross_entropy_loss import SmoothCrossEntropyLoss
class L_ZL_R(nn.Module):
    def __init__(self,ignore_label,num_class,m=8,d=10):
        super(L_ZL_R, self).__init__()
        self.num_class = num_class
        self.ignore_label = ignore_label
        self.m = m
        self.d = d
    def forward(self,pred, heatmaps,depth,loss_1,pred_depth,epoch):
        b,w,h = depth.size() 
        #pred = torch.max(prediction,1)[1]
        times = torch.tensor(.0).to(depth.device)
        loss = torch.tensor(.0).to(depth.device)

        log_prediction_d = torch.log(pred_depth*255+1)
        log_gt = torch.log(depth+1)
        N = b*w*h
        log_d_diff = log_prediction_d - log_gt
        s1 = torch.sum(torch.pow(log_d_diff,2))/N 
        s2 = torch.pow(torch.sum(log_d_diff),2) /(N*N) 
        #print (log_gt.size(), log_prediction_d.size(),log_gt.max(), log_prediction_d.max(),log_gt.min(), log_prediction_d.min(),s1-s2)
        data_loss = s1 - 0.5*s2
        #R = torch.abs(log_d_diff)
        #etaR = epoch/100*(R.max()-R.median())+R.median()
        #R_ = R*(R<etaR).float()
        #LR = (loss_1*R_).mean()
        LR = (loss_1*torch.abs(log_d_diff)).sum()/((heatmaps!=self.ignore_label).float().sum())

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
                        #Z = patch_loss
                        #etaZ = epoch/100*(Z.max()-Z.median())+Z.median()
                        #Z_ = Z*(Z<etaZ).float()
                        #loss += (torch.abs(one_hot_label-one_hot_pred).sum()*Z_).mean()

                        loss += torch.abs(one_hot_label-one_hot_pred).sum()*(patch_loss.mean())
                        times += 1
                    else:
                        continue
        #if times>0:
        #print (data_loss, LR ,loss/(times+10e-4))
        return data_loss, LR ,loss/(times+10e-4)
        #else:
        #    return data_loss, LR , torch.zeros((1)).to(depth.device)*loss
class BaseRefineNet4Cascade(nn.Module):
    def __init__(self,
                 input_shape,
                 refinenet_block,
                 num_classes=1,
                 features=256,
                 ignore_label = 255,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True,
                  loss = 'soft'):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__()

        input_channel, input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        resnet = resnet_factory(pretrained=pretrained)
        self.ignore_label = ignore_label
        self._loss = SmoothCrossEntropyLoss(reduction='none',ig_label=ignore_label) if loss=='soft' else nn.CrossEntropyLoss(ignore_index=ignore_label,reduction='none')
        #self._loss = nn.CrossEntropyLoss(ignore_index=ignore_label,reduction='none')
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)
        self.L_zr = L_ZL_R(ignore_label,num_classes)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.relu = nn.ReLU(inplace=True)
        if freeze_resnet:
            layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.layer1_rn = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)


        self.layer1_rn_d = nn.Conv2d(
            256, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_rn_d = nn.Conv2d(
            512, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_rn_d = nn.Conv2d(
            1024, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_rn_d = nn.Conv2d(
            2048, 2 * features, kernel_size=3, stride=1, padding=1, bias=False)

        self.refinenet4 = RefineNetBlock(2 * features,
                                         (2 * features, input_size // 32))
        self.refinenet3 = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2 = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1 = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))

        self.output_conv = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                num_classes,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

        self.output_conv_d = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))

    def forward(self, x,epoch,depth=None,label=None, train=True):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        layer_1_rn_d = self.layer1_rn_d(layer_1)
        layer_2_rn_d = self.layer2_rn_d(layer_2)
        layer_3_rn_d = self.layer3_rn_d(layer_3)
        layer_4_rn_d = self.layer4_rn_d(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        path_4_d = self.refinenet4(layer_4_rn_d)
        path_3_d = self.refinenet3(path_4_d, layer_3_rn_d)
        path_2_d = self.refinenet2(path_3_d, layer_2_rn_d)
        path_1_d = self.refinenet1(path_2_d, layer_1_rn_d)
        out_d = self.output_conv_d(path_1_d)
        out = self.output_conv(path_1)

        
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)
        out_label = torch.max(out,1)[1]
        pred_depth = nn.Sigmoid()(F.interpolate(out_d, scale_factor=4, mode='bilinear', align_corners=False)).squeeze(1)



        if label is not None:
            loss_1 = self._loss(out, label)
            loss_2,loss_3,loss_4 = self.L_zr(out_label,label,depth,loss_1,pred_depth,epoch)

        
        # NYU 1 1 2 0.5 


            loss_output = loss_1.sum()/((label!=self.ignore_label).float().sum())+ 2*loss_3 + 0.5*loss_4 if epoch>10 else  loss_1.sum()/((label!=255).float().sum()) + loss_2 + 0.5*loss_4

        if label is not None:
            if train:
                return loss_output,loss_1.sum()/((label!=self.ignore_label).float().sum()), loss_2,loss_3,loss_4
            else:
                return out,loss_output, loss_1.sum()/((label!=self.ignore_label).float().sum()), loss_2,loss_3,loss_4
        # return out_segm
        return out

    def named_parameter(self):
        """Returns parameters that requires a gradident to update."""
        return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 ignore_label = 255,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True,loss='soft'):
        """Multi-path 4-Cascaded RefineNet for image segmentation with improved pooling
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlockImprovedPooling,
            num_classes=num_classes,
            features=features,
            ignore_label = ignore_label,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet,loss = loss)
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for i in layer.parameters():
                    i.requires_grad = False
                #in_planes = layer.num_features
                #layer = BatchNorm2d(in_planes, affine=True, eps=1e-5, momentum=0.1)

class RefineNet4Cascade(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
        """Multi-path 4-Cascaded RefineNet for image segmentation
        Args:
            input_shape ((int, int)): (channel, size) assumes input has
                equal height and width
            refinenet_block (block): RefineNet Block
            num_classes (int, optional): number of classes
            features (int, optional): number of features in refinenet
            resnet_factory (func, optional): A Resnet model from torchvision.
                Default: models.resnet101
            pretrained (bool, optional): Use pretrained version of resnet
                Default: True
            freeze_resnet (bool, optional): Freeze resnet model
                Default: True
        Raises:
            ValueError: size of input_shape not divisible by 32
        """
        super().__init__(
            input_shape,
            RefineNetBlock,
            num_classes=num_classes,
            features=features,
            resnet_factory=resnet_factory,
            pretrained=pretrained,
            freeze_resnet=freeze_resnet)
            
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for i in layer.parameters():
                    i.requires_grad = False


