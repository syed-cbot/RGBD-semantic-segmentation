import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from .blocks import (RefineNetBlock, ResidualConvUnit,
                      RefineNetBlockImprovedPooling, MMF)
import torch

class BaseRefineNet4Cascade(nn.Module):
    def __init__(self,
                 input_shape,
                 refinenet_block,
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
        super().__init__()

        input_channel, input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        resnet = resnet_factory(pretrained=pretrained)

        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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

    def forward(self, x):

        layer_1 = self.layer1(x)
        layer_2 = self.layer2(layer_1)
        layer_3 = self.layer3(layer_2)
        layer_4 = self.layer4(layer_3)

        layer_1_rn = self.layer1_rn(layer_1)
        layer_2_rn = self.layer2_rn(layer_2)
        layer_3_rn = self.layer3_rn(layer_3)
        layer_4_rn = self.layer4_rn(layer_4)

        path_4 = self.refinenet4(layer_4_rn)
        path_3 = self.refinenet3(path_4, layer_3_rn)
        path_2 = self.refinenet2(path_3, layer_2_rn)
        path_1 = self.refinenet1(path_2, layer_1_rn)
        out = self.output_conv(path_1)
        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False)

    def named_parameter(self):
        """Returns parameters that requires a gradident to update."""
        return (p for p in super().named_parameters() if p[1].requires_grad)


class RefineNet4CascadePoolingImproved(BaseRefineNet4Cascade):
    def __init__(self,
                 input_shape,
                 num_classes=1,
                 features=256,
                 resnet_factory=models.resnet101,
                 pretrained=True,
                 freeze_resnet=True):
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

class RefineNet4CascadePoolingImprovedgennerateold(nn.Module):
    def __init__(self,input_size,refinenet_block = RefineNetBlockImprovedPooling,num_classes=1,features=256,resnet_factory=models.resnet101,pretrained=True,freeze_resnet=True):
        super(RefineNet4CascadePoolingImprovedgennerateold, self).__init__()

        #input_size = input_shape

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        resnet = resnet_factory(pretrained=pretrained)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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

        self.refinenet4_d = RefineNetBlock(2 * features,
                                         (2 * features, input_size // 32))
        self.refinenet3_d = RefineNetBlock(features,
                                         (2 * features, input_size // 32),
                                         (features, input_size // 16))
        self.refinenet2_d = RefineNetBlock(features,
                                         (features, input_size // 16),
                                         (features, input_size // 8))
        self.refinenet1_d = RefineNetBlock(features, (features, input_size // 8),
                                         (features, input_size // 4))

        self.output_conv_d = nn.Sequential(
            ResidualConvUnit(features), ResidualConvUnit(features),
            nn.Conv2d(
                features,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True))


    def forward(self, x):
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
        out = self.output_conv(path_1)

        path_4_d = self.refinenet4_d(layer_4_rn_d)
        path_3_d = self.refinenet3_d(path_4_d, layer_3_rn_d)
        path_2_d = self.refinenet2_d(path_3_d, layer_2_rn_d)
        path_1_d = self.refinenet1_d(path_2_d, layer_1_rn_d)
        out_d = self.output_conv_d(path_1_d)

        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False),nn.Sigmoid()(F.interpolate(out_d, scale_factor=4, mode='bilinear', align_corners=False))

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for i in layer.parameters():
                    i.requires_grad = False
class RefineNet4CascadePoolingImprovedgennerate(nn.Module):
    def __init__(self,input_size,refinenet_block = RefineNetBlockImprovedPooling,num_classes=1,features=256,resnet_factory=models.resnet101,pretrained=True,freeze_resnet=True):
        super(RefineNet4CascadePoolingImprovedgennerate, self).__init__()

        if input_size % 32 != 0:
            raise ValueError("{} not divisble by 32".format(input_shape))

        resnet = resnet_factory(pretrained=pretrained)
        self.layer1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                    resnet.maxpool, resnet.layer1)

        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

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


    def forward(self, x):
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
        out = self.output_conv(path_1)

        path_4_d = self.refinenet4(layer_4_rn_d)
        path_3_d = self.refinenet3(path_4_d, layer_3_rn_d)
        path_2_d = self.refinenet2(path_3_d, layer_2_rn_d)
        path_1_d = self.refinenet1(path_2_d, layer_1_rn_d)
        out_d = self.output_conv_d(path_1_d)

        return F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=False),nn.Sigmoid()(F.interpolate(out_d, scale_factor=4, mode='bilinear', align_corners=False))

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for i in layer.parameters():
                    i.requires_grad = False

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = RefineNet4CascadePoolingImprovedDepth(640,num_classes = 40,resnet_factory = models.resnet152, freeze_resnet=False).to(device)
    # ckpt = torch.load('ckpt_041.ckpt')
    # net.load_state_dict(ckpt['state_dict'])

    #net.eval()
    x = torch.rand(1, 3, 640, 640, device=device)
    y = net(x)
    print(y.shape)
    z = torch.rand(1, 1, 640, 640, device=device)
    y = net(x,z)
    print(y.shape)
    # torch.onnx.export(net, x, "temp.onnx", verbose=False)
