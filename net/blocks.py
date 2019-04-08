import torch.nn as nn
import torch

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_feats, *shapes):
        super().__init__()

        _, max_size = max(shapes, key=lambda x: x[1])

        self.scale_factors = []
        for i, shape in enumerate(shapes):
            feat, size = shape
            if max_size % size != 0:
                raise ValueError("max_size not divisble by shape {}".format(i))

            self.scale_factors.append(max_size // size)
            self.add_module(
                "resolve{}".format(i),
                nn.Conv2d(
                    feat,
                    out_feats,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False))

    def forward(self, *xs):

        output = self.resolve0(xs[0])
        if self.scale_factors[0] != 1:
            output = nn.functional.interpolate(
                output,
                scale_factor=self.scale_factors[0],
                mode='bilinear',
                align_corners=True)

        for i, x in enumerate(xs[1:], 1):
            output += self.__getattr__("resolve{}".format(i))(x)
            if self.scale_factors[i] != 1:
                output = nn.functional.interpolate(
                    output,
                    scale_factor=self.scale_factors[i],
                    mode='bilinear',
                    align_corners=True)

        return output


class ChainedResidualPool(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 4):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 4):
            path = self.__getattr__("block{}".format(i))(path)
            x = x + path

        return x


class ChainedResidualPoolImproved(nn.Module):
    def __init__(self, feats):
        super().__init__()

        self.relu = nn.ReLU(inplace=True)
        for i in range(1, 5):
            self.add_module(
                "block{}".format(i),
                nn.Sequential(
                    nn.Conv2d(
                        feats,
                        feats,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False),
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2)))

    def forward(self, x):
        x = self.relu(x)
        path = x

        for i in range(1, 5):
            path = self.__getattr__("block{}".format(i))(path)
            x  = out+ path

        return x


class BaseRefineNetBlock(nn.Module):
    def __init__(self, features, residual_conv_unit, multi_resolution_fusion,
                 chained_residual_pool, *shapes):
        super().__init__()

        for i, shape in enumerate(shapes):
            feats = shape[0]
            self.add_module(
                "rcu{}".format(i),
                nn.Sequential(
                    residual_conv_unit(feats), residual_conv_unit(feats)))

        if len(shapes) != 1:
            self.mrf = multi_resolution_fusion(features, *shapes)
        else:
            self.mrf = None

        self.crp = chained_residual_pool(features)
        self.output_conv = residual_conv_unit(features)

    def forward(self, *xs):
        rcu_xs = []

        for i, x in enumerate(xs):
            rcu_xs.append(self.__getattr__("rcu{}".format(i))(x))

        if self.mrf is not None:
            out = self.mrf(*rcu_xs)
        else:
            out = rcu_xs[0]

        out = self.crp(out)
        return self.output_conv(out)


class RefineNetBlock(BaseRefineNetBlock):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPool, *shapes)


class RefineNetBlockImprovedPooling(nn.Module):
    def __init__(self, features, *shapes):
        super().__init__(features, ResidualConvUnit, MultiResolutionFusion,
                         ChainedResidualPoolImproved, *shapes)

class MMF(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.dropout_rgb = nn.Dropout(p=0.5, inplace=False)
        self.dropout_d = nn.Dropout(p=0.5, inplace=False)

        self.conv1_rgb = nn.Conv2d(
            features, features, kernel_size=1, stride=1)
        self.conv1_d = nn.Conv2d(
            features, features, kernel_size=1, stride=1)
        self.conv2_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv5_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv6_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv7_rgb = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv7_d = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=False)

        self.maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding = 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x_rgb, x_d):
        if torch.sum(x_d) == 0:
            out = self.dropout_rgb(x_rgb)
            out = self.conv1_rgb(out)

            out1 = self.relu(out)
            out1 = self.conv2_rgb(out1)
            out1 = self.relu(out1)
            out1 = self.conv3_rgb(out1)
            out  = out+ out1

            out1 = self.relu(out)
            out1 = self.conv4_rgb(out1)
            out1 = self.relu(out1)
            out1 = self.conv5_rgb(out1)
            out  = out+ out1

            out = self.conv6_rgb(out)
            out = self.relu(out)
            out1 = self.maxpool5(out)
            out1 = self.conv7_rgb(out1)
            out  = out+ out1
            return out
        else:
            out = self.dropout_rgb(x_rgb)
            out = self.conv1_rgb(out)
            out_d = self.dropout_d(x_d)
            out_d = self.conv1_d(out_d)

            out1 = self.relu(out)
            out1 = self.conv2_rgb(out1)
            out1 = self.relu(out1)
            out1 = self.conv3_rgb(out1)
            out  = out+ out1
            out1_d = self.relu(out_d)
            out1_d = self.conv2_d(out1_d)
            out1_d = self.relu(out1_d)
            out1_d = self.conv3_d(out1_d)
            out_d  = out_d+ out1_d

            out1 = self.relu(out)
            out1 = self.conv4_rgb(out1)
            out1 = self.relu(out1)
            out1 = self.conv5_rgb(out1)
            out  = out+ out1
            out1_d = self.relu(out_d)
            out1_d = self.conv4_d(out1_d)
            out1_d = self.relu(out1_d)
            out1_d = self.conv5_d(out1_d)
            out_d  = out_d+ out1_d

            out = self.conv6_rgb(out)
            out_d = self.conv6_d(out_d)

            out = self.relu(out+out_d)
            out1 = self.maxpool5(out)
            out1 = self.conv7_rgb(out1)
            out  = out+ out1
            return out