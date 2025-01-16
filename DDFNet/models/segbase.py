"""Base Model for Semantic Segmentation"""
import torch.nn as nn
import torch.nn.functional as F

from .base_models.resnet import resnet18, resnet50, resnet101, resnet152
__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    """

    def __init__(self, nclass, aux, backbone='resnet50', jpu=False, pretrained_base=False, **kwargs):
        super(SegBaseModel, self).__init__()
        dilated = False if jpu else True
        self.aux = aux
        self.nclass = nclass
        self.backone = backbone
        if backbone == 'resnet18':
            self.pretrained = resnet18(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=pretrained_base, **kwargs)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        c0 = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(c0)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]

    def demo(self, x):
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y