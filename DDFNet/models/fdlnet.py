import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from .segbase import SegBaseModel
from .fcn import _FCNHead
from .fralayer import MultiSpectralAttentionLayer

__all__ = ['TestNet', 'mymodel']

class  TestNet(SegBaseModel):
    def __init__(self, nclass, criterion=None, backbone='resnet18', aux=False, pretrained_base=False, **kwargs):
        super(TestNet, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.criterion = criterion
        self.fcm = FCM(512, 128, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None)

        self.adapt = nn.Conv2d(1, 3, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, nclass)

        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['fcm', 'auxlayer'] if aux else ['fcm'])

    def forward(self, x, gts=None, segSize=None):
        size = x.size()[2:]
        x = self.adapt(x)
        c1, c2, c3, c4 = self.base_forward(x)
        fcm, v_map = self.fcm(c4)
        out = self.avgpool(fcm)        
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        
        return out

class FCM(nn.Module):
    def __init__(self, in_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(FCM, self).__init__()

        self.att = MultiSpectralAttentionLayer(in_channels, dct_h=4, dct_w=4, frenum=8)
        self.ppm = _PSPHead(out_channels=out_channels, **kwargs)
        self.fam =  _FAHead(in_channels=in_channels, inter_channels=out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        fre, v_map = self.att(x) #B 2048 1 1
        f = self.ppm(x)
        fa = self.fam(f, fre)
        seg_out = fa

        return seg_out, v_map

def _Test1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
        nn.ReLU(True)
    )

class _FreqAttentionModule(nn.Module):
    """ attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_FreqAttentionModule, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, fre):
        batch_size, _, height, width = x.size()
        fre = fre.expand_as(x)
        feat_a = x.view(batch_size, -1, height * width) #B C H*W
        feat_f_transpose = fre.view(batch_size, -1, height * width).permute(0, 2, 1) #B H*W C
        attention = torch.bmm(feat_a, feat_f_transpose)  # B C C
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new) # B C C

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width) # B C H*W
        out = self.alpha*feat_e + x
        return out


class _FAHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FAHead, self).__init__()
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_f1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.freatt = _FreqAttentionModule(inter_channels, **kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
        )

    def forward(self, x, fre):
        feat_x = self.conv_x1(x)
        feat_f = self.conv_f1(fre)

        feat_p = self.freatt(feat_x, feat_f)
        feat_p = self.conv_p2(feat_p)

        return feat_p

class _PyramidPooling(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels / 4)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(3)
        self.avgpool4 = nn.AdaptiveAvgPool2d(6)
        self.conv1 = _Test1x1Conv(in_channels, out_channels, **kwargs)
        self.conv2 = _Test1x1Conv(in_channels, out_channels, **kwargs)
        self.conv3 = _Test1x1Conv(in_channels, out_channels, **kwargs)
        self.conv4 = _Test1x1Conv(in_channels, out_channels, **kwargs)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = F.interpolate(self.conv1(self.avgpool1(x)), size, mode='bilinear', align_corners=True)
        feat2 = F.interpolate(self.conv2(self.avgpool2(x)), size, mode='bilinear', align_corners=True)
        feat3 = F.interpolate(self.conv3(self.avgpool3(x)), size, mode='bilinear', align_corners=True)
        feat4 = F.interpolate(self.conv4(self.avgpool4(x)), size, mode='bilinear', align_corners=True)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)

class _PSPHead(nn.Module):
    def __init__(self, out_channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.PSP = _PyramidPooling(in_channels=512, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(1024, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.PSP(x)
        return self.block(x)

def mymodel(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='/root/Gitdownload/awesome-semantic-segmentation-pytorch/runs/ckpt/',
            pretrained_base=True, numclass=1, **kwargs):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.
    Examples
    --------
    >>> model = get_test(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    
    model = TestNet(numclass, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device('cpu')
        checkpoint = torch.load(get_model_file('best_testnet_resnet101_night_epoch_260_mean_iu_0.53208', root=root),
                                    map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    return model