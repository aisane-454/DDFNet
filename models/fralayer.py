import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, frenum=8):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.frenum = frenum
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7
        self.dct_layer = MultiSpectralDCTLayer(self.dct_h, self.dct_w, channel, frenum)
        self.adv = ADAP(in_channels=channel,  kernel_size=1, stride=1, group=frenum**2)
    def forward(self, x):
        n,c,h,w = x.shape 
        size = x.size()[2:]
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        per = self.dct_layer(x_pooled)
        dct = torch.sum(per, dim=[2,3]) #B C 

        dct = dct.view(n, c, 1, 1)

        ad_dct, v_map = self.adv(dct)

        return ad_dct, v_map


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, channel, frenum):
        super(MultiSpectralDCTLayer, self).__init__()
        
        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, channel, frenum))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        a = x * self.weight
        return a

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, channel, frenum):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        
        c_part = channel // (frenum * frenum)
        
        for i in range(frenum):
            for j in range(frenum):
                for t_x in range(tile_size_x):
                    for t_y in range(tile_size_y):
                        dct_filter[(i*c_part*frenum + j*c_part): (i*c_part*frenum + (j+1)*c_part), t_x, t_y] = self.build_filter(t_x, i, tile_size_x) * self.build_filter(t_y, j, tile_size_y)
                        
        return dct_filter

class ADAP(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, pad_type='reflect', group=2, **kwargs):
        super(ADAP, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(in_channels, group*kernel_size*kernel_size, kernel_size=kernel_size, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group*kernel_size*kernel_size)
        self.softmax = nn.Softmax(dim=1)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, y):
        sigma = self.conv(y)
        sigma = self.bn(sigma)
        sigma = self.softmax(sigma)
        n,c,h,w = sigma.shape

        sigma = sigma.reshape(n,1,c,h*w)

        n,c,h,w = y.shape
        y = F.unfold(y, kernel_size=self.kernel_size).reshape((n,c,self.kernel_size*self.kernel_size,h*w))

        n,c1,p,q = y.shape
        y = y.permute(1,0,2,3).reshape(self.group, c1//self.group, n, p, q).permute(2,0,1,3,4)

        n,c2,p,q = sigma.shape
        sigma = sigma.permute(2,0,1,3).reshape((p//(self.kernel_size*self.kernel_size), self.kernel_size*self.kernel_size,n,c2,q)).permute(2,0,3,1,4)
        
        y = torch.sum(y*sigma, dim=3).reshape(n,c1,h,w)
        
        return y, sigma
