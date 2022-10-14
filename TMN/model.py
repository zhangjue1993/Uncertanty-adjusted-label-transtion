import torch
import torchvision.models as v_models
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torchvision.ops import roi_pool, RoIPool
from torchsummary import summary

#from data_pre import myDataSet

BATCH_SIZE = 1
R = 10


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def Conv3(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),nn.ReLU())
def Conv1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=1, padding=0, bias=False),nn.ReLU())
def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, 
                     stride=2, padding=0,output_padding=0, bias= False), nn.ReLU())

class RFA(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(RFA, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rb1 = self.RB()
        self.rb2 = self.RB()
        self.conv1 = Conv3(self.in_channels, in_channels)
        self.conv2 = Conv3(3*self.in_channels, 3*self.in_channels)
        self.conv3 = Conv1(3*self.in_channels, self.out_channels)
    
    def RB(self):
        rb_1 = Conv3(self.in_channels, self.in_channels)
        rb_2 = Conv3(self.in_channels, self.in_channels)
        rb = nn.Sequential(rb_1, rb_2)
        return rb

    def forward(self, x):
        c1 = self.rb1(x) #RB1
        c2 = self.rb2(c1 + x) #RB2
        c3 = self.conv1(c2 + c1 + x)
        concat123 = torch.cat((x,c1+x, c3), 1)
        c4 = self.conv2(concat123)
        c5 = self.conv3(c4)
        return c5

class RAN_test(nn.Module):
    def __init__(self):
        super(RAN_test, self).__init__()

        
        self.smooth1 = Conv3(3, 32)
        self.smooth2 = Conv3(32, 64)
        self.smooth3 = Conv3(64, 128)
        self.smooth4 = Conv3(128, 128)
        self.smooth5 = Conv3(128, 64)
        self.smooth6 = Conv3(64, 32)
        self.conv = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        p1 = self.smooth1(x)
        p2 = self.smooth2(p1)
        p3 = self.smooth3(p2)
        p4 = self.smooth4(p3)
        p5 = self.smooth5(p4)
        p6 = self.smooth6(p5)
        
        p7 = self.conv(p6)

        out_logits = p7
        out  = self.softmax(p7)

        # sigma = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        # out_10 = torch.zeros([10, sigma.size()[0], 1, 256, 256]).cuda()
        # for i in range(10):
        #     epsilon = torch.normal(torch.zeros_like(sigma), 1)
        #     out_10[i,:,:,:,:] = seg_out + sigma*epsilon
        # seg_out = torch.mean(out_10, axis = 0, keepdim = False)
        return out_logits,out




class RAN(nn.Module):
    def __init__(self):
        super(RAN, self).__init__()

        down_feature =[128, 256, 512, 1024]
        up_feature = 256
        in_channels = 3
        out_channels = 2
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])
        
        self.smooth1 = Conv1(down_feature[2], up_feature)
        self.smooth2 = Conv1(down_feature[1], up_feature)
        self.smooth3 = Conv1(down_feature[0], up_feature)
        self.smooth4 = Conv1(32, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)


        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

    
        self.softmax = nn.Softmax(dim=1)



    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))

        p5 = self.conv4_(c5)
        p4 = self.smooth1(c4)+self.deconv4(p5)
        p3 = self.smooth2(c3)+self.deconv3(self.conv3_(p4))
        p2 = self.smooth3(c2)+self.deconv2(self.conv2_(p3))
        p1 = self.smooth4(c1)+self.deconv1(self.conv1_(p2))

        logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        out  = self.softmax(logits)

        return out

class RAN_u(nn.Module):
    def __init__(self):
        super(RAN_u, self).__init__()

        down_feature = [32, 64, 128, 512]
        up_feature = 128
        in_channels = 3
        out_channels = 2
        self.in_conv1 = Conv3(in_channels, 32)
        self.relu = nn.ReLU(inplace=True)
        self.in_conv2 = Conv3(32, 32)

        self.maxpool = nn.MaxPool2d(2)
        self.RFA1 = RFA(32, down_feature[0])
        self.RFA2 = RFA(down_feature[0], down_feature[1])
        self.RFA3 = RFA(down_feature[1], down_feature[2])
        self.RFA4 = RFA(down_feature[2], down_feature[3])
        
        self.smooth1 = Conv1(down_feature[2], up_feature)
        self.smooth2 = Conv1(down_feature[1], up_feature)
        self.smooth3 = Conv1(down_feature[0], up_feature)
        self.smooth4 = Conv1(32, up_feature)

        self.conv4_ = Conv3(down_feature[3], up_feature)
        self.deconv4 = upconv(up_feature, up_feature)
        self.conv3_ = Conv3(up_feature, up_feature)
        self.deconv3 = upconv(up_feature, up_feature)
        self.conv2_ = Conv3(up_feature, up_feature)
        self.deconv2 = upconv(up_feature, up_feature)
        self.conv1_ = Conv3(up_feature, up_feature)
        self.deconv1 = upconv(up_feature, up_feature)


        self.out_conv1 = Conv3(up_feature, 64)
        self.out_conv2 = Conv3(64, 32)
        self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        #self.dropout = nn.Dropout(p=0.3)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.RFA1(self.maxpool(c1))
        c3 = self.RFA2(self.maxpool(c2))
        c4 = self.RFA3(self.maxpool(c3))
        c5 = self.RFA4(self.maxpool(c4))

        p5 = self.conv4_(c5)
        p4 = self.smooth1(c4)+self.deconv4(p5)
        p3 = self.smooth2(c3)+self.deconv3(self.conv3_(p4))
        p2 = self.smooth3(c2)+self.deconv2(self.conv2_(p3))
        p1 = self.smooth4(c1)+self.deconv1(self.conv1_(p2))

        logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        out = self.softmax(logits)

        return out