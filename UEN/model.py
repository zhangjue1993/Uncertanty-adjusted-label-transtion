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

def Conv3(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=1, padding=1, bias=False),nn.ReLU(),nn.Dropout2d(p=0.2))
def Conv1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=1, padding=0, bias=False),nn.ReLU())
def upconv(in_channels, out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, 
                     stride=2, padding=0,output_padding=0, bias= False), nn.ReLU(),nn.Dropout2d(p=0.2))

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



class RAN_un(nn.Module):
    def __init__(self):
        super(RAN_un, self).__init__()

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

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        #self.dropout = nn.Dropout(p=0.1)
        #self.dropout = nn.Dropout2d(p=0.2)
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
        #logits = self.out_conv3(self.out_conv2(self.out_conv1(p1)))
        #out  = self.softmax(out_logits)

        logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        logvar = torch.exp(logvar/2)
        # print(logits.size())
        epsilon = torch.randn_like(logvar)
        prev_attn = epsilon * logvar
        out_logits = logits + prev_attn
        seg_out = self.softmax(out_logits)
        out = self.softmax(logits)

        return seg_out, logvar,out

class RAN(nn.Module):
    def __init__(self):
        super(RAN, self).__init__()

        down_feature = [32, 64, 128, 512]
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

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
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

#0145
# class RAN_un_com(nn.Module):
#     def __init__(self):
#         super(RAN_un_com, self).__init__()

#         down_feature = [64, 128, 256, 1024]
#         up_feature = 256
#         in_channels = 3
#         out_channels = 2
#         self.in_conv1 = Conv3(in_channels, 32)
#         self.relu = nn.ReLU(inplace=True)
#         self.in_conv2 = Conv3(32, 32)

#         self.maxpool = nn.MaxPool2d(2)
#         self.RFA1 = RFA(32, down_feature[0])
#         self.RFA2 = RFA(down_feature[0], down_feature[1])
#         self.RFA3 = RFA(down_feature[1], down_feature[2])
#         self.RFA4 = RFA(down_feature[2], down_feature[3])
        
#         self.smooth1 = Conv1(down_feature[2], up_feature)
#         self.smooth2 = Conv1(down_feature[1], up_feature)
#         self.smooth3 = Conv1(down_feature[0], up_feature)
#         self.smooth4 = Conv1(32, up_feature)

#         self.conv4_ = Conv3(down_feature[3], up_feature)
#         self.deconv4 = upconv(up_feature, up_feature)
#         self.conv3_ = Conv3(up_feature, up_feature)
#         self.deconv3 = upconv(up_feature, up_feature)
#         self.conv2_ = Conv3(up_feature, up_feature)
#         self.deconv2 = upconv(up_feature, up_feature)
#         self.conv1_ = Conv3(up_feature, up_feature)
#         self.deconv1 = upconv(up_feature, up_feature)


#         self.out_conv1 = Conv3(up_feature, 64)
#         self.out_conv2 = Conv3(64, 32)
#         self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

#         self.sig_conv1 = Conv3(up_feature, 64)
#         self.sig_conv2 = Conv3(64, 32)
#         self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dropout1 = nn.Dropout2d(p=0.5)
#         self.dropout2 = nn.Dropout2d(p=0.5)
#         self.dropout3 = nn.Dropout2d(p=0.5)
#         self.dropout4 = nn.Dropout2d(p=0.5)
#         self.dropout5 = nn.Dropout2d(p=0.5)
#         self.dropout6 = nn.Dropout2d(p=0.5)
#         self.dropout7 = nn.Dropout2d(p=0.5)
#         self.dropout8 = nn.Dropout2d(p=0.5)
#         self.dropout9 = nn.Dropout2d(p=0.5)
#         self.dropout10 = nn.Dropout2d(p=0.5)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):

#         c1 = self.in_conv2(self.in_conv1(x))
        
#         c2 = self.RFA1(self.maxpool(c1))
#         c3 = self.RFA2(self.maxpool(c2))
#         c4 = self.RFA3(self.maxpool(c3))
#         c5 = self.RFA4(self.maxpool(c4))

#         p5 = self.dropout1(self.conv4_(c5))
#         p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
#         p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
#         p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
#         p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
#         logits = self.out_conv3(self.dropout7(self.out_conv2(self.dropout6(self.out_conv1(p1)))))


#         logvar = self.sig_conv3(self.dropout9(self.sig_conv2(self.dropout8(self.sig_conv1(p1)))))
#         logvar = torch.exp(logvar/2)
#         epsilon = torch.randn_like(logvar)
#         prev_attn = epsilon * logvar
#         out_logits = logits + prev_attn
#         seg_out = self.softmax(out_logits)

#         return seg_out, logvar

#1823 
# class RAN_un_com(nn.Module):
#     def __init__(self):
#         super(RAN_un_com, self).__init__()

#         down_feature = [64, 128, 256, 1024]
#         up_feature = 256
#         in_channels = 3
#         out_channels = 2
#         self.in_conv1 = Conv3(in_channels, 32)
#         self.relu = nn.ReLU(inplace=True)
#         self.in_conv2 = Conv3(32, 32)

#         self.maxpool = nn.MaxPool2d(2)
#         self.RFA1 = RFA(32, down_feature[0])
#         self.RFA2 = RFA(down_feature[0], down_feature[1])
#         self.RFA3 = RFA(down_feature[1], down_feature[2])
#         self.RFA4 = RFA(down_feature[2], down_feature[3])
        
#         self.smooth1 = Conv1(down_feature[2], up_feature)
#         self.smooth2 = Conv1(down_feature[1], up_feature)
#         self.smooth3 = Conv1(down_feature[0], up_feature)
#         self.smooth4 = Conv1(32, up_feature)

#         self.conv4_ = Conv3(down_feature[3], up_feature)
#         self.deconv4 = upconv(up_feature, up_feature)
#         self.conv3_ = Conv3(up_feature, up_feature)
#         self.deconv3 = upconv(up_feature, up_feature)
#         self.conv2_ = Conv3(up_feature, up_feature)
#         self.deconv2 = upconv(up_feature, up_feature)
#         self.conv1_ = Conv3(up_feature, up_feature)
#         self.deconv1 = upconv(up_feature, up_feature)


#         self.out_conv1 = Conv3(up_feature, 64)
#         self.out_conv2 = Conv3(64, 32)
#         self.out_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)

#         self.sig_conv1 = Conv3(up_feature, 64)
#         self.sig_conv2 = Conv3(64, 32)
#         self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
#         self.dropout1 = nn.Dropout2d(p=0.2)
#         self.dropout2 = nn.Dropout2d(p=0.3)
#         self.dropout3 = nn.Dropout2d(p=0.3)
#         self.dropout4 = nn.Dropout2d(p=0.3)
#         self.dropout5 = nn.Dropout2d(p=0.3)
#         self.dropout6 = nn.Dropout2d(p=0.1)
#         self.dropout7 = nn.Dropout2d(p=0.1)
#         self.dropout8 = nn.Dropout2d(p=0.3)
#         self.dropout9 = nn.Dropout2d(p=0.2)
#         self.dropout10 = nn.Dropout2d(p=0.2)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):

#         c1 = self.in_conv2(self.in_conv1(x))
        
#         c2 = self.RFA1(self.maxpool(c1))
#         c3 = self.RFA2(self.maxpool(c2))
#         c4 = self.dropout9(self.RFA3(self.maxpool(c3)))
#         c5 = self.dropout10(self.RFA4(self.maxpool(c4)))

#         p5 = self.dropout1(self.conv4_(c5))
#         p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
#         p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
#         p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
#         p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
#         logits = self.out_conv3(self.dropout7((self.out_conv2(self.dropout6(self.out_conv1(p1))))))


#         logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
#         logvar = torch.exp(logvar/2)
#         epsilon = torch.randn_like(logvar)
#         prev_attn = epsilon * logvar
#         out_logits = logits + prev_attn
#         seg_out = self.softmax(out_logits)

#         return seg_out, logvar

class RAN_un_com(nn.Module):
    def __init__(self):
        super(RAN_un_com, self).__init__()

        down_feature = [64, 128, 256, 1024]
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

        self.sig_conv1 = Conv3(up_feature, 64)
        self.sig_conv2 = Conv3(64, 32)
        self.sig_conv3 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.1)
        self.dropout2 = nn.Dropout2d(p=0.3)
        self.dropout3 = nn.Dropout2d(p=0.3)
        self.dropout4 = nn.Dropout2d(p=0.3)
        self.dropout5 = nn.Dropout2d(p=0.3)
        self.dropout6 = nn.Dropout2d(p=0.1)
        self.dropout7 = nn.Dropout2d(p=0.1)
        self.dropout8 = nn.Dropout2d(p=0.3)
        self.dropout9 = nn.Dropout2d(p=0.3)
        self.dropout10 = nn.Dropout2d(p=0.3)
        self.dropout11 = nn.Dropout2d(p=0.3)
        self.dropout12 = nn.Dropout2d(p=0.3)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        c1 = self.in_conv2(self.in_conv1(x))
        
        c2 = self.dropout11(self.RFA1(self.maxpool(c1)))
        c3 = self.dropout12(self.RFA2(self.maxpool(c2)))
        c4 = self.dropout9(self.RFA3(self.maxpool(c3)))
        c5 = self.dropout10(self.RFA4(self.maxpool(c4)))

        p5 = self.dropout1(self.conv4_(c5))
        p4 = self.dropout2(self.smooth1(c4)+self.deconv4(p5))
        p3 = self.dropout3(self.smooth2(c3)+self.deconv3(self.conv3_(p4)))
        p2 = self.dropout4(self.smooth3(c2)+self.deconv2(self.conv2_(p3)))
        p1 = self.dropout5(self.smooth4(c1)+self.deconv1(self.conv1_(p2)))
        logits = self.out_conv3(self.dropout7((self.out_conv2(self.dropout6(self.out_conv1(p1))))))


        logvar = self.sig_conv3(self.sig_conv2(self.sig_conv1(p1)))
        logvar = torch.exp(logvar/2)
        epsilon = torch.randn_like(logvar)
        prev_attn = epsilon * logvar
        out_logits = logits + prev_attn
        seg_out = self.softmax(out_logits)

        return seg_out, logvar


