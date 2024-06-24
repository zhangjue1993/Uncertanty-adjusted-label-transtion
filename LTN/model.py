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



class RAN(nn.Module):
    def __init__(self):
        super(RAN, self).__init__()

        down_feature = [64, 128, 256, 1024]
        up_feature = 256
        in_channels = 4
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

        self.sigmoid = nn.Sigmoid()


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

        out = self.sigmoid(logits)
        print(out.shape)

        return out



class RAN_uncertainty(nn.Module):
    def __init__(self):
        super(RAN_uncertainty, self).__init__()

        input_nc = 3
        output_nc = 1
        self.in_conv1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        self.in_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.RFA1 = RFA()
        self.RFA2 = RFA()
        self.RFA3 = RFA()
        self.RFA4 = RFA()
        self.maxpool = nn.MaxPool2d(2)
        self.drop2 = torch.nn.Dropout2d(0.5)

        # up path
        self.up_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.agg_node1 = self.agg_node(64, 64)
        self.agg_node2 = self.agg_node(64, 64)
        self.agg_node3 = self.agg_node(64, 64)
        self.agg_node4 = self.agg_node(64, 64)

        self.out_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.out_conv2 = nn.Conv2d(32, output_nc, kernel_size=3, stride=1, padding=1)

        # unceratinty
        self.up_conv1_un = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_conv2_un = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_conv3_un = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.up_conv4_un = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.deconv1_un = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv2_un = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv3_un = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.deconv4_un = nn.ConvTranspose2d(in_channels=64, out_channels= 64,  kernel_size=2, stride=2, padding=0,output_padding=0, bias= False)

        self.agg_node1_un = self.agg_node(64, 64)
        self.agg_node2_un = self.agg_node(64, 64)
        self.agg_node3_un = self.agg_node(64, 64)
        self.agg_node4_un = self.agg_node(64, 64)

        self.out_conv1_un = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.out_conv2_un = nn.Conv2d(32, output_nc, kernel_size=3, stride=1, padding=1)


        
    def agg_node(self, in_planes, out_planes):
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )


    def forward(self, x):
        
        # down path 
        x = self.relu(self.in_conv1(x))
        x = self.relu(self.in_conv2(x)) #[8, 64, 256, 256]
        
        e1_ = self.maxpool(x)
        e1 = self.RFA1(e1_) #[8, 64, 128, 128]

        e2_ = self.maxpool(e1)
        e2 = self.RFA2(e2_) #[8, 64, 64, 64]

        e3_ = self.maxpool(e2)
        e3 = self.RFA3(e3_) #[8, 64, 32, 32]

        e4_ = self.maxpool(e3)
        e4 = self.RFA4(e4_) #[8, 64, 16, 16]

        # up path to prediction
        d1 = self.up_conv1(e4)
        d1_ = self.deconv1(d1)
        d1_out = d1_ + self.agg_node1(e3) #[8, 64, 32, 32]
        
        d2 = self.up_conv2(d1_out)
        d2_ = self.deconv2(d2)
        d2_out = d2_ + self.agg_node2(e2) #[8, 64, 64, 64]

        d3 = self.up_conv3(d2_out)
        d3_ = self.deconv3(d3)
        d3_out = d3_ + self.agg_node3(e1) #[8, 64, 128, 128]
        d3_out = self.drop2(d3_out)

        d4 = self.up_conv3(d3_out)
        d4_ = self.deconv3(d4)
        d4_out = d4_ + self.agg_node3(x) #[8, 64, 256, 256]
        d4_out = self.drop2(d4_out)

        seg_out = self.out_conv2(self.relu(self.out_conv1(d4_out)))

        # up path to uncertainty

        d1_un = self.up_conv1_un(e4)
        d1__un = self.deconv1_un(d1_un)
        d1_out_un = d1__un + self.agg_node1_un(e3)
        
        d2_un = self.up_conv2_un(d1_out_un)
        d2__un = self.deconv2_un(d2_un)
        d2_out_un = d2__un + self.agg_node2_un(e2)

        d3_un = self.up_conv3_un(d2_out_un)
        d3__un = self.deconv3_un(d3_un)
        d3_out_un = d3__un + self.agg_node3_un(e1)

        d4_un = self.up_conv4_un(d3_out_un)
        d4__un = self.deconv4_un(d4_un)
        d4_out_un = d4__un + self.agg_node4_un(x)

        sigma = self.out_conv2_un(self.relu(self.out_conv1_un(d4_out_un)))

        out_10 = torch.zeros([10, sigma.size()[0], 1, 256, 256]).cuda()
        for i in range(10):
            epsilon = torch.normal(torch.zeros_like(sigma), 1)
            out_10[i,:,:,:,:] = seg_out + sigma*epsilon
        seg_out = torch.mean(out_10, axis = 0, keepdim = False)

        return seg_out, sigma

        
class LTN(nn.Module):
    def __init__(self):
        super(LTN, self).__init__()

        down_feature = [128, 256, 512, 1024]
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

        self.sigmoid = nn.Sigmoid()


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

        out = self.sigmoid(logits)
        #print(out.shape)

        return out

