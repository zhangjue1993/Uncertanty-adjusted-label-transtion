import torch 
import torch.nn as nn
import numpy as np
from utils import toLabel
eps = 1e-10

class loss_WCE(nn.Module):
    def __init__(self):
        super(loss_WCE, self).__init__()
    
    def forward(self, pred, targets):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # '''

        pred = nn.functional.softmax(pred,dim=1)
        #pred=torch.clamp(pred,1e-5,1-1e-5)
        print(torch.sum(targets))
        print(torch.mean(targets * torch.log(pred[:,0,:,:])),torch.mean((1-targets) * torch.log(pred[:,1,:,:])))
        loss = -torch.mean(0.1*targets * torch.log(pred[:,0,:,:])+0.9*(1.0-targets)*torch.log(pred[:,1,:,:]))
        return loss


class WCE(nn.Module):
    def __init__(self):
        super(WCE, self).__init__()
    
    def forward(self, pred, targets):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # '''
        # print(sigma.size())(1-sigma) (1-sigma)
        loss = -torch.mean(targets * torch.log(pred[:,1,:,:]+1e-5)+(1-targets)*torch.log(pred[:,0,:,:]+1e-5))
        return loss


class WCE_label_correction(nn.Module):
    def __init__(self):
        super(WCE_label_correction, self).__init__()
    
    def forward(self, pred, targets, T):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # # '''
        # if len(pred.size())<4:
        #     pred = torch.squeeze(pred,0)
        pred = torch.unsqueeze(pred,3)
        mask = torch.cat(((1-pred),pred),3)
        mask = mask.unsqueeze(4)
        
        T = T.permute(0, 2, 3, 1)
        T1,T2,T3,T4 = T[:,:,:,0].unsqueeze(3),(1-T[:,:,:,1]).unsqueeze(3),(1-T[:,:,:,0]).unsqueeze(3),T[:,:,:,1].unsqueeze(3)
        Transition = torch.cat((T1,T2,T3,T4),3)
        
        #print(mask.shape)
        _,h,w,_ = Transition.shape
        Transition = Transition.reshape(-1,h,w,2,2)
        #print(Transition.shape)
        pre_T = torch.matmul(Transition,mask)
        pre_T = pre_T.squeeze(4)
        #print('pre_t:',pre_T.shape)

        pre_T[pre_T<0] = 0
        pre_T[pre_T>1] = 1
        # print(pre_T.size())
        loss = -torch.mean(targets*torch.log(pre_T[:,:,:,1]+1e-5)+(1-targets)*torch.log(pre_T[:,:,:,0]+1e-5))
        
        return loss,pre_T[:,:,:,1]


class WCE_label_correction_noise(nn.Module):
    def __init__(self):
        super(WCE_label_correction_noise, self).__init__()
    
    def forward(self, pred, targets, T,noisy_mask):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # # '''
        # if len(pred.size())<4:
        #     pred = torch.squeeze(pred,0)
        pred = torch.unsqueeze(pred,3)
        mask = torch.cat(((1-pred),pred),3)
        mask = mask.unsqueeze(4)
        
        T = T.permute(0, 2, 3, 1)
        T1,T2,T3,T4 = T[:,:,:,0].unsqueeze(3),(1-T[:,:,:,1]).unsqueeze(3),(1-T[:,:,:,0]).unsqueeze(3),T[:,:,:,1].unsqueeze(3)
        Transition = torch.cat((T1,T2,T3,T4),3)
        
        #print(mask.shape)
        _,h,w,_ = Transition.shape
        Transition = Transition.reshape(-1,h,w,2,2)
        #print(Transition.shape)
        pre_T = torch.matmul(Transition,mask)
        pre_T = pre_T.squeeze(4)
        #print('pre_t:',pre_T.shape)

        pre_T[pre_T<0] = 0
        pre_T[pre_T>1] = 1
        # print(pre_T.size())
        temp = targets*torch.log(pre_T[:,:,:,1]+1e-5)+(1-targets)*torch.log(pre_T[:,:,:,0]+1e-5)
        loss = -torch.sum(temp*noisy_mask)/torch.sum(noisy_mask)
        return loss,pre_T[:,:,:,1]

class WCE_label_correction_un(nn.Module):
    def __init__(self):
        super(WCE_label_correction_un, self).__init__()
    
    def forward(self, pred, targets, T,noisy_mask,sigma):
        '''
        pred.size: [8, 2, 256, 256]
        labels.size: [8, 256, 256]
        # # '''
        # if len(pred.size())<4:
        #     pred = torch.squeeze(pred,0)
        #print(sigma.shape)
        pred = torch.unsqueeze(pred,3)
        mask = torch.cat(((1-pred),pred),3)
        mask = mask.unsqueeze(4)
        
        T = T.permute(0, 2, 3, 1)
        T1,T2,T3,T4 = T[:,:,:,0].unsqueeze(3),(1-T[:,:,:,1]).unsqueeze(3),(1-T[:,:,:,0]).unsqueeze(3),T[:,:,:,1].unsqueeze(3)
        Transition = torch.cat((T1,T2,T3,T4),3)
        
        #print(mask.shape)
        _,h,w,_ = Transition.shape
        Transition = Transition.reshape(-1,h,w,2,2)
        #print(Transition.shape)
        pre_T = torch.matmul(Transition,mask)
        pre_T = pre_T.squeeze(4)
        #print('pre_t:',pre_T.shape)

        pre_T[pre_T<0] = 0
        pre_T[pre_T>1] = 1
        # print(pre_T.size())
        temp = targets*torch.log(pre_T[:,:,:,1]+1e-5)+(1-targets)*torch.log(pre_T[:,:,:,0]+1e-5)
        loss = -torch.sum(temp*noisy_mask/torch.exp(sigma))/torch.sum(noisy_mask)
        return loss,pre_T[:,:,:,1]

# def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
#     """
#     Args:
#         sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
#         targets: true value, one-hot-like vector of size [N,C]
#         pos_weight: Weight for postive sample
#     """
#     if not (targets.size() == sigmoid_x.size()):
#         raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

#     loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

#     if weight is not None:
#         loss = loss * weight

#     if not reduce:
#         return loss
#     elif size_average:
#         return loss.mean()
#     else:
#         return loss.sum()


# class WeightedBCELoss(nn.Module):
#     def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
#         """
#         Args:
#             pos_weight = Weight for postive samples. Size [1,C]
#             weight = Weight for Each class. Size [1,C]
#             PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
#             WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
#         """
#         super().__init__()

#         self.register_buffer('weight', weight)
#         self.register_buffer('pos_weight', pos_weight)
#         self.size_average = size_average
#         self.reduce = reduce
#         self.PosWeightIsDynamic = PosWeightIsDynamic

#     def forward(self, input, target):
#         # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
#         if self.PosWeightIsDynamic:
#             positive_counts = target.sum(dim=0)
#             nBatch = len(target)
#             self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

#         if self.weight is not None:
#             # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
#             return weighted_binary_cross_entropy(input, target,
#                                                  self.pos_weight,
#                                                  weight=self.weight,
#                                                  size_average=self.size_average,
#                                                  reduce=self.reduce)
#         else:
#             return weighted_binary_cross_entropy(input, target,
#                                                  self.pos_weight,
#                                                  weight=None,
#                                                  size_average=self.size_average,
#                                                  reduce=self.reduce)