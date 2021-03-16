import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pdb


class MultiheadAttention(nn.Module):
    def __init__(self, feature_dim=512, n_head=8, key_feature_dim=64):
        super(MultiheadAttention, self).__init__()
        self.Nh = n_head
        self.head = nn.ModuleList()
        for N in range(self.Nh):
            self.head.append(RelationUnit(feature_dim, key_feature_dim))
        # self.out_conv = nn.Linear(n_head*key_feature_dim, feature_dim)  # bias=False

    def forward(self, query=None, key=None, value=None):
        isFirst = True
        for N in range(self.Nh):
            if(isFirst):
                concat = self.head[N](query, key, value)
                isFirst = False
            else:
                concat = torch.cat((concat, self.head[N](query, key, value)), -1)
        # output = self.out_conv(concat)
        output = concat
        return output
    

class RelationUnit(nn.Module):
    def __init__(self, feature_dim=512, key_feature_dim=64):
        super(RelationUnit, self).__init__()
        self.temp = 30  
        self.WK = nn.Linear(feature_dim, key_feature_dim)  # bias=False
        # self.WQ = nn.Linear(feature_dim, key_feature_dim)
        self.WV = nn.Linear(feature_dim, feature_dim)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        '''
        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        '''
        
        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            # m.weight.data.normal_(1. / m.in_features, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()
        

    def forward(self, query=None, key=None, value=None):
        w_k = self.WK(key)
        w_k = F.normalize(w_k, p=2, dim=-1)
        w_k = w_k.permute(1,2,0) # Batch, Dim, Len_1

        w_q = self.WK(query)
        w_q = F.normalize(w_q, p=2, dim=-1)
        w_q = w_q.permute(1,0,2) # Batch, Len_2, Dim
        
        dot_prod = torch.bmm(w_q, w_k) # Batch, Len_2, Len_1  
        affinity = F.softmax(dot_prod*self.temp, dim=-1) 
    
        w_v = value.permute(1,0,2) # Batch, Len_1, Dim
        output = torch.bmm(affinity, w_v) # Batch, Len_2, Dim
        output = output.permute(1,0,2)

        return output


