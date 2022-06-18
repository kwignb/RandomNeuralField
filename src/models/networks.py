import sys
from os.path import join, dirname

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(join(dirname(__file__), "../.."))
from src.models.initializers import Initializers


def read_model(cfg):
    
    init_type = cfg.INITIALIZER.TYPE
    device_id = cfg.GENERAL.GPUS
    
    # define the type of network
    if init_type == 'vanilla':
        net = VanillaNet(cfg)
    else:
        net = Networks(cfg)
        
    if torch.cuda.is_available():
        net.to(f'cuda:{device_id[0]}')
        
    return net


class LinearNTK(nn.Linear):
    def __init__(self, in_features, out_features, b_sig, w_sig=2, bias=True):
        super(LinearNTK, self).__init__(in_features, out_features)
        self.reset_parameters()
        self.b_sig = b_sig
        self.w_sig = w_sig
        
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0, std=1)
        if self.bias is not None:
            nn.init.normal_(self.bias, mean=0, std=1)
            
    def forward(self, input):
        return F.linear(input,
                        self.w_sig * self.weight / np.sqrt(self.in_features),
                        self.b_sig * self.bias)
        
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, b_sig={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.b_sig
        )
        
        
class VanillaNet(nn.Module):
    def __init__(self, cfg):
        super(VanillaNet, self).__init__()
        
        self.visualize = cfg.MODEL.VISUALIZE
        
        in_features = cfg.MODEL.INPUT_FEATURES
        mid_features = cfg.MODEL.MID_FEATURES
        out_features = cfg.DATA.CLASS
        b_sig = cfg.MODEL.B_SIGMA
        
        self.l1 = LinearNTK(in_features, mid_features, b_sig)
        self.l2 = LinearNTK(mid_features, mid_features, b_sig)
        self.l3 = LinearNTK(mid_features, out_features, b_sig)
        
    def forward(self, x):
        
        if self.visualize:
            return self.l1(x)
        else:        
            h1 = F.relu(self.l1(x))
            h2 = F.relu(self.l2(h1))
            h3 = self.l3(h2)
            return h3
    
    
class Networks(nn.Module):
    def __init__(self, cfg):
        super(Networks, self).__init__()
        
        self.init_type = cfg.INITIALIZER.TYPE
        self.mid_features = cfg.MODEL.MID_FEATURES
        self.visualize = cfg.MODEL.VISUALIZE
        
        in_features = cfg.MODEL.INPUT_FEATURES
        out_features = cfg.DATA.CLASS
        b_sig = cfg.MODEL.B_SIGMA
        
        init_weight = Initializers(cfg).get_initializer(in_features, self.mid_features)
        
        self.l1 = LinearNTK(in_features, self.mid_features, b_sig)
        self.l1.weight.data = init_weight
        
        if self.init_type == 'withmp':
            self.l2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        else:
            self.l2 = LinearNTK(self.mid_features, self.mid_features, b_sig)
        
        self.l3 = LinearNTK(self.mid_features, out_features, b_sig)
    
    def forward(self, x):
        
        if self.visualize:
            return self.l1(x)
        else:
            h1 = F.relu(self.l1(x))
            if self.init_type == 'withmp':
                h2 = self.l2(h1.view(1, -1, self.mid_features))
                h3 = self.l3(h2).squeeze()
            else:
                h2 = F.relu(self.l2(h1))
                h3 = self.l3(h2)
            return h3