import sys
from os.path import join, dirname

import numpy as np

import torch.nn as nn
from torch import Tensor

sys.path.append(join(dirname(__file__), "../.."))
from src.models.utils import sym_mat, receptive_mat, weight_correlation, matern_kernel


class Initializers(nn.Module):
    def __init__(self, cfg):
        
        self.type = cfg.INITIALIZER.TYPE
        self.r_sigma = cfg.INITIALIZER.R_SIGMA
        self.s_sigma = cfg.INITIALIZER.S_SIGMA
        self.m_sigma = cfg.INITIALIZER.M_SIGMA
        self.theta = cfg.INITIALIZER.M_THETA
        self.nu = cfg.INITIALIZER.M_NU
        
    def get_initializer(self, in_features, out_features):
        
        if self.type == 'gaussian' or self.type == 'withmp':
            init_weight = self.get_gaussian_type(in_features, out_features)
        elif self.type == 'mexican':
            init_weight = self.get_mexican_type(in_features, out_features)
        elif self.type == 'matern':
            init_weight = self.get_matern_type(in_features, out_features)
        else:
            NotImplementedError
            
        return init_weight
            
    def get_gaussian_type(self, in_features, out_features):
        
        if in_features != out_features:
            R = np.sqrt(
                np.exp(
                    - receptive_mat(in_features,
                                    out_features,
                                    self.r_sigma)
                    )
                )
        elif in_features == out_features:
            R = np.sqrt(
                np.exp(
                    - sym_mat(
                        in_features
                        ) / (2*(in_features*self.r_sigma)**2)
                    )
                )
        weight_correlated = weight_correlation(in_features,
                                               out_features,
                                               self.s_sigma)
        init_weight = R * weight_correlated
        
        return Tensor(init_weight)
    
    def get_mexican_type(self, in_features, out_features):
        
        coef = 2 / (np.sqrt(3*self.m_sigma) * pow(np.pi, 1/4))
        
        if in_features != out_features:
            mh = receptive_mat(in_features, out_features, self.m_sigma)
        elif in_features == out_features:
            mh = sym_mat(in_features) / self.m_sigma**2
            
        M = coef * (np.ones((out_features, in_features)) - mh) * np.exp( - mh / 2)
        
        weight_correlated = weight_correlation(in_features, 
                                               out_features,
                                               self.s_sigma)
        init_weight = M * weight_correlated
        
        return Tensor(init_weight)
    
    def get_matern_type(self, in_features, out_features):
        
        if in_features != out_features:
            R = np.sqrt(np.exp(-receptive_mat(
                in_features, out_features, self.r_sigma
                )))
        elif in_features == out_features:
            R = np.sqrt(np.exp(-sym_mat(
                in_features
                ) / (2 * (in_features * self.r_sigma)**2)))
            
        init_mk = matern_kernel(in_features, 
                                out_features,
                                self.theta,
                                self.nu)
        init_weight = R * init_mk
        
        return Tensor(init_weight)