from math import pow

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.gaussian_process.kernels import Matern


def sym_mat(features):
    
    vec = np.arange(features).reshape(-1, 1)
    X = np.tile(vec**2, (1, features))
    H = np.dot(vec, vec.T)
    
    return X - 2*H + X.T


def receptive_mat(in_features, out_features, sigma):
    
    vec_in = np.arange(1, in_features+1).reshape(-1, 1)
    vec_out = np.arange(1, out_features+1).reshape(-1, 1)
    
    X = np.tile((vec_out / out_features)**2, (1, in_features))
    Y = np.tile((vec_in / in_features)**2, (1, out_features)).T
    
    H = []
    if in_features < out_features:
        for i in range(in_features):
            H.append(vec_out * (i+1))
    elif in_features > out_features:
        for i in range(out_features):
            H.append(vec_in * (i+1))
    
    H = np.array(H).reshape(
        in_features, out_features
        ).T / (in_features * out_features)
    
    return (X - 2*H + Y) / sigma**2


def weight_correlation(in_features, out_features, s_sigma):
    
    weight = np.random.normal(0, 1, (out_features, in_features))
    
    in_scaled_sigma = (s_sigma * in_features)**2
    c = pow(2.0 / (np.pi * in_scaled_sigma), 1/4)
    
    A_in = c * np.exp( - sym_mat(in_features) / in_scaled_sigma)
    
    return weight @ A_in


def matern_kernel(in_features, out_features, theta, nu):
    
    Kernel = Matern(length_scale=in_features*theta,
                    length_scale_bounds=(1e-5, 1e5),
                    nu=nu)
    
    _range = np.linspace(0, in_features-1, num=in_features)
    grids = _range.reshape(-1, 1)
    kernel_mat = Kernel(grids, grids)
    
    gen = multivariate_normal(cov=kernel_mat)
    mk_init = gen.rvs(size=out_features, random_state=47)
    
    return mk_init
