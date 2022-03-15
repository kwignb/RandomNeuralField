import os

from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt


def read_yaml(fpath='config/config.yaml'):
    config = OmegaConf.load(fpath)
    return config


def calc_acc(outputs, labels):
    
    correct = 0
    for i in range(len(outputs)):
        if outputs[i].argmax() == labels[i].argmax():
            correct += 1
            
    acc = correct / len(labels)
    
    return acc


def calc_loss_NTK(outputs, labels):
    
    length = labels.shape[0] * labels.shape[1]
    loss = (np.linalg.norm(outputs - labels)**2) / (2 * length)
    
    return loss