import sys
from os.path import join, dirname

import torch
import torch.nn as nn
from torch import optim

sys.path.append(join(dirname(__file__), "../.."))
from src.ntk.generate import generate_ntk
from src.utils.factory import calc_diff_frob, calc_acc


def calc_ntk_frob(cfg, net, lr, train_loader, test_loader):
    
    n_epochs = cfg.GENERAL.EPOCH
    input_shape = cfg.MODEL.INPUT_FEATURES
    device_id = cfg.GENERAL.GPUS
    
    # define the loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr)
    
    for epoch in range(n_epochs):

        train_loss = 0
        
        net.train()
        for imgs, labels in train_loader:
            imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
            labels = labels.float().to(f'cuda:{device_id[0]}')
            
            optimizer.zero_grad()
            
            if epoch == 0:
                ntk_0, _ = generate_ntk(net, 0, imgs, imgs, cfg, calc_lr=True)
            
            outputs = net(imgs)
            train_loss = criterion(outputs, labels) / 2
            train_acc = calc_acc(outputs, labels)
            
            train_loss.backward()
            optimizer.step()
            
            if epoch == n_epochs - 1:
                ntk_t, _ = generate_ntk(net, 0, imgs, imgs, cfg, calc_lr=True)
        
        net.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
                labels = labels.float().to(f'cuda:{device_id[0]}')
                
                outputs = net(imgs)
                test_loss = criterion(outputs, labels) / 2
                outputs = outputs.cpu().detach().numpy()
                test_acc = calc_acc(outputs, labels)
    
    ntk_diff_frob = calc_diff_frob(ntk_0, ntk_t)           
    
    return ntk_diff_frob, train_acc, test_acc