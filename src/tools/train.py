import sys
from os.path import join, dirname

import torch
import torch.nn as nn
from torch import optim

sys.path.append(join(dirname(__file__), "../.."))
from src.utils.factory import calc_acc


def train(cfg, net, lr, database):
    
    n_epochs = cfg.GENERAL.EPOCH
    input_shape = cfg.MODEL.INPUT_FEATURES
    device_id = cfg.GENERAL.GPUS
    
    # define the loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr)
    
    train_loader, test_loader = database.get_loader()
    
    results = {
        'train_losses': [], 'test_losses': [],
        'train_accs': [], 'test_accs': [],
        'train_outputs': [], 'test_outputs': []
        }
    
    for epoch in range(n_epochs):

        train_loss, test_loss = 0, 0
        
        net.train()
        for imgs, labels in train_loader:
            imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
            labels = labels.float().to(f'cuda:{device_id[0]}')
            
            optimizer.zero_grad()
            
            outputs = net(imgs)
            train_loss = criterion(outputs, labels) / 2
            outputs = outputs.cpu().detach().numpy()
            train_acc = calc_acc(outputs, labels)
            
            train_loss.backward()
            optimizer.step()
            
            if epoch == 0:
                f_train_0 = outputs
                
        results['train_losses'].append(train_loss.cpu().detach().numpy())
        results['train_accs'].append(train_acc)
        results['train_outputs'].append(outputs)
        
        net.eval()
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
                labels = labels.float().to(f'cuda:{device_id[0]}')
                
                outputs = net(imgs)
                test_loss = criterion(outputs, labels) / 2
                outputs = outputs.cpu().detach().numpy()
                test_acc = calc_acc(outputs, labels)

                if epoch == 0:
                    f_test_0 = outputs
                    
            results['test_losses'].append(test_loss.cpu().detach().numpy())
            results['test_accs'].append(test_acc)
            results['test_outputs'].append(outputs)
            
        print('Epoch[{}/{}], TrainLoss: {:.4f}, TestLoss: {:.4f}, TestAcc: {:.4f}'
              .format(epoch+1, n_epochs, train_loss, test_loss, test_acc))
            
    return f_train_0, f_test_0, results