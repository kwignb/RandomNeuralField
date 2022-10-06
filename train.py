import sys
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.utils.factory import read_yaml
from src.models.networks import read_model
from src.utils.factory import calc_acc


def create_loader(phase):
    
    bs = 4096
    
    transform = transforms.Compose(
            [transforms.ToTensor()]
            )
    
    dataset = datasets.MNIST(
        root='data', 
        train=True if phase == 'train' else False,
        download=True, transform=transform
        )
    
    dataloader = DataLoader(dataset)
    
    X, y = [], []
    for img, label in dataloader:
        label_list = [-0.1 for _ in range(10)]
        img = img.numpy()
        label_list[label] = 0.9
        X.append(img / np.linalg.norm(img))
        y.append(label_list)
        
    X, y = np.array(X).squeeze(axis=1), np.array(y, dtype='float32')
    
    if phase == 'train':
        train_id, val_id = train_test_split(
            np.arange(50000),
            test_size=0.2,
            random_state=47
            )
        
        X_train, X_val = X[train_id], X[val_id]
        y_train, y_val = y[train_id], y[val_id]
        
        X_train, X_val = torch.tensor(X_train), torch.tensor(X_val)
        y_train, y_val = torch.tensor(y_train), torch.tensor(y_val)
        
        train_tensor = TensorDataset(X_train, y_train)
        val_tensor = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(train_tensor, batch_size=bs)
        val_loader = DataLoader(val_tensor, batch_size=bs)
        
        return train_loader, val_loader
    
    elif phase == 'test':
        X_test, y_test = torch.tensor(X), torch.tensor(y)
        test_tensor = TensorDataset(X_test, y_test)
        return DataLoader(test_tensor, batch_size=64)
    
    else:
        NotImplementedError


def train_one_epoch(cfg, net, train_loader, optimizer, criterion):
    
    input_shape = cfg.MODEL.INPUT_FEATURES
    device_id = cfg.GENERAL.GPUS
    
    running_loss, running_acc = 0., 0.
    for i, (imgs, labels) in enumerate(train_loader):
        
        imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
        labels = labels.float().to(f'cuda:{device_id[0]}')
        
        optimizer.zero_grad()
        
        outputs = net(imgs)
        loss = criterion(outputs, labels) / 2
        acc = calc_acc(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += acc
    
    return running_loss / (i+1), running_acc / (i+1)


def train(cfg, net, lr, train_loader, val_loader):

    n_epochs = cfg.GENERAL.EPOCH
    input_shape = cfg.MODEL.INPUT_FEATURES
    device_id = cfg.GENERAL.GPUS
    init_name = cfg.INITIALIZER.TYPE
    
    # define the loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr)
    
    best_val_loss = 1e10
    
    keys = ['train/loss', 'train/acc', 'val/loss', 'val/acc']    
    for epoch in range(n_epochs):

        net.train()
        avg_train_loss, avg_train_acc = train_one_epoch(
            cfg, net, train_loader, optimizer, criterion
            )
        
        net.eval()
        with torch.no_grad():
            running_vloss, running_vacc = 0.0, 0.0
            for i, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.view(-1, input_shape).to(f'cuda:{device_id[0]}')
                labels = labels.float().to(f'cuda:{device_id[0]}')
                
                outputs = net(imgs)
                val_loss = criterion(outputs, labels) / 2
                val_acc = calc_acc(outputs, labels)
                
                running_vloss += val_loss.item()
                running_vacc += val_acc
                
            avg_val_loss = running_vloss / (i+1)
            avg_val_acc = running_vacc / (i+1)
            
        vals = [avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]
        
        file_name = Path('output') / f'{init_name}_result.csv'
        x = {k: v for k, v in zip(keys, vals)}
        n_cols = len(x) + 1
        header = '' if file_name.exists() else (('%20s,' * n_cols % tuple(['epoch'] + keys)).rstrip(',') + '\n')
        with open(file_name, 'a') as f:
            f.write(header + ('%20.5g,' * n_cols % tuple([epoch] + vals)).rstrip(',') + '\n')
            
        if (epoch + 1) % 1000 == 0:
            print(
                'Epoch[{}/{}], TrainLoss: {:.5f}, ValLoss: {:.5f}, ValAcc: {:.5f}'
                .format(epoch+1, n_epochs, vals[0], vals[2], vals[3])
                )
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(net.state_dict(), f'pretrained/{init_name}_best.pth')


def main():
    
    cfg = read_yaml(fpath='src/config/config.yaml')
    cfg.GENERAL.EPOCH = 50000
            
    train_loader, val_loader = create_loader(phase='train')
    
    # init_types = ['vanilla', 'gaussian', 'withmp', 'mexican', 'matern']
    init_types = ['withmp']
    for it in init_types:
        if it == 'gaussian':
            cfg.INITIALIZER.R_SIGMA = 0.5
            cfg.INITIALIZER.S_SIGMA = 0.01
        elif it == 'withmp':
            cfg.INITIALIZER.R_SIGMA = 0.5
            cfg.INITIALIZER.S_SIGMA = 0.01
        elif it == 'mexican':
            cfg.INITIALIZER.M_SIGMA = 0.01
            cfg.INITIALIZER.S_SIGMA = 0.01
        elif it == 'matern':
            cfg.INITIALIZER.R_SIGMA = 0.5
            cfg.INITIALIZER.S_SIGMA = 0.01
        
        cfg.INITIALIZER.TYPE = it
        
        net = read_model(cfg)
        train(cfg, net, 0.5, train_loader, val_loader)
        
if __name__ == '__main__':
    main()