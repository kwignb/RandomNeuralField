import numpy as np

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


class MakeDataset:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.dataset_name = cfg.DATA.NAME
        self.n_class = cfg.DATA.CLASS
        self.data_num = cfg.DATA.DATA_NUM
        self.test_ratio = cfg.DATA.SPLIT_RATIO
        
        self.dataloader = self.loader_setup()
        
    def loader_setup(self):
        
        transform = transforms.Compose(
            [transforms.ToTensor()]
            )
        
        if self.dataset_name == 'mnist':
            dataset = datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
                )
        elif self.dataset_name == 'fashion':
            dataset = datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform
                )
        else:
            NotImplementedError
            
        dataloader = DataLoader(dataset)
            
        return dataloader
    
    def get_array(self):
        
        X, y = [], []
        for i, (img, label) in enumerate(self.dataloader):
            label_list = [-0.1 for i in range(self.n_class)]
            img = img.numpy()
            label_list[label] = 0.9
            X.append(img / np.linalg.norm(img))
            y.append(label_list)
            if i == self.data_num - 1:
                break
            
        X, y = np.array(X).squeeze(axis=1), np.array(y, dtype='float32')
        
        train_id, test_id = train_test_split(
            np.arange(self.data_num),
            test_size=self.test_ratio,
            random_state=47
        )
        
        X_train, X_test = X[train_id], X[test_id]
        y_train, y_test = y[train_id], y[test_id]
        
        return X_train, X_test, y_train, y_test
    
    def get_tensor(self):
        
        X_train, X_test, y_train, y_test = self.get_array()
        
        X_train, X_test = torch.tensor(X_train), torch.tensor(X_test)
        y_train, y_test = torch.tensor(y_train), torch.tensor(y_test)
        
        return X_train, X_test, y_train, y_test
    
    def get_loader(self):
        
        X_train, X_test, y_train, y_test = self.get_tensor()
        
        train_tensor = TensorDataset(X_train, y_train)
        test_tensor = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_tensor, batch_size=self.data_num)
        test_loader = DataLoader(test_tensor, batch_size=self.data_num)
        
        return train_loader, test_loader