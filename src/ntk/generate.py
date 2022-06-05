from tqdm import tqdm

import numpy as np

import torch
from torch.autograd import grad


def generate_ntk(net, label, train, test, cfg, calc_lr=False):
    
    input_shape = cfg.MODEL.INPUT_FEATURES
    device_id = cfg.GENERAL.GPUS
    
    if len(train.size()) > 2:
        train = train.view(-1, input_shape) 
        test = test.view(-1, input_shape)
    
    if torch.cuda.is_available():
        train = train.to(f'cuda:{device_id[0]}')
        test = test.to(f'cuda:{device_id[0]}')
        
    f_train = net(train)
    train_grads = []
    for i in range(len(f_train)):
        train_grads.append(
            grad(f_train[i][label], net.parameters(), retain_graph=True)
        )
        
    K_train = torch.zeros((len(f_train), len(f_train)))
    for i in tqdm(range(len(f_train))):
        grad_i = train_grads[i]
        for j in range(i+1):
            grad_j = train_grads[j]
            K_train[i, j] = sum([torch.sum(
                torch.mul(grad_i[k], grad_j[k])
                ) for k in range(len(grad_j))])
    
    K_train = K_train.cpu().numpy()
    K_train = K_train + K_train.T - np.diag(K_train.diagonal())
    
    if calc_lr:
        NTK_train = np.kron(K_train, np.eye(cfg.DATA.CLASS))
        vals = np.linalg.eigvalsh(NTK_train)
        lr = 2 / (max(vals) + 1e-12)
        return NTK_train, lr
    else:
        f_test = net(test)
        K_test = torch.zeros((len(f_test), len(f_train)))
        test_grads = []
        for i in tqdm(range(len(f_test))):
            test_grads.append(
                grad(f_test[i][label], net.parameters(), retain_graph=True)
            )
            for j, train_grad in enumerate(train_grads):
                for k, test_grad in enumerate(test_grads):
                    K_test[k, j] = sum([torch.sum(
                        torch.mul(train_grad[u], test_grad[u])
                        ) for u in range(len(test_grad))])
                    
        K_test = K_test.cpu().numpy()
        return K_train, K_test