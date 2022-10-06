import sys
from os.path import join, dirname
from tqdm.auto import tqdm

import numpy as np
import scipy as sp

sys.path.append(join(dirname(__file__), "../.."))
from src.utils.factory import calc_acc , calc_loss_NTK


class LearningCurve:
    def __init__(self, cfg, lr, NTK_train, train_label, f_train_0, f_test_0):
        
        self.time_range = np.arange(0, cfg.GENERAL.EPOCH, 1)
        self.n_train = int(cfg.DATA.DATA_NUM * (1 - cfg.DATA.SPLIT_RATIO))
        self.n_class = cfg.DATA.CLASS
        
        self.NTK_train = NTK_train
        self.train_label = train_label
        self.f_train_0 = f_train_0
        self.f_test_0 = f_test_0
        self.id_mat = np.eye(self.n_train)
        self.lr = lr
        self.diff, self.P, self.D = self.prepare()
        
    def prepare(self):
        
        diff = self.f_train_0 - self.train_label
        
        mat = self.id_mat - self.lr * self.NTK_train / (self.n_train * self.n_class)
        diag, P = np.linalg.eigh(mat)
        D = np.diag(diag)
        
        return diff, P, D
        
    def basic_calc(self, epoch, split, label, NTK_prod=None):

        if epoch == 0:
            p_mat = self.id_mat
        else:
            p_mat = self.P @ (self.D**epoch) @ self.P.T
        
        if split == 'train':
            output = np.dot(p_mat, self.diff) + label
        elif split == 'test':
            output = self.f_test_0 - np.dot(
                NTK_prod.T, np.dot(self.id_mat - p_mat, self.diff)
            )
        
        loss = calc_loss_NTK(output, label)
        acc = calc_acc(output, label)
            
        return loss, acc, output
        
    def train_curve(self):
        
        train_results = {'train_losses': [], 'train_accs': [], 'train_outputs': []}

        for i in tqdm(self.time_range):
            loss, acc, output = self.basic_calc(i, 'train', self.train_label)
            train_results['train_losses'].append(loss)
            train_results['train_accs'].append(acc)
            train_results['train_outputs'].append(output)
            
        return train_results
    
    def test_curve(self, NTK_test, test_label):
        
        test_results = {'test_losses': [], 'test_accs': [], 'test_outputs': []}
            
        NTK_prod = sp.linalg.solve(self.NTK_train, NTK_test.T)
        for i in tqdm(self.time_range):
            loss, acc, output = self.basic_calc(i, 'test', test_label, NTK_prod)
            test_results['test_losses'].append(loss)
            test_results['test_accs'].append(acc)
            test_results['test_outputs'].append(output)
            
        return test_results