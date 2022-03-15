from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def mean_output_dynamics(results, ntk_results):
    
    # train and test outputs of normal training 
    train_outputs = np.array(results['train_outputs'])
    test_outputs = np.array(results['test_outputs'])
    
    # train and test outputs of ntk learning curve
    ntk_train_outputs = np.array(ntk_results['train_outputs'])
    ntk_test_outputs = np.array(ntk_results['test_outputs'])
    
    mean_tr_out = train_outputs.mean(axis=1)
    mean_te_out = test_outputs.mean(axis=1)
    
    mean_ntk_tr_out = ntk_train_outputs.mean(axis=1)
    mean_ntk_te_out = ntk_test_outputs.mean(axis=1)
    
    return mean_tr_out, mean_te_out, mean_ntk_tr_out, mean_ntk_te_out


def output_dynamics_per_class(outputs, labels, class_num):
    
    outputs = np.array(outputs)
    n_class = outputs.shape[-1]
    
    m_lists = [[] for _ in range(n_class)]
    for op in outputs:
        c = []
        for i in range(len(op)):
            if labels[i][class_num] == max(labels[i]):
                c.append(op[i])
                
        c_lists = [[] for _ in range(n_class)]
        for i in c:    
            for j, cl in zip(range(n_class), c_lists):
                cl.append(i[j])
                
        for m, cl in zip(m_lists, c_lists):
            m.append(np.mean(cl))
            
    return m_lists


def visualize(cfg, results, ntk_results, train_labels, test_labels, class_num):
    
    n_epoch = cfg.GENERAL.EPOCH
    init_name = cfg.INITIALIZER.TYPE
    notebook = cfg.GENERAL.NOTEBOOK
    
    tr_out = output_dynamics_per_class(
        results['train_outputs'], train_labels, class_num
        )
    te_out = output_dynamics_per_class(
        results['test_outputs'], test_labels, class_num
        )
    ntk_tr_out = output_dynamics_per_class(
        ntk_results['train_outputs'], train_labels, class_num
        )
    ntk_te_out = output_dynamics_per_class(
        ntk_results['test_outputs'], test_labels, class_num
        )
    
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    x_ticks = np.power(10, np.arange(0, int(np.log10(n_epoch))+1, 1).tolist())
    
    for i in range(4):
        
        if i == 0:
            ax[i].plot(results['train_losses'], color='r', label='Train')
            ax[i].plot(results['test_losses'], color='b', label='Test')
            ax[i].plot(ntk_results['train_losses'], color='r', 
                       linestyle='dashed', label='NTK Train')
            ax[i].plot(ntk_results['test_losses'], color='b', 
                       linestyle='dashed', label='NTK Test')
            ax[i].set_title('Loss', fontsize=15)
        
        elif i == 1:
            ax[i].plot(results['train_accs'], color='r')
            ax[i].plot(results['test_accs'], color='b')
            ax[i].plot(ntk_results['train_accs'], color='r', linestyle='dashed')
            ax[i].plot(ntk_results['test_accs'], color='b', linestyle='dashed')
            ax[i].set_title('Accuracy', fontsize=15)
            
        elif i == 2:
            for j, (nn, nt) in enumerate(zip(tr_out, ntk_tr_out)):
                if j == 0:
                    ax[i].plot(nn, label='Model')
                    ax[i].plot(nt, linestyle='dashed', label='NTK model')
                else:
                    ax[i].plot(nn)
                    ax[i].plot(nt, linestyle='dashed')
            
            ax[i].set_title('Train output', fontsize=15)
            
        elif i == 3:
            for j, (nn, nt) in enumerate(zip(te_out, ntk_te_out)):
                ax[i].plot(nn)
                ax[i].plot(nt, linestyle='dashed')
                
            ax[i].set_title('Test output', fontsize=15)
            
        ax[i].set_xlim(xmin=10**0, xmax=n_epoch)
        ax[i].set_xlabel('Epoch', fontsize=15)
        ax[i].tick_params(labelsize=15)
        ax[i].set_xscale('log')
        ax[i].set_xticks(x_ticks)
        
    plt.tight_layout()
    
    if notebook:
        path = str(Path().resolve().parent)
    else:
        path = str(Path().resolve())
    
    fig.savefig(path + f'/output/{init_name}_regime.png')