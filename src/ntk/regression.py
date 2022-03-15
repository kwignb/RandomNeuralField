import numpy as np


def calc_acc_for_reg(yhat, y):
    
    correct = 0
    for i in range(len(y)):
        label, pred = np.argmax(y[i]), np.argmax(yhat[i])
        if label == pred:
            correct += 1
        
    return correct / len(y)


def ntk_regression(cfg, NTK_train, NTK_test, y_train, y_test):
    
    n_class = cfg.DATA.CLASS
    
    f_test = np.dot(NTK_test @ np.linalg.inv(NTK_train), y_train)
    
    loss = np.linalg.norm(f_test - y_test)**2 / (2*len(y_test)*n_class)
    acc = calc_acc_for_reg(f_test, y_test)

    return loss, acc    