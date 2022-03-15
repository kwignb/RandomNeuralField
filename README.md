# Random Neural Field
- [Deep Learning in Random Neural Fields: Numerical Experiments via Neural Tangent Kernel](https://arxiv.org/abs/2202.05254)
- Implementation of PyTorch

## Config
```
DATA:
  NAME: mnist           # Now: mnist or fashion
  CLASS: 10             # number of class of mnist
  DATA_NUM: 100
  SPLIT_RATIO: 0.2

MODEL:
  INPUT_FEATURES: 784   # mnist: 28x28=784
  MID_FEATURES: 1024
  B_SIGMA: 0.1

INITIALIZER:
  TYPE: gaussian        # vanilla, gaussian, withmp, mexican, matern
  R_SIGMA: 0.1          # for receptive field
  S_SIGMA: 0.1          # for correlation
  M_SIGMA: 0.1          # for mexican hat
  M_THETA: 1.0          # for matern kernel
  M_NU: 1.0             # for matern kernel

GENERAL:
  EPOCH: 1000           # power of 10
  GPUS: [1]
  NOTEBOOK: true        # if you use script, switch false
```