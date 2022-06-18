# Random Neural Fields
- [Deep Learning in Random Neural Fields: Numerical Experiments via Neural Tangent Kernel](https://arxiv.org/abs/2202.05254)
- Implementation of PyTorch

## Library
- PyTorch 1.10.2
- torchvision 0.11.3
- others can be installed with the following:
```
pip install -r requirements.txt
```

## Config
- ```src/config/config.yaml```
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
  NU: 0.5               # for matern kernel

GENERAL:
  EPOCH: 1000           
  GPUS: [1]
  NOTEBOOK: true        # if you use script, switch false
```

## Citation
```
@article{watanabe2022deep,
  title={Deep Learning in Random Neural Fields: Numerical Experiments via Neural Tangent Kernel},
  author={Watanabe, Kaito and Sakamoto, Kotaro and Karakida, Ryo and Sonoda, Sho and Amari, Shun-ichi},
  journal={arXiv preprint arXiv:2202.05254},
  year={2022}
}
```