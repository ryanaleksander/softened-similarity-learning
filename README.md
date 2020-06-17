# ARTNet Pytorch
## Introduction
Unofficial Pytorch implementation of Unsupervised Person Re-identification via Softened Similarity Learning

Paper: https://arxiv.org/abs/2004.03547

## Requirements
The code is written using the following environment. There isn't a strict version requirement, but deviate from the listed versions at your own risk

* python: 3.7.3
* pytorch: 1.2.0
* torchvision: 0.4.0
* matplotlib: 3.1.0
* numpy: 1.16.4
* tqdm: 4.32.1

## Training
### Preparing Data
1. Download and extract files for the memory table: https://drive.google.com/file/d/1Oc824mzfJ1LYRWc1R3sLWgcf_T6efpJc/view?usp=sharing
2. This implementation is built for the Market1501 dataset only. So just download it and extract to your desired location.

### Configuration
1. Make a copy of `config.ini`
2. Edit the configurations as you see fit
### Run
`python train.py --config [config file path]`  

## Testing
`python test.py --config [config file path]`  


## Extra
A pretrained model with Top-1 accuracy 0.66: https://drive.google.com/file/d/1V5JrR3Vqvj7HRQCcT99ZqBVrqUYnpRtv/view?usp=sharing

