import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import pynvml
from otdd.pytorch.distance import DatasetDistance, FeatureCost
from otdd.pytorch.datasets import load_torchvision_data
from transform import make_transform
from tqdm import tqdm
import numpy as np
import argparse
import torch
from torchvision.models import resnet18
import pandas as pd
import csv
import json
import random
import sys
from models import *



DATASET_SIZES = {
    'MNIST': (28, 28),
    'FashionMNIST': (28, 28),
    'EMNIST': (28, 28),
    'QMNIST': (28, 28),
    'KMNIST': (28, 28),
    'USPS': (16, 16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
    'CIFAR100': (32, 32),
    'STL10': (96, 96),
    'tiny-ImageNet': (64, 64)
}
DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'tiny-ImageNet': 200
}

DATASET_NUMBER = {
    'MNIST': 60000,
    'CIFAR10': 50000,
    'CIFAR100': 50000
}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if key == 'p' or key == 'off' or key == 'max_scale':
                value = float(value)
            elif key == 'length' or key == 'patch_size' or key == 'deg':
                value = int(value)
            getattr(namespace, self.dest)[key] = value


parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default=None, help="")
parser.add_argument('--trg', type=str, default='basic()', help='')
parser.add_argument("--dataset", type=str, default='CIFAR10', help="")
parser.add_argument('--ratio', type=int, default=0)
parser.add_argument('--aug_arg', nargs='*',action=ParseKwargs,default=None)
args = parser.parse_args()



maxsize = 50000
transform_ori = make_transform(data_augmentation=None,dataset_name=args.dataset)
transform_aug = make_transform(data_augmentation=args.trg, dataset_name=args.dataset )
print(transform_aug)
datadir = './data/'
loaders_src  = load_torchvision_data(args.dataset, transform=transform_ori, valid_size=0,  maxsize=maxsize,datadir=datadir )[0]
loaders_trg  = load_torchvision_data(args.dataset, transform=transform_aug, valid_size=0,  maxsize=maxsize,datadir=datadir )[0]


##################  Load the Embedding Model ########################
embedder = ResNet50().eval()
for p in embedder.parameters():
    p.requires_grad = False
embedder =  torch.nn.DataParallel(embedder, device_ids=[0,1]).cuda()
model_path = './embedder/ResNet50.pth'
# model_path = './embedder/ResNet18.pth'

state_dict = torch.load(model_path)['net']
embedder.load_state_dict(state_dict)
embedder.linear = torch.nn.Identity()
del state_dict
#######################################################
feature_cost = None
dist = DatasetDistance(loaders_src['train'], loaders_trg['train'],
                          class_num = DATASET_NCLASSES[args.dataset],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = feature_cost,
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          keepaugment=False,
                          precision='single',
                          p = 2, entreg = 1e-1,
                          device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                        #   device = 'cpu')

v_10_90 = dist.variaty(maxsamples=DATASET_NUMBER[args.dataset],embedder = embedder,name = args.trg)
v_10_90 = torch.tensor(v_10_90)
v_10_90 = torch.mean(v_10_90, dim=0).numpy()
print(v_10_90)





     

