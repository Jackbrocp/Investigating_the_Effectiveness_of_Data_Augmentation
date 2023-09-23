import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5"
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
from models import *
import time
DATASET_SIZES = {
    'MNIST': (28, 28),
    'FashionMNIST': (28, 28),
    'EMNIST': (28, 28),
    'QMNIST': (28, 28),
    'KMNIST': (28, 28),
    'USPS': (16, 16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
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
    'MNIST': 50000,
    'CIFAR10': 40000
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
parser.add_argument('--model', type=str, default='ResNet50')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
transform_ori = make_transform(data_augmentation=None,dataset_name=args.dataset)
transform_aug = make_transform(data_augmentation=args.trg, dataset_name=args.dataset )
print(transform_ori)
print(transform_aug)
maxsize = 50000
datadir = './data/'
loaders_src  = load_torchvision_data(args.dataset, transform=transform_ori, valid_size=0, maxsize=maxsize,datadir=datadir )[0]
loaders_trg  = load_torchvision_data(args.dataset, transform=transform_aug, valid_size=0, maxsize=maxsize,datadir=datadir )[0]
#############################################

num_classes = 10
if args.model == 'ResNet50':
    net = ResNet50()
    model_path = './embedder/ResNet50.pth'
elif args.model == 'ResNet18':
    net = ResNet18()
    model_path = './embedder/ResNet18.pth'
embedder = net.eval()
for p in embedder.parameters():
    p.requires_grad = False
embedder =  torch.nn.DataParallel(embedder, device_ids=[0,1]).cuda()


state_dict = torch.load(model_path)['net']
embedder.load_state_dict(state_dict)
embedder.linear = torch.nn.Identity()
del state_dict
#############################################
feature_cost = FeatureCost(src_embedding = embedder,
                           src_dim = (3, DATASET_SIZES[args.dataset][0],DATASET_SIZES[args.dataset][1]),
                           tgt_embedding = embedder,
                           tgt_dim = (3, DATASET_SIZES[args.dataset][0],DATASET_SIZES[args.dataset][1]),
                           p = 2,
                           device = device)
dist = DatasetDistance(loaders_src['train'], loaders_trg['train'],
                          class_num = DATASET_NCLASSES[args.dataset],
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          model = args.model,
                          aug_method = args.trg,
                          dataset = 'CIFAR10',
                          feature_cost = feature_cost,
                          p = 2, entreg = 1e-1,
                          device = device,
                        )
                  
d = dist.distance(maxsamples=DATASET_NUMBER[args.dataset] )
print('================================',d,'================================')