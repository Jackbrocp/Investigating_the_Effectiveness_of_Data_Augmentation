import os
import sys
import re
import datetime

import numpy
import pickle
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from data_augmentation import *

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

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307, ), (0.3081, )),
    'USPS': ((0.1307, ), (0.3081, )),
    'FashionMNIST': ((0.1307, ), (0.3081, )),
    'QMNIST': ((0.1307, ), (0.3081, )),
    'EMNIST': ((0.1307, ), (0.3081, )),
    'KMNIST': ((0.1307, ), (0.3081, )),
    'ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}


def make_transform(data_augmentation=None, dataset_name='MNIST', ratio=0):
    print('============================RATIO{}=============================:'.
          format(ratio))
    if data_augmentation == None:
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])
        ])
    else:  #basic
        transform = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(DATASET_SIZES[dataset_name][0], padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(ratio),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])
        ])
    if data_augmentation == 'cutout':
        transform.transforms.append(Cutout(1, 8))
    elif data_augmentation == 'has':
        transform.transforms.append(HaS())
    elif data_augmentation == 'randomerasing':
        transform.transforms.append(
            RandomErasing(
                probability=0.5,
                sh=0.4,
                r1=0.3,
            ))
    elif data_augmentation == 'gridmask':
        transform.transforms.append(
            Grid(d1=24, d2=33, rotate=1, ratio=0.4, mode=1, prob=1.))
    return transform