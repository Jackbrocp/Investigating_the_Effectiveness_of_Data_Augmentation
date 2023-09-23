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
# import sys 
# sys.path.append('../')
from  data_augmentation import *

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


def do_transform(data_augmentation=None, args=None):

    if data_augmentation == 'blur':
        return Blur(p=float(args[0]))
    elif data_augmentation == 'patchgaussian':
        return AddPatchGaussian(int(args[0]), float(args[1]))
    elif data_augmentation == 'crop':
        return Cutout(1, int(args[0]), float(args[1]))
    elif data_augmentation == 'cutout':
        return Cutout(1, int(args[0]), float(args[1]))  # n_holes, length
    elif data_augmentation == 'equalize':
        return Equalize(p=float(args[0]))
    elif data_augmentation == 'flipLR':
        return transforms.RandomHorizontalFlip(float(args[0]))
    elif data_augmentation == 'flipUD':
        return transforms.RandomVerticalFlip(float(args[0]))
    elif data_augmentation == 'rotate':
        return Rotate(deg=int(args[0]), p=float(args[1]), mode=args[2])
    elif data_augmentation == 'shearX':
        return ShearX(p=float(args[0]), off=float(args[1]), mode=args[2])
    elif data_augmentation == 'has':
        return HaS()
    elif data_augmentation == 'randomerasing':
        return RandomErasing(probability=float(args[0]), sh=0.4, r1=0.3)
    elif data_augmentation == 'gridmask':
        return Grid(d1=24, d2=33, rotate=1, ratio=0.4, mode=1, prob=float(args[0]))
    elif data_augmentation == 'brightness':
        return Brightness(mag=float(args[0]),p=float(args[1]))
    elif data_augmentation == 'color':
        return Color(mag=float(args[0]),p=float(args[1]))
    elif data_augmentation == 'contrast':
        return Contrast(mag=float(args[0]),p=float(args[1]))
    elif data_augmentation == 'invert':
        return Invert()
    elif data_augmentation == 'posterize':
        return Posterize(v=int(args[0]),p=float(args[1]))
    elif data_augmentation == 'sharpness':
        return Sharpness(mag=float(args[0]),p=float(args[1]))
    elif data_augmentation == 'solarize':
        return Solarize(v=int(args[0]),p=float(args[1]))

def make_transform(data_augmentation=None, dataset_name='CIFAR10',  to3channels=False):
    # print(aug_list)
    if data_augmentation == None:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])])
    aug_list = data_augmentation.split('+')
    transform_list_before = []
    transform_list_after = []
    if dataset_name in ['MNIST', 'USPS'] and to3channels:
        transform_list_before.append(transforms.Grayscale(3))
    for item in aug_list:
        method = item[:-1].split('(')[0]
        parameters_list = item[:-1].split('(')[1].split(',')
        if method in ['blur', 'equalize', 'flipLR', 'flipUD', 'shearX', 'rotate','color','brightness','contrast','posterize','sharpness','solarize']:
            transform_list_before.append(do_transform(method, parameters_list))
        elif method in [ 'cutout', 'randomerasing', 'gridmask', 'has', 'crop']:
            transform_list_after.append(do_transform(method, parameters_list))
        elif method in ['patchgaussian']:
            transform_list = [transforms.ToTensor(), do_transform(method, parameters_list) ,transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])]
            # return transforms.Compose(transform_list)
            return transforms.Compose(transform_list_before + transform_list)
        elif method in ['basic']:
            transform_list_before.append(transforms.RandomCrop(32, padding=4))
            transform_list_before.append(transforms.RandomHorizontalFlip())
        elif method in ['autoaugment']:
            policy_type='cifar_code'
            transform_list_before = [transforms.RandomCrop(DATASET_SIZES[dataset_name][0], padding=4), transforms.RandomHorizontalFlip()]
            transform_list_after = [augmentation.Cutout(1, 8), transforms.ToTensor(), 
                                    transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])]
            transform_mid = policy.Policy(policy_type, transform_list_before, transform_list_after)
            return transform_mid
        elif method in ['randaugment']:
            return  transforms.Compose([RandAugment(int(parameters_list[0]),int(parameters_list[1])),
                                        transforms.ToTensor(),
                                        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])])



    transform_list = transform_list_before + [transforms.ToTensor(), transforms.Normalize(*DATASET_NORMALIZATION[dataset_name])] \
        + transform_list_after
    return transforms.Compose(transform_list)


if __name__ == '__main__':
    # make_transform('flipLR(0.25)+crop(4,0.25)+cutout(16,0.25)')
    # t=make_transform('flipLR(0.25)+crop(4,0.25)+cutout(16,0.75)')
    # t = make_transform('flipLR(0.5)+cutout(16,1.0)+equalize(0.5)')
    t = make_transform('basic()+brightness(0.05,0.05)')
    print(t)
