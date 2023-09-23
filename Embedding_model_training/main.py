'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from transform import make_transform
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


seed = 0
setup_seed(seed)


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


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument("--src", type=str, default=None, help="")
parser.add_argument('--aug_arg', nargs='*', action=ParseKwargs, default=None)
parser.add_argument("--dataset", type=str, default='CIFAR100', help="")
parser.add_argument("--model", type=str, default='ResNet50', help="")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(*DATASET_NORMALIZATION[args.dataset]),
])
if args.dataset == 'CIFAR100':
    root = './data'
    trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch, shuffle=False, num_workers=4)
    num_classes = 100
elif args.dataset == 'CIFAR10':
    root = './data'
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch, shuffle=False, num_workers=4)
    num_classes = 10
# Model
print('==> Building model..')

# net = ResNet18(num_classes)
net = ResNet50(num_classes)
# net = DenseNet121(num_classes)
 
net = net.to(device)
net = torch.nn.DataParallel(net, device_ids=[0,1])
cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/home/yangsuorong/Optimal_transport/otdd-main/CIFAR100/embedder/embedder_CIFAR100_resnet50_basic.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        save_path = './embedder/'.format(args.dataset)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        torch.save(state, save_path + '{}.pth'.format(args.model))
        best_acc = acc
    print('Epoch:{}  Acc:{}'.format(epoch, best_acc))


for epoch in tqdm(range(start_epoch, start_epoch+args.epoch)):
    train(epoch)
    test(epoch)
    scheduler.step()
 
