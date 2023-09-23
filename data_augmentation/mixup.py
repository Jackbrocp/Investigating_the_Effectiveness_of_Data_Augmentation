import numpy as np
import torch
from torch.autograd import Variable
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
def mixup_operation(x,y):
    alpha = 1.
    inputs, targets_a, targets_b, lam, perm_index  = mixup_data(x, y, alpha ,use_cuda=False)
    inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
    if lam > 0.5:
        y = targets_a
    else:
        y = targets_b
    return inputs, y