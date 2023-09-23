import numpy as np
import torch
import torch.cuda
from numba import jit
from torch.autograd import Function
from numba import cuda
import math
import torch.nn.functional as F
def covariance_pca_v2(D1i, D2i,DA_name=None):
    '''
    D1i, D2i : N * 512
    CIFAR10: 5000*512
    '''
 

    D2i = D2i[torch.randperm(D2i.size()[0])]
    Cov1 = cov(D1i.T)
    E1, V1 = torch.eig(Cov1, eigenvectors=True)
    # V1 = F.normalize(V1)
    V1 = V1.T
    E1 = torch.squeeze(E1[:,[0]], dim=1)
    E1, idx = torch.sort(E1, descending=True)
    V1 = V1[idx]

 
    Cov2 = cov(D2i.T)
    E2, V2 = torch.eig(Cov2,eigenvectors=True)
    V2 = V2.T
    E2 = torch.squeeze(E2[:,[0]], dim=1)
    E2, idx = torch.sort(E2, descending=True)
    V2 = V2[idx]

    cumulative_dis = []
    therhold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    therhold_step = 0
    i = 1
    while i < E1.shape[0]+1:
    # for i in range(1,E1.shape[0]+1):
        if sum(E1[:i])/sum(E1) >= therhold[therhold_step] or sum(E2[:i])/sum(E2)>=therhold[therhold_step]:
            therhold_step += 1
            E1i, V1i = E1[:i].reshape(i,1), V1[:i]
            E2i, V2i = E2[:i].reshape(i,1), V2[:i]
            V1i, V2i = V1i * E1i, V2i * E2i
             
            dis = F.pairwise_distance(V1i, V2i, p=2)
            cumulative_dis.append(round(dis.sum().item() ,2))
            if therhold_step == len(therhold):
                break
        else:
            i += 1
    return cumulative_dis




def covariance_pca(D1i, D2i,DA_name=None):
    '''
    D1i, D2i : N * 512
    '''
 

    D2i = D2i[torch.randperm(D2i.size()[0])]
    # data_dir = '/home/yangsuorong/Optimal_transport/otdd-main/otdd/pytorch/Variaty/Analysis_eigenvalue/'
    Cov1 = cov(D1i.T)
    E1, V1 = torch.eig(Cov1, eigenvectors=True)
    # V1 = F.normalize(V1)
    V1 = V1.T
    E1 = torch.squeeze(E1[:,[0]], dim=1)
    E1, idx = torch.sort(E1, descending=True)
    V1 = V1[idx]
 

 
    Cov2 = cov(D2i.T)
    E2, V2 = torch.eig(Cov2,eigenvectors=True)
    V2 = V2.T
    E2 = torch.squeeze(E2[:,[0]], dim=1)
    E2, idx = torch.sort(E2, descending=True)
    V2 = V2[idx]

    cumulative_dis = []
    therhold = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    therhold_step = 0
    i = 1
    while i <= E1.shape[0]+1:
        if sum(E1[:i])/sum(E1) >= therhold[therhold_step] or sum(E2[:i])/sum(E2)>=therhold[therhold_step]:
            therhold_step += 1
            E1i, V1i = E1[:i].reshape(i,1), V1[:i]
            E2i, V2i = E2[:i].reshape(i,1), V2[:i]
            V1i, V2i = V1i * E1i, V2i * E2i
            dis = F.pairwise_distance(V1i, V2i, p=2)
            cumulative_dis.append(round(dis.sum().item(),2))
        else:
            i += 1
        if therhold_step == len(therhold):
            break
    return cumulative_dis


 

def cov(m, mean=None, rowvar=True, inplace=False):
    """ Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Arguments:
        m (tensor): A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar (bool): If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t() # m : 512 * N
    # print(m.shape)
    fact = 1.0 / (m.size(1) - 1) #fact=1/(N-1)
     
    if mean is None:
        mean = torch.mean(m, dim=1, keepdim=True) # 512 * 1
    else:
        mean = mean.unsqueeze(1) # For broadcasting
    if inplace:
        m -= mean
    else:
        m = (m-mean)/torch.sqrt((fact * torch.sum((m-mean)**2, dim=0)))
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


if __name__ == '__main__':
    a = torch.load('./D1.pt')
    res = covariance_pca(a,a)
    print(res)
