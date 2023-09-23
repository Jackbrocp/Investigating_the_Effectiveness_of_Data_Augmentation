import os
import pdb
from time import time
import itertools
import numpy as np
from tqdm.autonotebook import tqdm
import torch
from functools import partial
import inspect
import logging
import geomloss
import ot
from torch.distributions.multivariate_normal import MultivariateNormal
from matplotlib import cm
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.cluster import k_means, DBSCAN

def variaty_ot(D1i, D2i):
    variaty_geomloss = partial(
                inner_variaty,
                feature_cost = feature_cost)