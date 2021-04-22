# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the following repo
# (released under the MIT License).
#
# Source:
# https://github.com/tom-beer/deep-scientific-discovery/blob/master/hsic.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_HSIC). The modifications
# to this file are subject to the License
# located at the root directory.
# ---------------------------------------------------------------


import torch
import numpy as np

from useful_utils import ns_profiling_label


def pairwise_distances(x):
  x_distances = torch.sum(x**2,-1).reshape((-1,1))
  return -2*torch.mm(x,x.t()) + x_distances + x_distances.t() 

def kernelMatrixGaussian(x, sigma=1):

    pairwise_distances_ = pairwise_distances(x)
    gamma = -1.0 / (sigma ** 2)
    return torch.exp(gamma * pairwise_distances_)

def kernelMatrixLinear(x):
  return torch.matmul(x,x.t())

# check
def HSIC(X, Y, kernelX="Gaussian", kernelY="Gaussian", sigmaX=1, sigmaY=1,
         log_median_pairwise_distance=False):
  m,_ = X.shape
  assert(m>1)

  median_pairwise_distanceX, median_pairwise_distanceY = np.nan, np.nan
  if log_median_pairwise_distance:
      # This calc takes a long time. It is used for debugging and disabled by default.
      with ns_profiling_label('dist'):
          median_pairwise_distanceX = median_pairwise_distance(X)
          median_pairwise_distanceY = median_pairwise_distance(Y)

  with ns_profiling_label('Hkernel'):
      K = kernelMatrixGaussian(X,sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X)
      L = kernelMatrixGaussian(Y,sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y)

  with ns_profiling_label('Hfinal'):
      H = torch.eye(m, device='cuda') - 1.0/m * torch.ones((m,m), device='cuda')
      H = H.float().cuda()

      Kc = torch.mm(H,torch.mm(K,H))

      HSIC = torch.trace(torch.mm(L,Kc))/((m-1)**2)
  return HSIC, median_pairwise_distanceX, median_pairwise_distanceY


def median_pairwise_distance(X):
    t = pairwise_distances(X).detach()
    triu_indices = t.triu(diagonal=1).nonzero().T

    if triu_indices[0].shape[0] == 0 or triu_indices[1].shape[0] == 0:
        return 0.
    else:
        return torch.median(t[triu_indices[0], triu_indices[1]]).item()

