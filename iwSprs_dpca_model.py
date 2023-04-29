import numpy as np
import logging
import time
import torch
import sys
import random

from parsers import parser_iwsprs_dpca
from utils import save_score
from validators import checkargs_iwsprs_dpca
from cayley_adam import stiefel_optimizer
from torch import nn


# logger
file_handler = logging.FileHandler(filename='model_iwSprsdPCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_iwsprs_dpca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N, p = data.shape
li_loss = []
checkargs_iwsprs_dpca(args, data, N)


# fix seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# compute Gram matrix
def dual_corr_matrix(X):
    """ Compute the dual correlation matrix """
    K = X @ X.T
    return K


def standardize(X):
    """ Standardize dataset """
    EPS = 1e-12

    # compute mean and variance
    mean = X.mean(axis=0).reshape(1, -1)
    std = X.std(axis=0).reshape(1, -1) + EPS

    # standardize data
    Xs = (X - mean)/std
    cache = {'mu': mean, 'sig': std}
    return Xs, cache


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Xs, _ = standardize(data)
K = dual_corr_matrix(Xs)
Kc = torch.Tensor(K).to(device)
c_stab = np.linalg.norm(K, axis=0).max()


def st_init(N, s):
    h_init, _ = torch.linalg.qr(torch.randn(size=(N, s)))
    return h_init.t()


def l1_norm_approx():
    if args.approx_l1_method == 0:
        EPS = 1e-8
        return lambda x: torch.sqrt(x ** 2 + EPS)
    elif args.approx_l1_method == 1:
        return lambda x: torch.log(torch.cosh(x))


def sparse_kpca_loss(h, l1_norm_approx):
    f1 = torch.sum(h.t() * torch.mm(Kc, h.t()))  # quadratic terms
    l1_approx = l1_norm_approx(h.t())
    if args.norm == 0:
        f2 = torch.sum(l1_approx)
    elif args.norm == 1:
        f2 = torch.linalg.norm(torch.sum(l1_approx, dim=1))

    loss = -f1/2 + args.c_sprs * f2
    return loss


# initialization
h_init = st_init(N, args.h_dim)
h = nn.Parameter(h_init.to(device), requires_grad=True)
P = l1_norm_approx()
dict_m = {'params': [h], 'lr': args.lr, 'stiefel': True}
optimizer = stiefel_optimizer.AdamG([dict_m])


# train model
epoch = 0
prev_loss = np.inf
bool_converged = False
while epoch < args.max_iter and not bool_converged:
    loss = sparse_kpca_loss(h, P)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cur_loss = loss.item()
    li_loss.append(cur_loss)
    logging.info('Epoch: {}\t loss: {:.4f}'.format(epoch, cur_loss))
    epoch += 1

    bool_converged = np.abs(cur_loss - prev_loss) < args.err_tol
    prev_loss = cur_loss


cache = {'score': h.detach().cpu().numpy().T, 'losses': li_loss, 'args': args.__str__()}
save_score(ct, N, args, 'iwSprs_dpca', cache)


# garbage recycle
import gc
torch.cuda.empty_cache()
gc.collect()


logging.info('Computation finished.')


