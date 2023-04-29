import argparse
import logging
import numpy as np
import os
import pickle
import sys
import random
import torch
import time


from parsers import parser_sdrkm
from utils import save_score
from validators import checkargs_sdrkm
from cayley_adam import stiefel_optimizer
from torch import nn
from sklearn.metrics.pairwise import euclidean_distances
from torch.optim.lr_scheduler import StepLR


# logger
file_handler = logging.FileHandler(filename='model_sdrkm_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_sdrkm()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N, p = data.shape
li_loss = []
checkargs_sdrkm(args, data, N)


# fix seed
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# compute similarity matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity = -np.square(euclidean_distances(data.reshape(N, -1)))
similarity = torch.Tensor(similarity).to(device)
J = torch.ones(N).to(device)  # vector of all 1's, length N


# get parameters
s1, s2 = args.h_dim
c1, c2 = args.c_sparse
s = s1 + s2
init_sig_sparse = args.init_sig_sparse
final_sig_sparse = args.final_sig_sparse
beta = args.decay / args.max_iter


def st_init(N, s):
    h_init, _ = torch.linalg.qr(torch.randn(size=(N, s)))
    return h_init.t()


def l1_norm_approx():
    if args.approx_l1_method == 0:
        EPS = 1e-8
        return lambda x: torch.sqrt(x ** 2 + EPS)
    elif args.approx_l1_method == 1:
        return lambda x: torch.log(torch.cosh(x))


def sdrkm(h, h1_sig, h2_sig, sig_sparse, l1_norm_approx):
    # split levels
    h1 = h[:s1, :]
    h2 = h[s1:, :]

    # update similarity matrix
    G = torch.mm(h1.t(), h1)
    g = torch.diag(G)
    similarity_latent = - (torch.outer(g, J) + torch.outer(J, g) - 2*G)

    # kernel matrices
    Kx = torch.exp(similarity / (2*h1_sig ** 2))
    Kh = torch.exp(similarity_latent / (2*h2_sig ** 2))

    # penalized energy
    f1 = torch.sum(h1.t() * torch.mm(Kx, h1.t())) + \
         torch.sum(h2.t() * torch.mm(Kh, h2.t()))/args.gamma

    if c1 == 0 and c2 == 0:
        loss = -f1/2  # DRKM

    else:
        if args.norm == 0:
            f2 = torch.sum(l1_norm_approx(h1.t()))  # sum over all elements
            f3 = torch.sum(l1_norm_approx(h2.t()))
        elif args.norm == 1:
            l1_sum_h1 = torch.sum(l1_norm_approx(h1.t()), dim=1)  # sum over each obs
            l1_sum_h2 = torch.sum(l1_norm_approx(h2.t()), dim=1)
            f2 = torch.linalg.norm(l1_sum_h1)  # take norm over instances, all L1(h[i]) must be dense
            f3 = torch.linalg.norm(l1_sum_h2)
        elif args.norm == 2:
            f2 = torch.sum(h1.shape[0] - torch.sum(torch.exp(-h1.t() ** 2/(2 * sig_sparse ** 2)), dim=1))  # Approx L0(h[i]) and sum
            f3 = torch.sum(h2.shape[0] - torch.sum(torch.exp(-h2.t() ** 2/(2 * sig_sparse ** 2)), dim=1))
        elif args.norm == 3:
            f2 = torch.linalg.norm(h1.shape[0] - torch.sum(torch.exp(-h1.t() ** 2/(2 * sig_sparse ** 2)), dim=1))  # Approx L0(h[i]) and take norm
            f3 = torch.linalg.norm(h2.shape[0] - torch.sum(torch.exp(-h2.t() ** 2/(2 * sig_sparse ** 2)), dim=1))
        loss = -f1/2 + c1 * f2 + c2 * f3  # sparse DRKM
    return loss


# initialization
h_init = st_init(N, s)
h = nn.Parameter(h_init.to(device), requires_grad=True)
h1_sig = nn.Parameter(args.sig[0] * torch.ones(1).to(device), requires_grad=args.need_grad[0])  # level 1
h2_sig = nn.Parameter(args.sig[1] * torch.ones(1).to(device), requires_grad=args.need_grad[1])  # level 2
sig_sparse = args.init_sig_sparse  # initialize the bandwidth for SL0 approx.
P = l1_norm_approx() # penalty functional


# Cayley adam optimizer
dict_m = {'params': [h], 'lr': args.lr_h, 'stiefel': True}  # variables subjected to St. manifold
dict_nm = {'params': [h1_sig, h2_sig], 'lr': args.lr_sig, 'stiefel': False}  # variables without constraints
optimizer = stiefel_optimizer.AdamG([dict_m, dict_nm])
scheduler = StepLR(optimizer, step_size=args.step_size_lr, gamma=0.9)  # learning rate decay


# train model
epoch = 0
prev_loss = np.inf
bool_converged = False
while epoch < args.max_iter and not bool_converged:
    loss = sdrkm(h, h1_sig, h2_sig, sig_sparse, P)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if args.is_reduce_lr:
        scheduler.step()
    sig_sparse = np.maximum(args.final_sig_sparse, np.exp(-beta * epoch) * args.init_sig_sparse)  # decay SL0 bandwidth

    cur_loss = loss.item()
    li_loss.append(cur_loss)
    logging.info('Epoch: {}\t loss: {:.4f}\t Bandwidths (h1 h2): {:.4f}\t {:.4f}'.format(epoch, cur_loss, h1_sig.item(), h2_sig.item()))
    epoch += 1

    bool_converged = np.abs(cur_loss - prev_loss) < args.err_tol
    prev_loss = cur_loss


cache = {'score': h.detach().cpu().numpy().T, 'bandwidths': [h1_sig.detach().cpu().numpy(), h2_sig.detach().cpu().numpy()], 'losses': li_loss, 'args': args.__str__()}
save_score(ct, N, args, 'sdrkm', cache)

# garbage recycle
import gc
torch.cuda.empty_cache()
gc.collect()


logging.info('Computation finished.')