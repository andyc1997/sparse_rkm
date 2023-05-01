import numpy as np
import logging
import time
import torch
import sys
import random

from parsers import parser_iwsprs_kpca
from utils import save_score
from validators import checkargs_iwsprs_kpca
from cayley_adam import stiefel_optimizer
from torch import nn
from sklearn.metrics.pairwise import euclidean_distances


# logger
file_handler = logging.FileHandler(filename='model_iwSprskPCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_iwsprs_kpca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N = data.shape[0]
p = data.reshape(N, -1).shape[1]
li_loss = []
checkargs_iwsprs_kpca(args, data, N)


# fix seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# compute similarity matrix
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
similarity = -np.square(euclidean_distances(data.reshape(N, -1)))
similarity = torch.Tensor(similarity).to(device)


# center kernel matrix
K = torch.exp(similarity / (2 * args.sig2))
rowsum_K = torch.sum(K, dim=1).reshape((-1, 1))
totalsum_K = torch.sum(rowsum_K)
rowsum_K = torch.repeat_interleave(rowsum_K, repeats=N, dim=1)
Kc = K - rowsum_K.T / N - rowsum_K / N + totalsum_K / N ** 2


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
save_score(ct, N, args, 'iwSprs_kpca', cache)

# garbage recycle
import gc
torch.cuda.empty_cache()
gc.collect()


logging.info('Computation finished.')


