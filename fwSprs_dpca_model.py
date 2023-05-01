import logging
import time
import sys

from parsers import parser_fwsprs_dpca
from utils import save_score
from validators import checkargs_fwsprs_dpca
from spca import *


# logger
file_handler = logging.FileHandler(filename='model_fwSprsdPCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# disable numba logger, too many messages
logging.getLogger('numba').setLevel(logging.WARNING)


# parser
parser = parser_fwsprs_dpca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N = data.shape[0]
p = data.reshape(N, -1).shape[1]
checkargs_fwsprs_dpca(args, data, N)


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


def spca(X, N):
    """ X: N-by-p 2D-dataset """
    Xs, cache = standardize(X)
    K = dual_corr_matrix(Xs)
    U, _, _ = tpower(K, N, args.h_dim, args.rho, args.n_warmup, args.max_iter, args.err_tol)
    return U


score = spca(data.reshape(N, -1), N)
cache = {'score': score, 'args': args.__str__()}
save_score(ct, N, args, 'fwSprs_dpca', cache)
logging.info('Computation finished.')

