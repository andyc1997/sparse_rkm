import numpy as np
import logging
import time
import sys

from parsers import parser_pca
from utils import save_score
from validators import checkargs_pca
from scipy.sparse.linalg import svds


# logger
file_handler = logging.FileHandler(filename='model_PCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_pca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N = data.shape[0]
p = data.reshape(N, -1).shape[1]
checkargs_pca(args, data, N)


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


def pca(X):
    """ X: N-by-p 2D-dataset """
    Xs, cache = standardize(X)
    K = dual_corr_matrix(Xs)
    U, s, _ = svds(K, k=args.h_dim)
    return U, s, cache


score, svdvals, stat = pca(data.reshape(N, -1))
cache = {'score': score, 'svdvals': svdvals, 'trace': np.sum(svdvals), 'stat': stat, 'args': args.__str__()}
save_score(ct, N, args, 'pca', cache)
logging.info('Computation finished.')

