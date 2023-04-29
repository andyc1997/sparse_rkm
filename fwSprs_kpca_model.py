import logging
import time
import sys

from parsers import parser_fwsprs_kpca
from utils import save_score
from validators import checkargs_fwsprs_kpca
from sklearn.metrics.pairwise import euclidean_distances
from spca import *


# logger
file_handler = logging.FileHandler(filename='model_fwSprskPCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_fwsprs_kpca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N, p = data.shape
checkargs_fwsprs_kpca(args, data, N)


def rbf_kernel_matrix(X):
    """ Compute the kernel matrix with RBF kernel function """
    dist = euclidean_distances(X)
    K = np.exp(-np.square(dist) / (2 * args.sig2))
    return K


def centering(K, N):
    """ Center the kernel matrix """
    J = np.ones((N, N)) / N
    Kc = K - J @ K - K @ J + J @ (K @ J)
    return Kc


def kpca(X, N):
    """ X: N-by-p 2D-dataset """
    K = rbf_kernel_matrix(X)
    Kc = centering(K, N)
    U, _, _ = tpower(Kc, N, args.h_dim, args.rho, args.n_warmup, args.max_iter, args.err_tol)
    return U


score, svdvals = kpca(data.reshape(N, -1), N)
cache = {'score': score, 'args': args.__str__()}
save_score(ct, N, args, 'fwSprs_kpca', cache)
logging.info('Computation finished.')

