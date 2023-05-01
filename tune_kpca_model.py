import numpy as np
import pandas as pd
import logging
import time
import sys
from tqdm import tqdm

from parsers import parser_tune_kpca
from utils import save_score
from validators import checkargs_tune_kpca
from sklearn.metrics.pairwise import euclidean_distances
from scipy.sparse.linalg import svds


# logger
file_handler = logging.FileHandler(filename='model_kPCA_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_tune_kpca()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N = data.shape[0]
p = data.reshape(N, -1).shape[1]
checkargs_tune_kpca(args, data, N)

li_trace = []
grids = np.linspace(start=args.start_sig2, stop=args.stop_sig2, num=args.num)

df = pd.DataFrame()
df['grids'] = grids

for sig2 in tqdm(grids):
    def rbf_kernel_matrix(X):
        """ Compute the kernel matrix with RBF kernel function """
        dist = euclidean_distances(X)
        K = np.exp(-np.square(dist) / (2 * sig2))
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
        U, s, _ = svds(Kc, k=args.h_dim)
        return U, s


    score, svdvals = kpca(data.reshape(N, -1), N)
    cache = {'score': score, 'svdvals': svdvals, 'trace': np.sum(svdvals), 'args': args.__str__()}
    logging.info('sig2: {}, trace: {}'.format(sig2, cache['trace']))

    li_trace.append(cache['trace'])

df['trace'] = np.array(li_trace)
df.to_csv('{dataset}-{ct}-{model}-{h_dim}.csv'.format(dataset=args.dataset, ct=ct, model='kPCA', h_dim=args.h_dim), index=False)
logging.info('Computation finished.')

