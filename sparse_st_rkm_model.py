import numpy as np
import random
import time
import torch
import logging
import sys

import net_dsprites
import net_norb
import net_cars3D

from parsers import parser_gen_rkm
from utils import save_score
from validators import checkargs_gen_rkm
from datetime import datetime
from utils import *
from torch import optim, autograd
from torch.utils.data import DataLoader, TensorDataset


# logger
file_handler = logging.FileHandler(filename='model_StRKM_{ct}.log'.format(ct=time.strftime('%Y%m%d-%H%M')))
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=handlers)


# parser
parser = parser_gen_rkm()
args = parser.parse_args()
logging.info(args.__str__())


# load data
ct = time.strftime('%Y%m%d-%H%M')
data = np.load(args.path, allow_pickle=True)
N, p = data.shape
li_loss = []
checkargs_gen_rkm(args, data, args.batch_size)
logging.info('*'*30 + '\nModel: Penalized St-RKM' + '\nParameters: c_sprs = {c_sprs} and c_accu = {c_accu}\n'.format(c_sprs=args.c_sprs, c_accu=args.c_accu) + '*'*30)


# fix seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# directory for saving
folder = './{dataset}/N-{N}/score/{model}'.format(dataset=args.dataset, N=N, model='StRKM')
dirs = CreateDirs(ct=ct, folder=folder, model='StRKM')
dirs.create()


# preprocessing
def preprocessing(X: np.ndarray):
    X = torch.Tensor(X).type(torch.float32)
    return X.view(data_img.size(0), args.n_channels, args.n_width, args.n_height)
data_img = preprocessing(data)
p_input = args.n_channels * args.n_width * args.n_height
X = DataLoader(TensorDataset(data_img), batch_size=args.batch_size, shuffle=True)


# device: gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f'Using {device} device')


# encoder-decoder networks
if args.dataset == 'norb':
    encoder = net_norb.FeatMap(p_f=args.p_feat).to(device)
    decoder = net_norb.PreImgMap(p_f=args.p_feat).to(device)
elif args.dataset == 'dsprites':
    encoder = net_dsprites.FeatMap(p_f=args.p_feat).to(device)
    decoder = net_dsprites.PreImgMap(p_f=args.p_feat).to(device)
elif args.dataset == 'cars3D':
    encoder = net_cars3D.FeatMap(p_f=args.p_feat).to(device)
    decoder = net_cars3D.PreImgMap(p_f=args.p_feat).to(device)
else:
    logging.error('unsupported dataset!')


# Adam optimizer
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=args.lr, weight_decay=0)


# Centering + kPCA
def kPCA(X):
    K = torch.mm(X, torch.t(X))
    N, _ = K.shape
    oneN = torch.div(torch.ones(N, N), N).float().to(device)
    K = K - torch.mm(oneN, K) - torch.mm(K, oneN) + torch.mm(torch.mm(oneN, K), oneN)

    # do svd
    h, s, _ = torch.svd(K, some=False)

    # select components
    h = h[:, :args.h_dim]
    s = s[:args.h_dim]
    return h, s


# RKM Loss
def rkm_loss(phi, X):
    # get preimage
    h, s = kPCA(phi)
    W = torch.mm(phi.t(), h)
    Phi_hat = torch.mm(h, W.t())
    x_tilde = decoder(Phi_hat)

    # calculate loss
    recon_loss = set_loss_func(args.custom_loss, args.reduction_type)
    f1 = - torch.trace(torch.mm(torch.mm(phi, W), h.t()))  # RKM unit
    f2 = 0.5 * torch.trace(torch.mm(h, torch.mm(torch.diag(s), h.t())))  # h regularization
    f3 = 0.5 * torch.trace(torch.mm(W.t(), W))  # W regularization
    f4 = recon_loss(x_tilde.view(-1, p_input), X.view(-1, p_input))  # reconstruction

    loss = f1 + f2 + f3 + 0.5 * args.c_stab * (f1 + f2 + f3) ** 2 + args.c_accu * f4
    return loss


# Initialize
l_cost = np.inf
cost = np.inf
start = datetime.now()
loss_holder = []
epoch = 0


# train
while epoch < args.max_epoch and cost > args.t_cost:
    avg_loss = 0.0
    for i, x in enumerate(X, 0):
        with autograd.set_detect_anomaly(True):
            if i < np.floor(N / args.batch_size):
                try:
                    x = x[0].to(device)
                    phi = encoder(x)
                    loss = rkm_loss(phi, x)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                except Exception as e:
                    logging.error(e, exc_info=True)
                    loss = torch.inf * torch.ones(1, device=device)

                avg_loss += loss.detach().cpu().numpy()
            else:
                break

    # epoch logging
    logging.info('epoch: {epoch}'.format(epoch=epoch) +
                 '\t avg. loss: {:10.4f}'.format(float(avg_loss)))
    loss_holder.append(avg_loss)
    cost = avg_loss
    epoch += 1


    # Remember lowest cost and save checkpoint
    is_best = cost < l_cost
    l_cost = min(cost, l_cost)
    dirs.save_checkpoint({
        'epochs': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'l_cost': l_cost,
        'optimizer': optimizer.state_dict(),
    }, is_best)

logging.info('RKM finished Training. Lowest cost: {:10.4f}'.format(l_cost))


# load from existing savepoints
if os.path.exists(folder+'/cp/{}'.format(dirs.dircp)):
    sd_mdl = torch.load(folder+'/cp/{}'.format(dirs.dircp))
    encoder.load_state_dict(sd_mdl['encoder_state_dict'])
    decoder.load_state_dict(sd_mdl['decoder_state_dict'])


# final computation
encoder = encoder.cpu()
decoder = decoder.cpu()
device = 'cpu'
phi = encoder(torch.Tensor(data_img)).detach()
h, s = kPCA(phi)
W = torch.mm(phi.t(), h)

# save all model parameters
torch.save({'args': args,
            'encoder': encoder,
            'decoder': decoder,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'h': h,
            's': s,
            'W': W,
            'phi': phi,
            'loss': loss_holder},
           folder+'/out/{}'.format(dirs.dirout))


# garbage recycle
import gc
del encoder
del decoder
torch.cuda.empty_cache()
gc.collect()


logging.info('Final computation finished.')



