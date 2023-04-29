import os
import pickle
import torch


def save_score(ct, N, args, model, cache):
    folder = './{dataset}/N-{N}/score/{model}'.format(dataset=args.dataset, N=N, model=model)
    if not os.path.exists(folder):
        os.makedirs(folder)

    if model == 'fwSprs_dpca' or model == 'fwSprs_kpca':
        filename = folder + '/{card}/{dataset}-{ct}-{model}-{h_dim}'.format(card=str(int(args.rho*1000)), dataset=args.dataset, ct=ct, model=model, h_dim=args.h_dim)
    elif model == 'iwSprs_kpca' or model == 'iwSprs_dpca':
        filename = folder + '/{card}/{dataset}-{ct}-{model}-{h_dim}'.format(card=str(int(args.c_sprs*1000)), dataset=args.dataset, ct=ct, model=model, h_dim=args.h_dim)
    elif model == 'sdrkm':
        norm_direc = {0: '11norm', 1: '12norm', 2: '01norm', 3: '02norm'}
        filename = folder + '/{norm}/{dataset}-{ct}-{model}-s1-{h_dim1}-s2-{h_dim2}'.format(norm=norm_direc[args.norm], dataset=args.dataset, ct=ct, model=model, h_dim1=args.h_dim[0], h_dim2=args.h_dim[1])
    else:
        filename = folder + '/{dataset}-{ct}-{model}-{h_dim}'.format(dataset=args.dataset, ct=ct, model=model, h_dim=args.h_dim)

    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(cache, f)


class CreateDirs:
    """
    Creates directories for Checkpoints and saving trained models
    Source: https://github.com/MrPandey01/Stiefel_Restricted_Kernel_Machine/blob/main/code/utils.py
    """
    def __init__(self, ct, folder, model):
        self.ct = ct  # checkpoint time
        self.folder = folder
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = 'Mul_trained_{}_{}.tar'.format(model, self.ct)

    def create(self):
        # folder for checkpoints
        if not os.path.exists(self.folder+'/cp/'):
            os.makedirs(self.folder+'/cp/')
        # folder for model
        if not os.path.exists(self.folder+'/out/'):
            os.makedirs(self.folder+'/out/')

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, self.folder+'/cp/{}'.format(self.dircp))


def set_loss_func(loss_type: str = 'MSE', reduction_type: str = 'mean'):
    """
    Return a pytorch loss function.

    :param str loss_type: Type of the loss function for reconstruction, either MSE or BCE
    :param str reduction_type: Type of reduction, either mean or sum
    """
    if loss_type == 'MSE':
        return torch.nn.MSELoss(reduction=reduction_type)
    elif loss_type == 'BCE':
        return torch.nn.BCELoss(reduction=reduction_type)
    raise Exception(f'Invalid input for loss_type or reduction_type: {loss_type} | {reduction_type}')