import argparse


def parser_pca():
    parser = argparse.ArgumentParser(prog='model:PCA', description='Train PCA.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    return parser


def parser_kpca():
    parser = argparse.ArgumentParser(prog='model:kPCA', description='Train kernel PCA. The bandwidth parameter should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--sig2', type=float, help='Bandwidth parameter for RBF kernel function.')
    return parser


def parser_tune_kpca():
    parser = argparse.ArgumentParser(prog='model:kPCA', description='Tune kernel PCA. The bandwidth parameter should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--start_sig2', type=float, help='Starting point of sig2.')
    parser.add_argument('--stop_sig2', type=float, help='End point of sig2.')
    parser.add_argument('--num', type=int, help='The number of grid points.')
    return parser


def parser_fwsprs_kpca():
    parser = argparse.ArgumentParser(prog='model:fwSprskPCA', description='Train featurewise sparse kernel PCA. The bandwidth and sparsity parameters should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--sig2', type=float, help='Bandwidth parameter for RBF kernel function.')
    parser.add_argument('--rho', type=float, help='Cardinality parameter for sparse PCA.')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum iterations for sparse PCA.')
    parser.add_argument('--err_tol', type=float, default=1e-5, help='Convergence condition for sparse PCA.')
    parser.add_argument('--n_warmup', type=int, default=3, help='Iterations in warm-up for sparse PCA.')
    return parser


def parser_fwsprs_dpca():
    parser = argparse.ArgumentParser(prog='model:fwSprsdPCA', description='Train featurewise sparse PCA. The sparsity parameters should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--rho', type=float, help='Cardinality parameter for sparse PCA.')
    parser.add_argument('--max_iter', type=int, default=500, help='Maximum iterations for sparse PCA.')
    parser.add_argument('--err_tol', type=float, default=1e-5, help='Convergence condition for sparse PCA.')
    parser.add_argument('--n_warmup', type=int, default=3, help='Iterations in warm-up for sparse PCA.')
    return parser


def parser_iwsprs_kpca():
    parser = argparse.ArgumentParser(prog='model:iwSprskPCA', description='Train instancewise sparse kPCA. The bandwidth and sparsity parameters should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--c_sprs', type=float, help='Sparsity penalty parameters.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for Cayley Adam algorithm')
    parser.add_argument('--max_iter', type=int, help='Maximum iteration for Cayley Adam algorithm.')
    parser.add_argument('--err_tol', type=float, default=1e-3, help='Convergence condition for Cayley Adam algorithm.')
    parser.add_argument('--sig2', type=float, help='Bandwidth parameter for RBF kernel.')
    parser.add_argument('--norm', type=int, choices=[0, 1], help='Norm for penalty: (0) L11 norm; (1) L12 norm.')
    parser.add_argument('--approx_l1_method', type=int, choices=[0, 1], help='Approximation method for L1-norm: (0) sqrt(x**2); (1) log(cosh(x)).')
    parser.add_argument('--seed', type=int, help='Seed', metavar='')
    return parser


def parser_iwsprs_dpca():
    parser = argparse.ArgumentParser(prog='model:iwSprsdPCA', description='Train instancewise sparse PCA. The sparsity parameters should be provided.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', type=int, help='Latent dimension.')
    parser.add_argument('--c_sprs', type=float, help='Sparsity penalty parameters.')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate for Cayley Adam algorithm')
    parser.add_argument('--max_iter', type=int, help='Maximum iteration for Cayley Adam algorithm.')
    parser.add_argument('--err_tol', type=float, default=1e-3, help='Convergence condition for Cayley Adam algorithm.')
    parser.add_argument('--norm', type=int, choices=[0, 1], help='Norm for penalty: (0) L11 norm; (1) L12 norm.')
    parser.add_argument('--approx_l1_method', type=int, choices=[0, 1], help='Approximation method for L1-norm: (0) sqrt(x**2); (1) log(cosh(x)).')
    parser.add_argument('--seed', type=int, help='Seed', metavar='')
    return parser


def parser_sdrkm():
    parser = argparse.ArgumentParser(prog='model:SprsDRKM', description='Train sparse deep RKM.')
    parser.add_argument('path', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--h_dim', nargs=2, type=int, help='Latent dimension: [h1_dim, h2_dim].')
    parser.add_argument('--c_sprs', nargs=2, type=float, help='Sparsity penalty parameters: [c1, c2].')
    parser.add_argument('--max_iter', type=int, help='Maximum iteration for Adam-Cayley algorithm')
    parser.add_argument('--err_tol', type=float, default=1e-3, help='Convergence condition for Cayley Adam algorithm.')
    parser.add_argument('--sig', nargs=2, type=float, help='Bandwidth parameters for RBF kernel: [h1_sig, h2_sig].')
    parser.add_argument('--need_grad', nargs=2, type=bool, help='Need gradients for bandwidth parameters per level: [h1_bool, h2_bool].')
    parser.add_argument('--gamma', type=float, default=1, help='Trade-off between layers.')
    parser.add_argument('--decay', type=float, default=10, help='Decaying rate for sparsity.')
    parser.add_argument('--init_sig_sparse', type=float, default=3, help='Bandwidth parameters for sparsity.')
    parser.add_argument('--final_sig_sparse', type=float, default=1e-2, help='Bandwidth parameters for sparsity.')
    parser.add_argument('--norm', type=int, choices=[0, 1, 2, 3], help='Norm for penalty: (0) L11 norm; (1) L12 norm; (2) L01 norm ; (3) L02 norm.')
    parser.add_argument('--approx_l1_method', type=int, choices=[0, 1], help='Approximation method for L1-norm: (0) sqrt(x**2); (1) log(cosh(x)).')
    parser.add_argument('--lr_h', type=float, default=0.1, help='Learning rate for latent representations.')
    parser.add_argument('--lr_sig', type=float, default=0.2, help='Learning rate for bandwidth parameters.')
    parser.add_argument('--step_size_lr', type=int, default=1000, help='Reducing learning rate after certain epoch.')
    parser.add_argument('--is_reduce_lr', type=bool, default=False, help='Whether to reduce learning rate.')
    parser.add_argument('--seed', type=int, help='Seed', metavar='')
    return parser


def parser_gen_rkm(model='GenRKM'):
    parser = argparse.ArgumentParser(prog='model:{model}'.format(model=model), description='Train {model} model.'.format(model=model))
    parser.add_argument('path', type=str, help='The path to the dataset.', metavar='')
    parser.add_argument('--dataset', type=str, help='Name for the dataset.')
    parser.add_argument('--c_accu', type=float, default=100, help='The parameter for accuracy of reconstruction error.', metavar='')
    parser.add_argument('--c_stab', type=float, default=1, help='The parameter for stability of RKM energy function.', metavar='')
    parser.add_argument('--p_feat', type=int, default=256, help='The dimension for feature space.', metavar='')
    parser.add_argument('--n_width', type=int, default=96, help='The dimension for width.', metavar='')
    parser.add_argument('--n_height', type=int, default=96, help='The dimension for height.', metavar='')
    parser.add_argument('--n_channels', type=int, default=1, help='The dimension for channels.', metavar='')
    parser.add_argument('--h_dim', type=int, default=5, help='The dimension for the latent space.', metavar='')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate.', metavar='')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.', metavar='')
    parser.add_argument('--max_epoch', type=int, default=1000, help='The maximum number of epochs.', metavar='')
    parser.add_argument('--l_cost', type=float, default=100., help='The cost start to save checkpoint.', metavar='')
    parser.add_argument('--t_cost', type=float, default=1., help='The cost to terminate the training process.', metavar='')
    parser.add_argument('--custom_loss', type=str, default='MSE', choices=['MSE', 'BCE'], help='The type of loss function.', metavar='')
    parser.add_argument('--reduction_type', type=str, default='mean', choices=['mean', 'sum'], help='The type of reduction for loss.', metavar='')
    parser.add_argument('--seed', type=int, help='Seed', metavar='')
    return parser


def parser_ct_gen_rkm(model):
    parser = parser_gen_rkm(model)
    parser.add_argument('--const', type=float, default=1, help='Level of MAD for pruning.', metavar='')
    return parser


def parser_st_gen_rkm(model):
    parser = parser_gen_rkm(model)
    parser.add_argument('--const', type=int, default=100, help='Cardinality for pruning.', metavar='')
    return parser


def parser_spca_gen_rkm(model):
    parser = parser_gen_rkm(model)
    parser.add_argument('--th', type=float, default=1, help='Level of thresholding for sPCA.', metavar='')
    parser.add_argument('--inner_iter', type=int, default=5, help='Level of inner iterations for sPCA.', metavar='')
    return parser