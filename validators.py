def checkargs_pca(args, data, N):
    assert args.h_dim > 0, f'h_dim should be positive, but found {args.h_dim}.'
    assert args.h_dim < N, f'h_dim should be less than the sample size, but found {args.h_dim}.'
    assert data.max().astype(int) <= 1, f'data should be normalized to [0, 1].'


def checkargs_kpca(args, data, N):
    assert args.h_dim > 0, f'h_dim should be positive, but found {args.h_dim}.'
    assert args.h_dim < N, f'h_dim should be less than the sample size, but found {args.h_dim}.'
    assert args.sig2 > 0, f'sig2 should be positive, but found {args.sig2}.'
    assert data.max().astype(int) <= 1, f'data should be normalized to [0, 1].'


def checkargs_fwsprs_kpca(args, data, N):
    checkargs_kpca(args, data, N)
    assert args.rho > 0, f'rho should be positive, but found {args.rho}.'
    assert args.err_tol > 1e-8, f'err_tol should > 1e-8, but found {args.err_tol}.'
    assert args.n_warmup > 0, f'n_warmup should be positive, but found {args.n_warmup}.'
    assert args.max_iter > 100, f'max_iter should > 100, but found {args.max_iter}.'


def checkargs_fwsprs_dpca(args, data, N):
    checkargs_pca(args, data, N)
    assert args.rho > 0, f'rho should be positive, but found {args.rho}.'
    assert args.err_tol > 1e-8, f'err_tol should > 1e-8, but found {args.err_tol}.'
    assert args.n_warmup > 0, f'n_warmup should be positive, but found {args.n_warmup}.'
    assert args.max_iter > 100, f'max_iter should > 100, but found {args.max_iter}.'


def checkargs_iwsprs_kpca(args, data, N):
    checkargs_kpca(args, data, N)
    assert args.c_sprs > 0, f'Sparsity parameter should be positive, but found {args.c_sprs}.'
    assert args.lr > 0, f'Learning rate should be positive, but found {args.lr}.'
    assert args.max_iter > 0, f'max_iter should be positive, but found {args.max_iter}.'
    assert args.err_tol > 0, f'eps should be positive, but found {args.err_tol}.'


def checkargs_iwsprs_dpca(args, data, N):
    checkargs_pca(args, data, N)
    assert args.c_sprs > 0, f'Sparsity parameter should be positive, but found {args.c_sprs}.'
    assert args.lr > 0, f'Learning rate should be positive, but found {args.lr}.'
    assert args.max_iter > 0, f'max_iter should be positive, but found {args.max_iter}.'
    assert args.err_tol > 0, f'eps should be positive, but found {args.err_tol}.'


def checkargs_sdrkm(args, data, N):
    assert args.lr_h > 0 and args.lr_sig > 0, f'Learning rate should be positive, but found {args.lr_h} and {args.lr_sig}.'
    assert args.max_iter > 0, f'max_iter should be positive, but found {args.max_iter}.'
    assert args.err_tol > 0, f'err_tol should be positive, but found {args.err_tol}.'
    assert args.init_sig_sparse > 0 and args.final_sig_sparse > 0, f'Bandwidth parameters should be positive, but found {args.init_sig_sparse} and {args.final_sig_sparse}.'
    assert args.decay > 0, f'Decaying parameter should be positive, but found {args.decay}.'
    assert args.gamma > 0, f'Layer trade-off parameter should be positive, but found {args.gamma}.'

    check_list = lambda x: (x[0] > 0) and (x[1] > 0)
    check_dim = lambda x: (x[0] < N) and (x[1] < N)
    assert check_list(args.sig), f'Bandwidth parameters should be positive, but found {args.sig}.'
    assert check_list(args.h_dim) and check_dim(args.h_dim), f'Latent dimensions should be positive and < sample size, but found {args.h_dim}.'
    assert check_list(args.c_sprs), f'Sparsity parameters should be positive, but found {args.c_sprs}.'
    assert data.max().astype(int) <= 1, f'data should be normalized to [0, 1].'


def checkargs_gen_rkm(args, data, N):
    assert args.c_accu > 0 and args.c_stab > 0, f'Network parameters should be positive, but found {args.c_accu} and {args.c_stab}.'
    assert args.h_dim > 0, f'h_dim should be positive, but found {args.h_dim}.'
    assert args.h_dim < N, f'h_dim should be less than the batch size, but found {args.h_dim}.'
    assert args.lr > 0, f'Learning rate should be positive, but found {args.lr}.'
    assert args.max_epoch > 0, f'max_iter should be positive, but found {args.max_epoch}.'
    assert data.max().astype(int) <= 1, f'data should be normalized to [0, 1].'


def checkargs_ct_gen_rkm(args, data, N):
    checkargs_gen_rkm(args, data, N)
    assert args.const > 0, f'threshold level should be positive.'


def checkargs_st_gen_rkm(args, data, N):
    checkargs_gen_rkm(args, data, N)
    assert args.const > 0, f'threshold level should be positive.'
    assert args.const < args.p_feat, f'threshold level should be positive.'


def checkargs_spca_gen_rkm(args, data, N):
    checkargs_gen_rkm(args, data, N)
    assert args.th > 0, f'threshold level should be positive.'
    assert args.inner_iter > 0, f'inner iterations should be positive.'