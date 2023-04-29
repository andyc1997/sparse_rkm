#!/bin/bash
echo "Script executed from: ${PWD}"
python ../kpca_model.py --dataset norb --h_dim 5 --sig2 90 "/content/drive/My Drive/sparse-deep-rkm-research-proj/2023-MAR-experiments/data/norb-N-2500/img-norb-N-2500-seed-1278.npy"
echo "Finished"