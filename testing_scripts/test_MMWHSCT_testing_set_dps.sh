#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python test/inference.py \
    model_path=./Model/DiffAtlas_MMWHS-CT_full \
    model_num=pretrained_MMWHSCT_full \
    dataset=MMWHS \
    data_type=CT \
    mode=test \
    diffusion_img_size=64 \
    diffusion_depth_size=64 \
    diffusion_num_channels=6 \
    timesteps=300 \
    dir_name=MMWHSCT_testing_set_dps \
    root_dir=./data/MMWHS/CT/testing_set \
    sampler.name=ddim \
    sampler.ddim_steps=50 \
    sampler.eta=0.0 \
    guidance.mode=dps \
    guidance.gamma=0.5 \
    guidance.gamma_schedule=mid \
    guidance.lambda_lncc=1.0 \
    guidance.lambda_edge=0.1 \
    guidance.lncc_win=9 \
    guidance.grad_clip=1.0 \
    guidance.apply_to=mask_only
