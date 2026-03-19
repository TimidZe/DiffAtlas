#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 python test/inference.py \
    model_path=./Model/DiffAtlas_TotalSegmentator \
    model_num=pretrained_TotalSegmentator_for_MMWHSMRITest \
    dataset=MMWHS \
    data_type=MRI \
    mode=test \
    diffusion_img_size=64 \
    diffusion_depth_size=64 \
    diffusion_num_channels=6 \
    timesteps=300 \
    dir_name=MMWHSMRI_all_from_TotalSegmentator \
    root_dir=./data/MMWHS/MRI/all
