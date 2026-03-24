#!/usr/bin/env bash
set -euo pipefail

# Launch 5 training jobs concurrently (one per GPU: 1..5) with single-GPU-safe settings.
# Logs:   ./log_train/<run_id>/
# Models: ./Model/<run_id>/
#
# Usage:
#   bash ./traing_scripts/train_5jobs_singlegpu.sh
#   RUN_ID=my_run bash ./traing_scripts/train_5jobs_singlegpu.sh
#   PY=/path/to/python RUN_ID=my_run bash ./traing_scripts/train_5jobs_singlegpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-/home/estar/TZNEW/python_env_TZ/diffatlas/bin/python}"
RUN_ID="${RUN_ID:-manual_$(date +%Y%m%d_%H%M%S)_singlegpu}"

MODEL_BASE="./Model/${RUN_ID}"
LOG_BASE="./log_train/${RUN_ID}"

mkdir -p "$MODEL_BASE" "$LOG_BASE"

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  local log_file="${LOG_BASE}/${name}.log"

  # -u for unbuffered logs; CUDA_VISIBLE_DEVICES limits DataParallel to a single GPU.
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" \
    "$PY" -u train/train.py "$@" \
    </dev/null >"$log_file" 2>&1 &

  local pid="$!"
  echo "${pid}" > "${LOG_BASE}/${name}.pid"
  echo "launched ${name} on GPU ${gpu} (pid=${pid})"
}

# Original scripts used a global batch size of 12 across multiple GPUs.
# For single-GPU launches, reduce per-step batch and increase gradient accumulation
# to keep an approximate effective batch.

# GPU 1: MMWHS CT all
launch 1 "train_MMWHSCT_all_gpu1" \
  model=ddpm \
  dataset=MMWHS dataset.data_type=CT dataset.root_dir=./data/MMWHS/CT/all dataset.mode=train \
  model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=6 \
  model.batch_size=2 model.gradient_accumulate_every=12 \
  model.results_folder="${MODEL_BASE}/DiffAtlas_MMWHS-CT_all_gpu1" \
  model.load_milestone=False model.save_and_sample_every=1000 model.train_num_steps=10000 model.timesteps=300 model.num_workers=20

# GPU 2: MMWHS CT full training set
launch 2 "train_MMWHSCT_full_gpu2" \
  model=ddpm \
  dataset=MMWHS dataset.data_type=CT dataset.root_dir=./data/MMWHS/CT/training_set_full dataset.mode=train \
  model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=6 \
  model.batch_size=2 model.gradient_accumulate_every=12 \
  model.results_folder="${MODEL_BASE}/DiffAtlas_MMWHS-CT_full_gpu2" \
  model.load_milestone=False model.save_and_sample_every=1000 model.train_num_steps=10000 model.timesteps=300 model.num_workers=20

# GPU 3: MMWHS MRI all
launch 3 "train_MMWHSMRI_all_gpu3" \
  model=ddpm \
  dataset=MMWHS dataset.data_type=MRI dataset.root_dir=./data/MMWHS/MRI/all dataset.mode=train \
  model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=6 \
  model.batch_size=1 model.gradient_accumulate_every=24 \
  model.results_folder="${MODEL_BASE}/DiffAtlas_MMWHS-MRI_all_gpu3" \
  model.load_milestone=False model.save_and_sample_every=1000 model.train_num_steps=10000 model.timesteps=300 model.num_workers=20

# GPU 4: MMWHS MRI full training set
launch 4 "train_MMWHSMRI_full_gpu4" \
  model=ddpm \
  dataset=MMWHS dataset.data_type=MRI dataset.root_dir=./data/MMWHS/MRI/training_set_full dataset.mode=train \
  model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=6 \
  model.batch_size=1 model.gradient_accumulate_every=24 \
  model.results_folder="${MODEL_BASE}/DiffAtlas_MMWHS-MRI_full_gpu4" \
  model.load_milestone=False model.save_and_sample_every=1000 model.train_num_steps=10000 model.timesteps=300 model.num_workers=20

# GPU 5: TotalSegmentator train
launch 5 "train_TotalSegmentator_gpu5" \
  model=ddpm \
  dataset=TS dataset.root_dir=./data/TotalSegmentator/train dataset.mode=train \
  model.diffusion_img_size=64 model.diffusion_depth_size=64 model.diffusion_num_channels=6 \
  model.batch_size=1 model.gradient_accumulate_every=24 \
  model.results_folder="${MODEL_BASE}/DiffAtlas_TotalSegmentator_gpu5" \
  model.load_milestone=False model.save_and_sample_every=2500 model.train_num_steps=50000 model.timesteps=300 model.num_workers=20

echo
echo "logs:   ${LOG_BASE}"
echo "models: ${MODEL_BASE}"
echo
echo "monitor:"
echo "  tail -f ${LOG_BASE}/train_MMWHSCT_all_gpu1.log"
echo "  nvidia-smi"

