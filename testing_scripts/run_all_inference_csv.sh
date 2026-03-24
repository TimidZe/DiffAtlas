#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="/home/estar/TZNEW/python_env_TZ/diffatlas/bin/python"
BATCH_NAME="${1:-batch_20260324_all_models}"
WORKER="${2:-all}"
RESULTS_ROOT="results/${BATCH_NAME}"

mkdir -p "${RESULTS_ROOT}"

run_job() {
    local gpu="$1"
    local results_dir="$2"
    local dir_name="$3"
    local model_path="$4"
    local model_num="$5"
    local dataset="$6"
    local data_type="$7"
    local root_dir="$8"

    local args=(
        test/inference_csv.py
        "model_path=${model_path}"
        "model_num=${model_num}"
        "dataset=${dataset}"
        "mode=test"
        "diffusion_img_size=64"
        "diffusion_depth_size=64"
        "diffusion_num_channels=6"
        "timesteps=300"
        "dir_name=${dir_name}"
        "root_dir=${root_dir}"
        "+results_dir=${results_dir}"
        "+save_outputs=false"
    )

    if [[ -n "${data_type}" ]]; then
        args+=("data_type=${data_type}")
    fi

    echo "[GPU ${gpu}] ${dir_name} :: ${model_path} :: ${model_num}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${args[@]}"
}

run_queue() {
    local gpu="$1"
    shift
    local results_dir="${RESULTS_ROOT}/gpu${gpu}"
    mkdir -p "${results_dir}"

    while (($#)); do
        run_job "${gpu}" "${results_dir}" "$1" "$2" "$3" "$4" "$5" "$6"
        shift 6
    done
}

queue_gpu1() {
    run_queue 1 \
        batch_TS_test_manual_gpu5_model_1 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_TotalSegmentator_gpu5 1 TS "" ./data/TotalSegmentator/test \
        batch_TS_test_manual_gpu5_model_2 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_TotalSegmentator_gpu5 2 TS "" ./data/TotalSegmentator/test \
        batch_MMWHSMRI_all_manual_gpu1_model_1 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 1 MMWHS MRI ./data/MMWHS/MRI/all
}

queue_gpu2() {
    run_queue 2 \
        batch_TS_test_manual_gpu5_model_3 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_TotalSegmentator_gpu5 3 TS "" ./data/TotalSegmentator/test \
        batch_MMWHSMRI_all_manual_gpu1_model_2 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 2 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_3 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 3 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_4 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 4 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_5 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 5 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_6 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 6 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSCT_all_manual_gpu3_model_1 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 1 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_testing_set_manual_gpu2_model_1 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 1 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_2 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 2 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_3 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 3 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_4 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 4 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_5 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 5 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_1 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 1 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_2 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 2 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_3 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 3 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_4 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 4 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_5 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 5 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSCT_testing_set_pretrained_MMWHSCT_full ./Model/DiffAtlas_MMWHS-CT_full pretrained_MMWHSCT_full MMWHS CT ./data/MMWHS/CT/testing_set
}

queue_gpu3() {
    run_queue 3 \
        batch_TS_test_manual_gpu5_model_4 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_TotalSegmentator_gpu5 4 TS "" ./data/TotalSegmentator/test \
        batch_MMWHSMRI_all_manual_gpu1_model_7 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 7 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_8 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 8 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_9 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 9 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSMRI_all_manual_gpu1_model_10 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1 10 MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSCT_all_manual_gpu3_model_2 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 2 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_3 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 3 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_testing_set_manual_gpu2_model_6 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 6 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_7 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 7 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_8 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 8 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_9 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 9 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSCT_testing_set_manual_gpu2_model_10 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_full_gpu2 10 MMWHS CT ./data/MMWHS/CT/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_6 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 6 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_7 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 7 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_8 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 8 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_9 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 9 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_manual_gpu4_model_10 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_full_gpu4 10 MMWHS MRI ./data/MMWHS/MRI/testing_set \
        batch_MMWHSMRI_testing_set_pretrained_MMWHSMRI_full ./Model/DiffAtlas_MMWHS-MRI_full pretrained_MMWHSMRI_full MMWHS MRI ./data/MMWHS/MRI/testing_set
}

queue_gpu4() {
    run_queue 4 \
        batch_TS_test_manual_gpu5_model_5 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_TotalSegmentator_gpu5 5 TS "" ./data/TotalSegmentator/test \
        batch_MMWHSCT_all_manual_gpu3_model_4 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 4 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_5 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 5 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_6 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 6 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_7 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 7 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_8 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 8 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_9 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 9 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSCT_all_manual_gpu3_model_10 ./Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-MRI_all_gpu3 10 MMWHS CT ./data/MMWHS/CT/all \
        batch_MMWHSMRI_all_pretrained_MMWHSCT_all ./Model/DiffAtlas_MMWHS-CT_all pretrained_MMWHSCT_all MMWHS MRI ./data/MMWHS/MRI/all \
        batch_MMWHSCT_all_pretrained_MMWHSMRI_all ./Model/DiffAtlas_MMWHS-MRI_all pretrained_MMWHSMRI_all MMWHS CT ./data/MMWHS/CT/all
}

case "${WORKER}" in
    gpu1)
        queue_gpu1
        ;;
    gpu2)
        queue_gpu2
        ;;
    gpu3)
        queue_gpu3
        ;;
    gpu4)
        queue_gpu4
        ;;
    merge)
        "${PYTHON_BIN}" test/merge_inference_results.py "${RESULTS_ROOT}"
        ;;
    all)
        echo "Use worker mode: gpu1 | gpu2 | gpu3 | gpu4 | merge"
        exit 1
        ;;
    *)
        echo "Unknown worker mode: ${WORKER}"
        exit 1
        ;;
esac
