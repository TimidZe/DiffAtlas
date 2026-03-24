import os
import io
import csv
import blobfile as bf
import torch as th
import sys
from pathlib import Path
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from ddpm import Unet3D, GaussianDiffusion_Nolatent
from Dataset.TS_Dataset import get_TS_dataloader
from Dataset.MMWHS_Dataset import get_MMWHS_dataloader
import torchio as tio
from omegaconf import DictConfig
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
import atexit
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt


NUM_FOREGROUND_CLASSES = 5


def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device("cuda")
        return th.device("cpu")
    return th.device(device)


def load_state_dict(path, backend=None, **kwargs):
    with bf.BlobFile(path, "rb") as f:
        data = f.read()
    return th.load(io.BytesIO(data), **kwargs)


def resolve_checkpoint_path(model_path, model_num):
    model_path = Path(os.path.expanduser(str(model_path)))
    if model_path.is_file():
        return model_path
    return model_path / f"model-{model_num}.pt"


def strip_module_prefix(state_dict):
    weights_dict = {}
    for key, value in state_dict.items():
        weights_dict[key.replace("module.", "")] = value
    return weights_dict


def set_seed(seed, deterministic=True):
    if seed is None:
        return

    th.manual_seed(seed)
    np.random.seed(seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)

    th.backends.cudnn.deterministic = deterministic
    th.backends.cudnn.benchmark = not deterministic


def build_case_seed(base_seed, case_index, trial_index, seed_num):
    if base_seed is None:
        return None
    return int(base_seed) + case_index * seed_num + trial_index


try:
    import ctypes

    libgcc_s = ctypes.CDLL("libgcc_s.so.1")
except:
    pass


def get_dice(preds, labels):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.reshape(preds.shape[0], -1)
    target = labels.reshape(labels.shape[0], -1)
    if np.sum(target) == 0 and np.sum(predict) == 0:
        return 1.0
    else:
        num = np.sum(np.multiply(predict, target), axis=1)
        den = np.sum(predict, axis=1) + np.sum(target, axis=1)
        dice = 2 * num / den
        return dice.mean()


def ignore_background(y_pred: torch.Tensor, y: torch.Tensor):
    return y_pred[:, 1:], y[:, 1:]


def prepare_spacing(spacing, batch_size, img_dim):
    if spacing is None:
        spacing = tuple([1.0] * img_dim)
    if isinstance(spacing, (int, float)):
        spacing = tuple([float(spacing)] * img_dim)
    elif isinstance(spacing, (tuple, list)):
        if len(spacing) == 1:
            spacing = tuple([float(spacing[0])] * img_dim)
        elif len(spacing) == img_dim:
            spacing = tuple(float(s) for s in spacing)
        else:
            raise ValueError("spacing should be a number or sequence of numbers matching image dimensions")
    return [spacing] * batch_size


def get_edge_surface_distance(pred, gt, distance_metric="euclidean", spacing=None, use_subvoxels=False, symmetric=True, class_index=None):
    pred = pred.cpu().numpy().astype(bool)
    gt = gt.cpu().numpy().astype(bool)

    edges_pred = ndimage.binary_dilation(pred).astype(bool) ^ pred
    edges_gt = ndimage.binary_dilation(gt).astype(bool) ^ gt

    if distance_metric == "euclidean":
        dt_pred = distance_transform_edt(~edges_pred, sampling=spacing)
        dt_gt = distance_transform_edt(~edges_gt, sampling=spacing)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    distances_pred_gt = dt_gt[edges_pred]
    distances_gt_pred = dt_pred[edges_gt]

    if use_subvoxels:
        areas = None
    else:
        areas = None

    return (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas


def compute_surface_dice(y_pred, y, class_thresholds, include_background=False, distance_metric="euclidean", spacing=None, use_subvoxels=False):
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y).")

    batch_size, n_class = y_pred.shape[:2]
    img_dim = y_pred.ndim - 2
    spacing_list = prepare_spacing(spacing=spacing, batch_size=batch_size, img_dim=img_dim)
    nsd = torch.empty((batch_size, n_class), device=y_pred.device, dtype=torch.float)

    for b, c in np.ndindex(batch_size, n_class):
        (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas = get_edge_surface_distance(
            y_pred[b, c],
            y[b, c],
            distance_metric=distance_metric,
            spacing=spacing_list[b],
            use_subvoxels=use_subvoxels,
            symmetric=True,
            class_index=c,
        )

        boundary_complete = len(distances_pred_gt) + len(distances_gt_pred)
        boundary_correct = torch.sum(torch.tensor(distances_pred_gt <= class_thresholds[c])) + torch.sum(
            torch.tensor(distances_gt_pred <= class_thresholds[c])
        )

        if boundary_complete == 0:
            nsd[b, c] = torch.tensor(float("nan"))
        else:
            nsd[b, c] = boundary_correct / boundary_complete

    return nsd


class NSDMetric(nn.Module):
    def __init__(self, n_classes, percentile=95):
        super(NSDMetric, self).__init__()
        self.n_classes = n_classes
        self.class_thresholds = [1.0] * n_classes

    def forward(self, inputs, target, spacing=(1.0, 1.0, 1.0), softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        inputs = F.one_hot(inputs, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()

        nsd_scores = compute_surface_dice(
            inputs,
            target,
            class_thresholds=self.class_thresholds,
            include_background=False,
            spacing=spacing,
        )
        return nsd_scores[0]


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)
                f.flush()

    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()


def metric_field_names(prefix):
    return [f"{prefix}_c{i}" for i in range(1, NUM_FOREGROUND_CLASSES + 1)]


def flatten_metric_row(prefix, values):
    return {f"{prefix}_c{i + 1}": float(values[i]) for i in range(len(values))}


def metric_mean(values):
    return float(np.mean([float(value) for value in values]))


def ensure_csv_header(path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def append_csv_row(path, fieldnames, row):
    ensure_csv_header(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(row)


def sanitize_token(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))


def infer_model_source(model_path, model_num):
    joined = f"{model_path} {model_num}".lower()
    return "pretrained" if "pretrained" in joined else "self_trained"


def build_run_id(conf, timestamp):
    return f"{timestamp}_{sanitize_token(conf.dir_name)}_{sanitize_token(conf.model_num)}"


CASE_METRIC_FIELDS = [
    "run_id",
    "timestamp",
    "model_name",
    "model_num",
    "model_source",
    "dataset",
    "data_type",
    "mode",
    "root_dir",
    "dir_name",
    "case_index",
    "case_name",
    "seed",
    "seed_num",
    *metric_field_names("dice"),
    "dice_mean",
    *metric_field_names("nsd"),
    "nsd_mean",
]


RUN_SUMMARY_FIELDS = [
    "run_id",
    "timestamp",
    "model_name",
    "model_path",
    "checkpoint_path",
    "model_num",
    "model_source",
    "weight_key",
    "dataset",
    "data_type",
    "mode",
    "root_dir",
    "dir_name",
    "diffusion_img_size",
    "diffusion_depth_size",
    "diffusion_num_channels",
    "timesteps",
    "seed",
    "seed_num",
    "deterministic",
    "num_cases",
    *metric_field_names("dice"),
    "dice_mean",
    *metric_field_names("nsd"),
    "nsd_mean",
]


@hydra.main(config_path="confs", config_name="infer", version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_container(conf, resolve=True))

    log_dir = "log_inference"
    filename = os.path.join(log_dir, conf.dir_name, str(conf.model_num) + ".log")
    os.makedirs(os.path.join(log_dir, conf.dir_name), exist_ok=True)
    log_file = open(filename, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    atexit.register(lambda: log_file.close())

    visdir = "inference_visualization"
    vis_dir_name = os.path.join(visdir, conf.dir_name, str(conf.model_num))

    device = dev(conf.get("device"))

    model = Unet3D(
        dim=conf.diffusion_img_size,
        dim_mults=conf.dim_mults,
        channels=conf.diffusion_num_channels,
        cond_dim=16,
    )

    diffusion = GaussianDiffusion_Nolatent(
        model,
        image_size=conf.diffusion_img_size,
        num_frames=conf.diffusion_depth_size,
        channels=conf.diffusion_num_channels,
        timesteps=conf.timesteps,
        loss_type=conf.loss_type,
        use_guide=conf.use_guide,
    )
    diffusion.to(device)

    checkpoint_path = resolve_checkpoint_path(conf.model_path, conf.model_num)
    checkpoint = load_state_dict(os.path.expanduser(str(checkpoint_path)), map_location="cpu")
    if conf.weight_key not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain weight key '{conf.weight_key}'")
    weights_dict = strip_module_prefix(checkpoint[conf.weight_key])
    diffusion.load_state_dict(weights_dict)

    diffusion.eval()
    model.eval()

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    run_id = build_run_id(conf, datetime.now().strftime("%Y%m%d_%H%M%S"))
    results_dir = Path(conf.get("results_dir", "results"))
    run_summary_path = results_dir / "run_summary.csv"
    case_metrics_path = results_dir / "case_metrics.csv"
    model_name = Path(os.path.expanduser(str(conf.model_path))).name
    model_source = infer_model_source(conf.model_path, conf.model_num)
    save_outputs = bool(conf.get("save_outputs", True))

    print(f"results summary csv: {run_summary_path}")
    print(f"results case csv: {case_metrics_path}")
    print("sampling...")

    if conf.dataset == "MMWHS":
        dataloader = get_MMWHS_dataloader(root_dir=conf.root_dir, mode=conf.mode, data_type=conf.data_type)
    elif conf.dataset == "TS":
        dataloader = get_TS_dataloader(root_dir=conf.root_dir, mode=conf.mode)
    else:
        raise ValueError("No Such Dataset")

    idx = 0
    dice_total = [0.0] * NUM_FOREGROUND_CLASSES
    nsd_total = [0.0] * NUM_FOREGROUND_CLASSES
    seed_num = conf.seed_num
    set_seed(conf.seed, deterministic=conf.deterministic)

    for batch in iter(dataloader):
        idx += 1
        for k in batch.keys():
            if k == "affine":
                continue
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)
                if th.is_floating_point(batch[k]):
                    batch[k] = batch[k].float()
        affine = torch.as_tensor(batch["affine"]).squeeze(0).cpu()

        real_image = batch["img"]
        real_mask = batch.get("mask").cpu()
        real_mask_sdf = batch.get("mask_sdf").cpu()
        gt_name = batch["name"][0]

        gt_name = gt_name.split("_image")[0]
        gt_name = gt_name.split("-image")[0]

        print(idx, ":", gt_name)

        dice = [0.0] * NUM_FOREGROUND_CLASSES
        nsd = [0.0] * NUM_FOREGROUND_CLASSES
        for trial_index in range(seed_num):
            seed = build_case_seed(conf.seed, idx - 1, trial_index, seed_num)
            print("     seed:", seed)
            set_seed(seed, deterministic=conf.deterministic)

            sample_fn = diffusion.p_sample_loop
            result = sample_fn(
                shape_image=real_image.size(),
                shape_mask=real_mask_sdf.size(),
                device=device,
                image=real_image,
            )

            gen_mask = result[:, 1:(result.size()[1]), :, :, :]
            gen_mask_de_sdf = (gen_mask < 0.0).to(dtype=gen_mask.dtype)

            if save_outputs:
                real_img_to_save = tio.ScalarImage(tensor=real_image.squeeze(0).cpu(), channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, "Image"), exist_ok=True)
                real_img_to_save.save(os.path.join(vis_dir_name, "Image", f"{gt_name}-image-real.nii.gz"))

            for i in range(gen_mask.size()[1]):
                gen_mask_i = gen_mask[:, i:i + 1, :, :, :]
                gen_mask_i_cpu = gen_mask_i.cpu()
                gen_mask_i_de_sdf = gen_mask_de_sdf[:, i:i + 1, :, :, :].cpu()

                if save_outputs:
                    gen_mask_sdf_to_save = tio.LabelMap(tensor=gen_mask_i_cpu.squeeze(1), channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, "Label"), exist_ok=True)
                    gen_mask_sdf_to_save.save(os.path.join(vis_dir_name, "Label", f"{gt_name}-{seed}-label-sdf-{i+1}-gen.nii.gz"))

                    gen_mask_de_sdf_to_save = tio.LabelMap(tensor=gen_mask_i_de_sdf.squeeze(1), channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, "Label"), exist_ok=True)
                    gen_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, "Label", f"{gt_name}-{seed}-label-de-sdf-{i+1}-gen.nii.gz"))

                    real_mask_sdf_to_save = tio.LabelMap(tensor=real_mask_sdf[:, i, :, :, :], channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, "Label"), exist_ok=True)
                    real_mask_sdf_to_save.save(os.path.join(vis_dir_name, "Label", f"{gt_name}-{seed}-label-sdf-{i+1}-real.nii.gz"))

                    real_mask_de_sdf_to_save = tio.LabelMap(tensor=real_mask[:, i, :, :, :], channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, "Label"), exist_ok=True)
                    real_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, "Label", f"{gt_name}-{seed}-label-de-sdf-{i+1}-real.nii.gz"))

                real_mask_i = real_mask[:, i:i + 1, :, :, :]
                dice_value = float(get_dice(real_mask_i.numpy(), gen_mask_i_de_sdf.numpy()))
                print(f"        {i+1}_dice:", dice_value)
                dice[i] += dice_value

            background_mask = (gen_mask_de_sdf.sum(dim=1, keepdim=True) == 0).to(dtype=gen_mask_de_sdf.dtype)
            gen_mask_togather = torch.cat((background_mask, gen_mask_de_sdf), dim=1)
            gen_mask_togather = torch.argmax(gen_mask_togather, dim=1)
            if save_outputs:
                gen_mask_togather_to_save = tio.LabelMap(tensor=gen_mask_togather.cpu().int(), channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, "Label"), exist_ok=True)
                gen_mask_togather_to_save.save(os.path.join(vis_dir_name, "Label", f"{gt_name}-{seed}-label-together-gen.nii.gz"))

            get_nsd = NSDMetric(n_classes=6)
            real_mask_togather = real_mask
            background_mask = (real_mask_togather.sum(dim=1, keepdim=True) == 0).to(dtype=real_mask_togather.dtype)
            real_mask_togather_ = torch.cat((background_mask, real_mask_togather), dim=1)
            real_mask_togather_ = torch.argmax(real_mask_togather_, dim=1)
            nnsd = get_nsd(inputs=gen_mask_togather.long().cpu(), target=real_mask_togather_.long().cpu())
            for i in range(NUM_FOREGROUND_CLASSES):
                nsd[i] += float(nnsd[i].item())
            print(f"        {nnsd}")

        dice_avg = [item / seed_num for item in dice]
        nsd_avg = [item / seed_num for item in nsd]
        print("     average:")
        print(f"        dice:", dice_avg)
        print(f"        nsd:", nsd_avg)
        for i in range(NUM_FOREGROUND_CLASSES):
            dice_total[i] += dice_avg[i]
            nsd_total[i] += nsd_avg[i]

        case_row = {
            "run_id": run_id,
            "timestamp": timestamp,
            "model_name": model_name,
            "model_num": str(conf.model_num),
            "model_source": model_source,
            "dataset": str(conf.dataset),
            "data_type": str(conf.get("data_type", "")),
            "mode": str(conf.mode),
            "root_dir": str(conf.root_dir),
            "dir_name": str(conf.dir_name),
            "case_index": idx,
            "case_name": gt_name,
            "seed": "" if conf.seed is None else int(conf.seed),
            "seed_num": int(seed_num),
            "dice_mean": metric_mean(dice_avg),
            "nsd_mean": metric_mean(nsd_avg),
            **flatten_metric_row("dice", dice_avg),
            **flatten_metric_row("nsd", nsd_avg),
        }
        append_csv_row(case_metrics_path, CASE_METRIC_FIELDS, case_row)

    dice_total_avg = [item / idx for item in dice_total]
    nsd_total_avg = [item / idx for item in nsd_total]
    print("total average:")
    print(f"    dice:", dice_total_avg)
    print(f"    nsd:", nsd_total_avg)

    summary_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_name": model_name,
        "model_path": str(conf.model_path),
        "checkpoint_path": str(checkpoint_path),
        "model_num": str(conf.model_num),
        "model_source": model_source,
        "weight_key": str(conf.weight_key),
        "dataset": str(conf.dataset),
        "data_type": str(conf.get("data_type", "")),
        "mode": str(conf.mode),
        "root_dir": str(conf.root_dir),
        "dir_name": str(conf.dir_name),
        "diffusion_img_size": int(conf.diffusion_img_size),
        "diffusion_depth_size": int(conf.diffusion_depth_size),
        "diffusion_num_channels": int(conf.diffusion_num_channels),
        "timesteps": int(conf.timesteps),
        "seed": "" if conf.seed is None else int(conf.seed),
        "seed_num": int(seed_num),
        "deterministic": bool(conf.deterministic),
        "num_cases": int(idx),
        "dice_mean": metric_mean(dice_total_avg),
        "nsd_mean": metric_mean(nsd_total_avg),
        **flatten_metric_row("dice", dice_total_avg),
        **flatten_metric_row("nsd", nsd_total_avg),
    }
    append_csv_row(run_summary_path, RUN_SUMMARY_FIELDS, summary_row)


if __name__ == "__main__":
    main()
