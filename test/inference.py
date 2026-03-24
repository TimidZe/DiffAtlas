import os
import io
import blobfile as bf
import torch as th
import sys
from pathlib import Path
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

def dev(device):
    if device is None:
        if th.cuda.is_available():
            return th.device(f"cuda")
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
        weights_dict[key.replace('module.', '')] = value
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
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
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
    # Convert tensors to numpy arrays and ensure boolean type
    pred = pred.cpu().numpy().astype(bool)
    gt = gt.cpu().numpy().astype(bool)
    
    # Get surface voxels using boolean operations
    edges_pred = ndimage.binary_dilation(pred).astype(bool) ^ pred
    edges_gt = ndimage.binary_dilation(gt).astype(bool) ^ gt
    
    # Compute distance transforms
    if distance_metric == "euclidean":
        dt_pred = distance_transform_edt(~edges_pred, sampling=spacing)
        dt_gt = distance_transform_edt(~edges_gt, sampling=spacing)
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    
    # Get surface distances
    distances_pred_gt = dt_gt[edges_pred]
    distances_gt_pred = dt_pred[edges_gt]
    
    if use_subvoxels:
        areas = None  # Simplified version without subvoxel precision
    else:
        areas = None
        
    return (edges_pred, edges_gt), (distances_pred_gt, distances_gt_pred), areas


def compute_surface_dice(y_pred, y, class_thresholds, include_background=False, 
                       distance_metric="euclidean", spacing=None, use_subvoxels=False):
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    if not isinstance(y_pred, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("y_pred and y must be PyTorch Tensor.")

    if y_pred.ndimension() not in (4, 5) or y.ndimension() not in (4, 5):
        raise ValueError("y_pred and y should be one-hot encoded: [B,C,H,W] or [B,C,H,W,D].")

    if y_pred.shape != y.shape:
        raise ValueError(
            f"y_pred and y should have same shape, but instead, shapes are {y_pred.shape} (y_pred) and {y.shape} (y)."
        )

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
        boundary_correct = torch.sum(torch.tensor(distances_pred_gt <= class_thresholds[c])) + \
                         torch.sum(torch.tensor(distances_gt_pred <= class_thresholds[c]))

        if boundary_complete == 0:
            nsd[b, c] = torch.tensor(float('nan'))
        else:
            nsd[b, c] = boundary_correct / boundary_complete

    return nsd

class NSDMetric(nn.Module):
    def __init__(self, n_classes, percentile=95):
        super(NSDMetric, self).__init__()
        self.n_classes = n_classes
        self.class_thresholds = [1.0] * n_classes  # 1mm threshold for all classes

    def forward(self, inputs, target, spacing=(1.0, 1.0, 1.0), softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            
        # Convert to one-hot encoding
        inputs = F.one_hot(inputs, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        target = F.one_hot(target, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
        
        nsd_scores = compute_surface_dice(
            inputs, 
            target,
            class_thresholds=self.class_thresholds,
            include_background=False,
            spacing=spacing
        )
        
        # return torch.nanmean(nsd_scores)  # Average over batch and classes, ignoring NaN values
        return nsd_scores[0]  # Average over batch and classes, ignoring NaN values
    

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


@hydra.main(config_path='confs', config_name='infer', version_base=None)
def main(conf: DictConfig):
    print(OmegaConf.to_container(conf, resolve=True))

    log_dir = "log_inference"
    filename = os.path.join(log_dir, conf.dir_name,  str(conf.model_num) + ".log")
    os.makedirs(os.path.join(log_dir, conf.dir_name), exist_ok=True)
    log_file = open(filename, 'w', encoding='utf-8')  
    sys.stdout = Tee(sys.stdout, log_file)
    atexit.register(lambda: log_file.close())
    
    visdir = "inference_visualization"
    vis_dir_name = os.path.join(visdir, conf.dir_name, str(conf.model_num))

    device = dev(conf.get('device'))

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
    save_outputs = bool(conf.get('save_outputs', True))

    sampler_cfg = conf.get('sampler')
    if sampler_cfg is None:
        sampler_cfg = OmegaConf.create({
            'name': 'ddpm',
            'ddim_steps': 50,
            'eta': 0.0,
        })

    guidance_cfg = conf.get('guidance')
    if guidance_cfg is None:
        guidance_cfg = OmegaConf.create({
            'mode': 'replace',
            'gamma': 0.5,
            'gamma_schedule': 'mid',
            'lambda_lncc': 1.0,
            'lambda_edge': 0.1,
            'lncc_win': 9,
            'grad_clip': 1.0,
            'apply_to': 'mask_only',
            'log_every_step': False,
        })

    print("sampling...")
    
    num_workers = conf.get('num_workers')

    if conf.dataset == 'MMWHS':
        dataloader = get_MMWHS_dataloader(root_dir=conf.root_dir, mode=conf.mode, data_type=conf.data_type, num_workers=num_workers) 
    elif conf.dataset == 'TS' :
        dataloader = get_TS_dataloader(root_dir=conf.root_dir, mode=conf.mode, num_workers=num_workers)
    else :
        raise ValueError ("No Such Dataset")
    
    idx = 0
    dice_total = [0, 0, 0, 0, 0]
    nsd_total = [0, 0, 0, 0, 0]
    seed_num = conf.seed_num
    set_seed(conf.seed, deterministic=conf.deterministic)
    for batch in iter(dataloader):
        idx += 1
        for k in batch.keys():
            if k == 'affine':
                continue
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)
                if th.is_floating_point(batch[k]):
                    batch[k] = batch[k].float()
        affine = torch.as_tensor(batch['affine']).squeeze(0).cpu()

        real_image = batch["img"]
        real_mask = batch.get('mask').cpu()
        real_mask_sdf = batch.get('mask_sdf').cpu()
        gt_name = batch['name'][0]

        gt_name = gt_name.split('_image')[0]
        gt_name = gt_name.split('-image')[0]
        
        print(idx,":", gt_name)

        dice = [0, 0, 0, 0, 0]
        nsd = [0, 0, 0, 0, 0]
        for trial_index in range(seed_num):
            seed = build_case_seed(conf.seed, idx - 1, trial_index, seed_num)
            print("     seed:", seed)
            set_seed(seed, deterministic=conf.deterministic)

            result = diffusion.sample(
                shape_image = real_image.size(),
                shape_mask = real_mask_sdf.size(),
                device=device,
                image=real_image,
                sampler=sampler_cfg.name,
                ddim_steps=sampler_cfg.ddim_steps,
                eta=sampler_cfg.eta,
                guidance_mode=guidance_cfg.mode,
                guidance_cfg=guidance_cfg,
            )

            gen_mask = result[:,1:(result.size()[1]),:,:,:]
            gen_mask_de_sdf = (gen_mask < 0.0).to(dtype=gen_mask.dtype)

            if save_outputs:
                real_img_to_save = tio.ScalarImage(tensor=real_image.squeeze(0).cpu(), channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Image'), exist_ok=True)
                real_img_to_save.save(os.path.join(vis_dir_name, 'Image', f"{gt_name}-image-real.nii.gz"))
            
            for i in range(gen_mask.size()[1]):
                gen_mask_i = gen_mask[:, i:i + 1, :, :, :]
                gen_mask_i_cpu = gen_mask_i.cpu()
                gen_mask_i_de_sdf = gen_mask_de_sdf[:, i:i + 1, :, :, :].cpu()
                
                if save_outputs:
                    gen_mask_sdf_to_save = tio.LabelMap(tensor=gen_mask_i_cpu.squeeze(1), channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                    gen_mask_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-sdf-{i+1}-gen.nii.gz"))
                    
                    gen_mask_de_sdf_to_save = tio.LabelMap(tensor=gen_mask_i_de_sdf.squeeze(1), channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                    gen_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-de-sdf-{i+1}-gen.nii.gz"))

                    real_mask_sdf_to_save = tio.LabelMap(tensor=real_mask_sdf[:,i,:,:,:], channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                    real_mask_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-sdf-{i+1}-real.nii.gz"))
                    
                    real_mask_de_sdf_to_save = tio.LabelMap(tensor=real_mask[:,i,:,:,:], channels_last=False, affine=affine)
                    os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                    real_mask_de_sdf_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-de-sdf-{i+1}-real.nii.gz"))

                real_mask_i = real_mask[:, i:i + 1, :, :, :]
                Dice = get_dice(real_mask_i.numpy(), gen_mask_i_de_sdf.numpy())
                print(f"        {i+1}_dice:", Dice)
                dice[i] += Dice
            
            background_mask = (gen_mask_de_sdf.sum(dim=1, keepdim=True) == 0).to(dtype=gen_mask_de_sdf.dtype)
            gen_mask_togather = torch.cat((background_mask, gen_mask_de_sdf), dim=1)
            gen_mask_togather = torch.argmax(gen_mask_togather, dim=1)
            if save_outputs:
                gen_mask_togather_to_save = tio.LabelMap(tensor=gen_mask_togather.cpu().int(), channels_last=False, affine=affine)
                os.makedirs(os.path.join(vis_dir_name, 'Label'), exist_ok=True)
                gen_mask_togather_to_save.save(os.path.join(vis_dir_name, 'Label', f"{gt_name}-{seed}-label-together-gen.nii.gz"))

            get_nsd = NSDMetric(n_classes=6)
            real_mask_togather = real_mask
            background_mask = (real_mask_togather.sum(dim=1, keepdim=True) == 0).to(dtype=real_mask_togather.dtype)
            real_mask_togather_ = torch.cat((background_mask, real_mask_togather), dim=1)
            real_mask_togather_ = torch.argmax(real_mask_togather_, dim=1)
            nnsd = get_nsd(inputs=gen_mask_togather.long().cpu(), target=real_mask_togather_.long().cpu())
            for i in range(0, 5):
                nsd[i] += nnsd[i]
            print(f"        {nnsd}")

        dice_avg = [item / seed_num for item in dice]
        nsd_avg = [item / seed_num for item in nsd]
        print("     average:")
        print(f"        dice:", dice_avg)
        print(f"        nsd:", nsd_avg)
        for i in range(5):
            dice_total[i] += dice_avg[i]
            nsd_total[i] += nsd_avg[i]

    dice_total_avg = [item / idx for item in dice_total]
    nsd_total_avg = [item / idx for item in nsd_total]
    print("total average:")
            
    print(f"    dice:", dice_total_avg)
    print(f"    nsd:", nsd_total_avg)


if __name__ == "__main__":
    
    main()
