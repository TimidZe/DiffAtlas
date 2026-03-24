import torch
import torch.nn as nn
import torch.nn.functional as F


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def gradient_3d(x):
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return dx, dy, dz


class LNCCLoss3D(nn.Module):
    def __init__(self, win=9, eps=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps
        self.register_buffer("filt", torch.ones(1, 1, win, win, win), persistent=False)
        self.last_stats = {}

    def forward(self, x, y):
        filt = self.filt.to(device=x.device, dtype=x.dtype)
        pad = self.win // 2

        x2 = x * x
        y2 = y * y
        xy = x * y

        x_sum = F.conv3d(x, filt, padding=pad)
        y_sum = F.conv3d(y, filt, padding=pad)
        x2_sum = F.conv3d(x2, filt, padding=pad)
        y2_sum = F.conv3d(y2, filt, padding=pad)
        xy_sum = F.conv3d(xy, filt, padding=pad)

        win_size = float(self.win ** 3)

        cross = xy_sum - (x_sum * y_sum) / win_size
        x_var_pre = x2_sum - (x_sum * x_sum) / win_size
        y_var_pre = y2_sum - (y_sum * y_sum) / win_size
        # Uniform regions can produce tiny negative variances from float32 cancellation.
        x_var = torch.clamp(x_var_pre, min=self.eps)
        y_var = torch.clamp(y_var_pre, min=self.eps)

        self.last_stats = {
            "x_var_min_pre_clamp": float(x_var_pre.min().detach().cpu()),
            "y_var_min_pre_clamp": float(y_var_pre.min().detach().cpu()),
            "x_var_min": float(x_var.min().detach().cpu()),
            "y_var_min": float(y_var.min().detach().cpu()),
        }

        cc = (cross * cross) / (x_var * y_var)
        return -cc.mean()


class EdgeLoss3D(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x_dx, x_dy, x_dz = gradient_3d(x)
        y_dx, y_dy, y_dz = gradient_3d(y)
        # Compare edge magnitude so modality-specific sign flips do not dominate the loss.
        x_mag = torch.sqrt(x_dx ** 2 + x_dy ** 2 + x_dz ** 2 + self.eps)
        y_mag = torch.sqrt(y_dx ** 2 + y_dy ** 2 + y_dz ** 2 + self.eps)
        return F.l1_loss(x_mag, y_mag)


def build_guidance_scale(t, total_steps, base_scale, mode="mid"):
    ratio = t.float() / max(total_steps - 1, 1)
    if mode == "constant":
        scale = torch.full_like(ratio, fill_value=base_scale)
    elif mode == "mid":
        scale = base_scale * torch.sin(torch.pi * ratio)
    elif mode == "late":
        scale = base_scale * (1.0 - ratio)
    else:
        raise ValueError(f"Unknown gamma schedule: {mode}")
    return torch.clamp(scale, min=0.0)


def clamp_joint_x0(x0_hat, image_clip=1.0, sdf_clip=0.2):
    image = x0_hat[:, :1].clamp(-image_clip, image_clip)
    sdf = x0_hat[:, 1:].clamp(-sdf_clip, sdf_clip)
    return torch.cat((image, sdf), dim=1)
