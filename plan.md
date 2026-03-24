# Plan: Add Reconstruction-Guided DPS Guidance to DiffAtlas

## 1. Goal

Implement the idea from `New-idea.md` and `plan_note_AboutLoss.md` as an **inference-time improvement** for DiffAtlas:

- keep the current training pipeline unchanged for the first iteration
- replace or augment the current hard image replacement guidance with a **gradient-based reconstruction guidance**
- use a **DDIM sampler** instead of the current full DDPM loop
- compute guidance from the predicted clean image `I0_hat`
- use a **modality-robust guidance loss** based on:
  - LNCC
  - 3D edge-magnitude loss

More precisely, this first phase should be framed as:

- **DPS-style reconstruction guidance**
- not strict Classifier Guidance in the academic sense
- with **MCG deferred to a later phase**

That distinction matters because this implementation will use reconstruction losses and latent-state gradients, not an external trained classifier `p(y | x_t)`.

The target is better segmentation quality, especially for:

- cross-modality transfer
- small structures
- anatomically ambiguous cases where hard replacement is too crude

## 2. Current Baseline In This Repo

The current inference path is:

1. initialize a random 6-channel tensor `[I_t, S_t]`
2. at every reverse step, overwrite channel `0` with the real noisy image
3. run one DDPM reverse step
4. keep only the generated SDF channels `1:`
5. threshold at `0`

The relevant code paths are:

- `ddpm/diffusion.py`
  - `GaussianDiffusion_Nolatent._apply_guidance()`
  - `GaussianDiffusion_Nolatent.p_sample_loop()`
- `test/inference.py`
  - calls `diffusion.p_sample_loop(...)`

This is a hard imputation strategy. It works, but it does not explicitly optimize the latent state to make the predicted clean image consistent with the observed input image.

## 3. Proposed Improvement

### Core idea

At timestep `t`, instead of only replacing the image channel, we:

1. predict noise `eps_theta(x_t, t)`
2. recover the predicted clean state `x0_hat`
3. extract the clean image channel `I0_hat`
4. compute a differentiable guidance loss between `I0_hat` and the observed image `I_input`
5. backpropagate this loss to the current latent state `x_t`
6. modify the reverse step using that gradient

This is an inverse-problem style posterior correction for the current joint diffusion prior.

Terminology to use in the code and write-up:

- first implementation: `DPS`
- future extension: `MCG`
- avoid calling the implementation `CG` in comments, docs, or config descriptions except when explicitly comparing against the original inspiration

### Why it fits this codebase

This repo is already ideal for this idea because:

- the model predicts noise over the **joint** state `[image, SDF mask]`
- the diffusion object already exposes `predict_start_from_noise()`
- the current guidance is already inference-only
- checkpoints already contain the learned prior, so we can reuse pretrained models

## 4. Implementation Strategy

Use a staged rollout:

### Stage A: Add a new sampler without changing training

This is the main implementation target.

- add DDIM sampling
- add reconstruction-guided gradient correction
- keep existing DDPM hard-replacement sampler as baseline
- add config switches to compare:
  - `none`
  - `replace`
  - `dps`
  - `hybrid`

### Stage B: Add ablations and tuning support

- LNCC-only
- Edge-magnitude-only
- LNCC + Edge
- mask-only gradient application vs full-state gradient application
- fixed gamma vs scheduled gamma

### Stage C: Optional follow-up if Stage A helps

- optional fine-tuning with a sampler-aware objective
- optional channel-aware clipping improvements
- optional modality-specific guidance schedules
- optional MCG-style manifold correction to prevent strong-guidance off-manifold drift

Stage C should not be attempted before Stage A is stable.

## 5. File-Level Change Plan

### 5.1 `ddpm/diffusion.py`

This is the main file to extend.

Add:

- DDIM timestep schedule helper
- DDIM step helper
- a way to predict `x0_hat` from `(x_t, t)`
- a DPS-guided sampling loop
- a sampler dispatcher that can choose between:
  - current DDPM replacement
  - DDIM without guidance
  - DDIM with DPS guidance
  - hybrid replacement + DPS

Do not break current checkpoint loading.

Important constraint:

- current inference loads old checkpoints with `diffusion.load_state_dict(weights_dict)` using strict loading
- therefore, do **not** add new persistent parameters or persistent registered buffers unless they are guaranteed to exist in old checkpoints

Safe options:

- compute DDIM schedules on the fly
- store sampler-only tensors as local variables
- if buffers are needed, register them with `persistent=False`

### 5.2 New file: `ddpm/guidance.py`

Create a small module for guidance losses and schedules.

Recommended contents:

- `LNCCLoss3D`
- `EdgeLoss3D`
- `build_guidance_scale()`
- optional `channelwise_clamp_x0()`

This module should also document two non-obvious numerical constraints from `plan_note_AboutLoss.md`:

- LNCC variance terms must be clamped before division
- edge loss should compare **gradient magnitude**, not signed directional gradients

This keeps `ddpm/diffusion.py` from becoming unmanageably large.

### 5.3 `test/inference.py`

Extend config handling and call the new sampler interface.

Add:

- sampler selection
- guidance configuration
- optional debug logging:
  - loss value per step
  - gradient norm
  - gamma at each step

This file should remain the top-level experiment driver.

### 5.4 `test/confs/infer.yaml`

Add new config groups or new flat keys for:

- sampler type
- DDIM step count
- guidance mode
- LNCC and edge weights
- gamma schedule
- gradient normalization/clipping
- whether to apply guidance to:
  - full state
  - mask channels only

### 5.5 `testing_scripts/*.sh`

Add one or more dedicated scripts for new experiments, for example:

- `test_MMWHSMRI_all_dps.sh`
- `test_MMWHSCT_testing_set_dps.sh`

Do not replace the old scripts yet. Keep a clean baseline.

## 6. Sampler Design

## 6.1 Add a unified inference API

The current code only exposes:

```python
result = diffusion.p_sample_loop(
    shape_image=real_image.size(),
    shape_mask=real_mask_sdf.size(),
    device=device,
    image=real_image,
)
```

Replace this with a dispatcher-style API:

```python
result = diffusion.sample(
    shape_image=real_image.size(),
    shape_mask=real_mask_sdf.size(),
    device=device,
    image=real_image,
    sampler=conf.sampler.name,
    ddim_steps=conf.sampler.ddim_steps,
    guidance_mode=conf.guidance.mode,
    guidance_cfg=conf.guidance,
)
```

Recommended behavior:

- `sampler=ddpm`, `guidance_mode=replace` reproduces the current method
- `sampler=ddim`, `guidance_mode=none` is a clean speed baseline
- `sampler=ddim`, `guidance_mode=dps` is the new method
- `sampler=ddim`, `guidance_mode=hybrid` runs both hard image replacement and DPS

## 6.2 Keep the old code path intact

Do not delete the current implementation immediately.

Instead:

- rename current `p_sample_loop()` to something like `p_sample_loop_ddpm_replace()`
- keep it available for baseline reproduction
- implement the new code alongside it

This is important for fair comparison.

## 7. DDIM Implementation Plan

The current code only implements a stochastic DDPM reverse step.

We need a deterministic or low-noise DDIM step because:

- guidance requires backward passes
- 300 reverse steps with backward on every step will be too slow
- DDIM with 20-50 steps is the practical path

### 7.1 Add DDIM timestep generation

Example helper:

```python
def make_ddim_timesteps(num_ddim_steps: int, num_ddpm_steps: int) -> torch.Tensor:
    if num_ddim_steps >= num_ddpm_steps:
        return torch.arange(num_ddpm_steps - 1, -1, -1)
    step_ratio = num_ddpm_steps / num_ddim_steps
    steps = torch.arange(num_ddim_steps, dtype=torch.float32) * step_ratio
    steps = torch.round(steps).long().clamp(max=num_ddpm_steps - 1)
    steps = torch.flip(steps.unique(sorted=True), dims=[0])
    return steps
```

### 7.2 Add a DDIM step helper

Use the existing `predict_start_from_noise()` logic and the standard DDIM update:

```python
def ddim_step(self, x_t, t, t_prev, eps, eta=0.0):
    alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
    alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t.shape)

    x0_hat = self.predict_start_from_noise(x_t, t=t, noise=eps)

    sigma = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t))
        * torch.sqrt(1 - alpha_bar_t / alpha_bar_prev)
    )

    noise = torch.randn_like(x_t) if eta > 0 else torch.zeros_like(x_t)
    dir_xt = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps

    x_prev = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma * noise
    return x_prev, x0_hat
```

Notes:

- start with `eta=0.0`
- deterministic DDIM is easier to debug
- only later test `eta > 0`

## 8. Reconstruction Guidance Design

## 8.1 Predict `x0_hat` using existing code

The repo already has:

```python
def predict_start_from_noise(self, x_t, t, noise):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )
```

Use this directly instead of re-deriving image-only formulas. Then extract:

```python
x0_hat = self.predict_start_from_noise(x_t, t=t, noise=eps_theta)
I0_hat = x0_hat[:, :1]
S0_hat = x0_hat[:, 1:]
```

This is cleaner and reduces implementation error.

## 8.2 Use a modality-robust image consistency loss

The proposed guidance loss should compare the **predicted clean image** to the **observed image**, not the noisy image.

Recommended loss:

```python
L_guide = lambda_lncc * L_lncc(I0_hat, I_input) + lambda_edge * L_edge(I0_hat, I_input)
```

This should live in `ddpm/guidance.py`.

Important interpretation from `plan_note_AboutLoss.md`:

- `L_lncc` is the primary structural guidance term
- `L_edge` should be defined on **edge magnitude**, not signed gradients
- this makes the edge term safer for cross-modality settings such as CT-to-MRI, where intensity polarity can invert across anatomical boundaries

## 8.3 LNCC implementation

Use a 3D local normalized cross-correlation with convolutional window sums.

The first plan version used a correct but less stable and less efficient formulation. The revised implementation should follow the safer version from `plan_note_AboutLoss.md`:

- register the all-ones convolution kernel as a buffer
- use the algebraically simplified covariance/variance formulas
- clamp variance terms before division

Recommended implementation:

```python
class LNCCLoss3D(nn.Module):
    def __init__(self, win=9, eps=1e-5):
        super().__init__()
        self.win = win
        self.eps = eps
        filt = torch.ones(1, 1, win, win, win)
        self.register_buffer("filt", filt)

    def forward(self, x, y):
        # x, y: [B, 1, D, H, W]
        filt = self.filt.to(dtype=x.dtype)
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
        x_var = x2_sum - (x_sum * x_sum) / win_size
        y_var = y2_sum - (y_sum * y_sum) / win_size

        x_var = torch.clamp(x_var, min=self.eps)
        y_var = torch.clamp(y_var, min=self.eps)

        cc = (cross * cross) / (x_var * y_var)
        return -cc.mean()
```

Implementation notes:

- the variance clamp is mandatory, not optional
- relying only on `+ eps` in the denominator is not enough
- large uniform regions in medical scans can otherwise produce tiny negative variances from float32 cancellation and trigger unstable gradients or NaNs
- keep this module in `float32` in the first implementation

## 8.4 3D edge loss implementation

The first plan version proposed direct signed finite-difference matching. That is too fragile for cross-modality guidance.

Revised design:

- still start with finite differences, because they are easy to verify and fully differentiable
- but compare **gradient magnitude**, not gradient direction
- pad each directional difference back to the original tensor size before magnitude fusion

Recommended implementation:

```python
def gradient_3d(x):
    dz = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    dz = F.pad(dz, (0, 0, 0, 0, 0, 1))
    dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
    dx = F.pad(dx, (0, 1, 0, 0, 0, 0))
    return dx, dy, dz


class EdgeLoss3D(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        x_dx, x_dy, x_dz = gradient_3d(x)
        y_dx, y_dy, y_dz = gradient_3d(y)
        x_mag = torch.sqrt(x_dx ** 2 + x_dy ** 2 + x_dz ** 2 + self.eps)
        y_mag = torch.sqrt(y_dx ** 2 + y_dy ** 2 + y_dz ** 2 + self.eps)
        return F.l1_loss(x_mag, y_mag)
```

Why this version is preferable:

- it is still simple and fully differentiable
- it is robust to intensity polarity flips across modalities
- it aligns anatomical boundary strength rather than raw signed direction

Do not make Sobel the default first implementation. Only add it later as an ablation if needed.

## 9. Gradient Guidance Step

## 9.1 Remove `@torch.inference_mode()` from the new DPS path

This is critical.

The current `p_sample()` and `p_sample_loop()` use `@torch.inference_mode()`. That will disable gradients and make DPS impossible.

Do not remove it globally from the old path.

Instead:

- keep the old no-grad sampler
- add a separate DPS sampler without `inference_mode`

Example:

```python
@torch.inference_mode()
def p_sample_loop_ddpm_replace(...):
    ...

def sample_loop_ddim_dps(...):
    ...
```

## 9.2 Guidance on full state vs mask-only

Your sketch proposes differentiating with respect to the whole current state `x_t`.

That is correct mathematically, but for the first implementation I recommend supporting two modes:

- `apply_to = "full"`
- `apply_to = "mask_only"`

Reason:

- `full` is the pure formulation
- `mask_only` is often more stable for segmentation because the image channel is only an auxiliary carrier, while the output of interest is the mask/SDF channel

Implementation:

```python
grad_x = torch.autograd.grad(loss, x_t, retain_graph=False, create_graph=False)[0]

if guidance_cfg.apply_to == "mask_only":
    grad_x = grad_x.clone()
    grad_x[:, :1] = 0
```

Start experiments with `mask_only` as the default, then compare.

## 9.3 Normalize and clip the gradient

This should not be optional in the first version.

Use:

```python
grad_flat = grad_x.reshape(grad_x.shape[0], -1)
grad_norm = torch.linalg.norm(grad_flat, dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-8)
grad_unit = grad_x / grad_norm
grad_unit = torch.clamp(grad_unit, min=-guidance_cfg.grad_clip, max=guidance_cfg.grad_clip)
```

This is much safer than injecting raw gradients.

## 9.4 Modify the score / noise prediction

The simplest adaptation to the current implementation is:

```python
eps_guided = eps_theta - torch.sqrt(1 - alpha_bar_t) * gamma_t * grad_unit
```

Then use `eps_guided` in the DDIM update.

This matches the direction in your sketch and preserves the structure of the current denoising code.

## 10. Recommended New Sampling Loop

Below is the target algorithm adapted to this repo.

```python
def sample_loop_ddim_dps(
    self,
    shape_image,
    shape_mask,
    device,
    image,
    ddim_steps=50,
    eta=0.0,
    guidance_cfg=None,
):
    b = shape_image[0]
    x_t = torch.cat(
        [
            torch.randn(shape_image, device=device),
            torch.randn(shape_mask, device=device),
        ],
        dim=1,
    )

    ddim_ts = make_ddim_timesteps(ddim_steps, self.num_timesteps).to(device)

    lncc_loss = LNCCLoss3D(win=guidance_cfg.lncc_win)
    edge_loss = EdgeLoss3D()

    self.denoise_fn.eval()

    for i, t_scalar in enumerate(ddim_ts):
        t = torch.full((b,), int(t_scalar.item()), device=device, dtype=torch.long)
        t_prev_scalar = ddim_ts[i + 1] if i + 1 < len(ddim_ts) else torch.tensor(0, device=device)
        t_prev = torch.full((b,), int(t_prev_scalar.item()), device=device, dtype=torch.long)

        x_t = x_t.detach().requires_grad_(True)

        if isinstance(self.denoise_fn, torch.nn.DataParallel):
            eps_theta = self.denoise_fn.module.forward_with_cond_scale(x_t, t, cond=None, cond_scale=1.0)
        else:
            eps_theta = self.denoise_fn.forward_with_cond_scale(x_t, t, cond=None, cond_scale=1.0)

        x0_hat = self.predict_start_from_noise(x_t, t=t, noise=eps_theta)
        I0_hat = x0_hat[:, :1]

        # Optional stability clamp
        I0_hat = I0_hat.clamp(-1.0, 1.0)

        loss = 0.0
        if guidance_cfg.lambda_lncc > 0:
            loss = loss + guidance_cfg.lambda_lncc * lncc_loss(I0_hat, image)
        if guidance_cfg.lambda_edge > 0:
            loss = loss + guidance_cfg.lambda_edge * edge_loss(I0_hat, image)

        grad_x = torch.autograd.grad(loss, x_t)[0]

        if guidance_cfg.apply_to == "mask_only":
            grad_x = grad_x.clone()
            grad_x[:, :1] = 0

        grad_flat = grad_x.reshape(b, -1)
        grad_norm = torch.linalg.norm(grad_flat, dim=1).view(b, 1, 1, 1, 1).clamp_min(1e-8)
        grad_unit = grad_x / grad_norm
        grad_unit = torch.clamp(grad_unit, -guidance_cfg.grad_clip, guidance_cfg.grad_clip)

        gamma_t = build_guidance_scale(
            t=t,
            total_steps=self.num_timesteps,
            base_scale=guidance_cfg.gamma,
            mode=guidance_cfg.gamma_schedule,
        ).view(b, 1, 1, 1, 1)

        alpha_bar_t = extract(self.alphas_cumprod, t, x_t.shape)
        eps_guided = eps_theta - torch.sqrt(1.0 - alpha_bar_t) * gamma_t * grad_unit

        x_prev, _ = self.ddim_step(x_t.detach(), t, t_prev, eps_guided.detach(), eta=eta)

        if guidance_cfg.mode == "hybrid":
            noisy_img = self.q_sample(x_start=image, t=t_prev)
            x_prev[:, :1] = noisy_img[:, :1]

        x_t = x_prev

    return x_t.detach()
```

## 11. Config Design

The current `test/confs/infer.yaml` is too small for this feature. Expand it.

Recommended structure:

```yaml
diffusion_img_size: ???
diffusion_depth_size: ???
diffusion_num_channels: ???
dim_mults: [1, 2, 4, 8]
denoising_fn: Unet3D
timesteps: ???
loss_type: l1

model_path: ???
model_num: ???
weight_key: ema
dir_name: ???

dataset: ???
mode: ???
data_type: ???
root_dir: ???
device: null
seed: 1
seed_num: 1
deterministic: true

sampler:
  name: ddim
  ddim_steps: 50
  eta: 0.0

guidance:
  mode: dps        # none | replace | dps | hybrid
  gamma: 1.0
  gamma_schedule: mid
  lambda_lncc: 1.0
  lambda_edge: 0.1
  lncc_win: 9
  grad_clip: 1.0
  apply_to: mask_only
  log_every_step: false
```

Backward compatibility plan:

- if `guidance.mode` is absent, treat it as the old `replace`
- keep `use_guide` temporarily for old scripts, but deprecate it

## 12. Recommended Code Changes

## 12.1 Add a sampler dispatcher in `ddpm/diffusion.py`

```python
def sample(
    self,
    shape_image,
    shape_mask,
    device=None,
    image=None,
    sampler="ddpm",
    ddim_steps=50,
    eta=0.0,
    guidance_mode="replace",
    guidance_cfg=None,
):
    if sampler == "ddpm" and guidance_mode == "replace":
        return self.p_sample_loop_ddpm_replace(
            shape_image=shape_image,
            shape_mask=shape_mask,
            device=device,
            image=image,
        )

    if sampler == "ddim":
        return self.sample_loop_ddim(
            shape_image=shape_image,
            shape_mask=shape_mask,
            device=device,
            image=image,
            ddim_steps=ddim_steps,
            eta=eta,
            guidance_mode=guidance_mode,
            guidance_cfg=guidance_cfg,
        )

    raise ValueError(f"Unsupported sampler={sampler}, guidance_mode={guidance_mode}")
```

## 12.2 Update `test/inference.py`

Replace:

```python
sample_fn = diffusion.p_sample_loop
result = sample_fn(
    shape_image=real_image.size(),
    shape_mask=real_mask_sdf.size(),
    device=device,
    image=real_image,
)
```

With:

```python
result = diffusion.sample(
    shape_image=real_image.size(),
    shape_mask=real_mask_sdf.size(),
    device=device,
    image=real_image,
    sampler=conf.sampler.name,
    ddim_steps=conf.sampler.ddim_steps,
    eta=conf.sampler.eta,
    guidance_mode=conf.guidance.mode,
    guidance_cfg=conf.guidance,
)
```

## 12.3 Add a clean guidance module

Example file layout:

```python
# ddpm/guidance.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LNCCLoss3D(nn.Module):
    ...


class EdgeLoss3D(nn.Module):
    ...


def build_guidance_scale(t, total_steps, base_scale, mode="mid"):
    ratio = t.float() / max(total_steps - 1, 1)

    if mode == "constant":
        scale = torch.full_like(ratio, fill_value=base_scale)
    elif mode == "mid":
        # strong in middle timesteps, weaker at both ends
        scale = base_scale * torch.sin(torch.pi * ratio)
    elif mode == "late":
        scale = base_scale * (1.0 - ratio)
    else:
        raise ValueError(f"Unknown gamma schedule: {mode}")

    return torch.clamp(scale, min=0.0)
```

Use comments and docstrings that describe this as:

- reconstruction-guided DPS
- with MCG reserved for a later extension

## 13. Important Engineering Details

## 13.1 Keep old checkpoints loadable

This is non-negotiable.

Avoid:

- adding persistent buffers to `GaussianDiffusion_Nolatent`
- adding new model parameters to the diffusion object

Otherwise:

- `diffusion.load_state_dict(weights_dict)` will fail on old checkpoints

Recommended solution:

- all new sampler state is temporary
- all new config is external
- all new loss modules are instantiated only at inference time

## 13.2 Channel-aware `x0` clamping

This repo currently clamps the reconstructed state to `[-1, 1]` inside `p_mean_variance()`.

That is acceptable for the image channel, but suboptimal for SDF channels because their natural range is closer to `[-0.2, 0.2]`.

For the new DPS path, add an optional helper:

```python
def clamp_joint_x0(x0_hat, image_clip=1.0, sdf_clip=0.2):
    x0_hat = x0_hat.clone()
    x0_hat[:, :1] = x0_hat[:, :1].clamp(-image_clip, image_clip)
    x0_hat[:, 1:] = x0_hat[:, 1:].clamp(-sdf_clip, sdf_clip)
    return x0_hat
```

This should improve stability of both:

- the image guidance loss
- the final SDF prediction

## 13.3 Do not compute guidance under autocast first

For the first implementation, keep the DPS path in full `float32`.

Reason:

- LNCC denominators can become numerically delicate
- edge magnitude uses `sqrt`, which is also a place where numerical sloppiness can surface
- mixed precision complicates debugging

After the method works, add optional AMP for inference.

## 13.4 Batch size

Start with:

- inference batch size = 1

The current code already effectively assumes this in metric logging and file writing. DPS with backprop will also be memory-heavy.

## 14. Recommended Experiment Matrix

Run these in order:

### Phase 1: Sanity checks

1. DDIM 50-step, no guidance
2. DDIM 50-step, hard replacement only
3. DDIM 50-step, DPS with LNCC only
4. DDIM 50-step, DPS with LNCC + edge-magnitude

Target datasets:

- MMWHS CT full -> CT testing set

Reason:

- same-modality is the least confounded place to debug

### Phase 2: Cross-modality benchmark

1. CT-all model -> MRI-all test
2. MRI-all model -> CT-all test

Compare:

- baseline replace
- DPS LNCC only
- DPS LNCC + edge-magnitude
- hybrid replace + DPS

### Phase 3: Guidance schedule ablation

Compare:

- `gamma_schedule=constant`
- `gamma_schedule=mid`
- `gamma_schedule=late`

### Phase 4: Gradient application ablation

Compare:

- `apply_to=full`
- `apply_to=mask_only`

## 15. Recommended Metrics And Logging

Keep the current:

- per-class Dice
- per-class NSD

Add DPS-specific debug logs:

- average guidance loss per step
- gradient norm per step
- gamma per step
- max absolute value in `I0_hat`
- max absolute value in `S0_hat`
- min variance value seen inside LNCC before clamping, if debug mode is enabled

If a run fails, these logs will usually explain why.

## 16. Likely Failure Modes And Mitigations

### Failure mode 1: gradients explode

Symptoms:

- masks become all zeros or all ones
- `nan` in loss or outputs

Mitigation:

- normalize gradients
- clamp gradients
- reduce `gamma`
- use `mask_only`
- disable edge loss temporarily

### Failure mode 2: no visible effect

Symptoms:

- output indistinguishable from unguided DDIM

Mitigation:

- increase `gamma`
- start with LNCC only
- verify `loss.backward` path is active
- log `grad_norm`

### Failure mode 3: over-constraining same-modality cases

Symptoms:

- textures align, but segmentation becomes noisy

Mitigation:

- reduce `lambda_edge`
- reduce late-step guidance
- turn off guidance in the last 10-20% of steps

### Failure mode 4: old checkpoints stop loading

Cause:

- new state dict keys in diffusion object

Mitigation:

- keep sampler state non-persistent
- avoid new registered parameters/buffers

## 17. First Implementation Order

This is the exact order I would implement:

1. Add `ddpm/guidance.py` with:
   - `LNCCLoss3D`
   - `EdgeLoss3D`
   - `build_guidance_scale`
2. Add DDIM timestep helper and `ddim_step()` to `GaussianDiffusion_Nolatent`
3. Add `sample()` dispatcher
4. Move the current DDPM hard-replacement loop into `p_sample_loop_ddpm_replace()`
5. Add `sample_loop_ddim()` with:
   - `none`
   - `replace`
   - `dps`
   - `hybrid`
6. Extend `test/confs/infer.yaml`
7. Update `test/inference.py` to call the new dispatcher
8. Add one new testing shell script for same-modality debugging
9. Run one-case smoke test
10. Run a short cross-modality test

## 18. Minimal Smoke Test Before Full Evaluation

Before launching whole datasets, validate on one case:

```bash
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
  dir_name=debug_dps_ct \
  root_dir=./data/MMWHS/CT/testing_set \
  sampler.name=ddim \
  sampler.ddim_steps=50 \
  guidance.mode=dps \
  guidance.gamma=1.0 \
  guidance.lambda_lncc=1.0 \
  guidance.lambda_edge=0.1 \
  guidance.apply_to=mask_only
```

If this fails:

- inspect gradient norm logs
- rerun with `guidance.lambda_edge=0.0`
- rerun with `guidance.gamma=0.3`

## 19. Recommended Default Starting Point

For the first real experiment, use:

- sampler: `ddim`
- ddim steps: `50`
- eta: `0.0`
- guidance mode: `dps`
- lambda_lncc: `1.0`
- lambda_edge: `0.1`
- gamma: `0.5`
- gamma schedule: `mid`
- apply_to: `mask_only`
- grad clip: `1.0`

This is conservative and should be much more stable than starting with aggressive full-state guidance.

## 20. Final Recommendation

Do **not** start by rewriting the current pipeline end-to-end.

The right implementation path is:

1. preserve the current hard-replacement sampler as an exact baseline
2. add DDIM as a second sampler
3. add DPS guidance on top of DDIM
4. compare `replace` vs `dps` vs `hybrid`
5. only after those results are clear, consider training-time changes

The key reason is that your idea is fundamentally a **posterior-sampling improvement**, not a training-loss idea. This repo already has the prior. The correct first move is to improve inference, not retrain the model.

In naming and scope, the correct phrasing for this phase is:

- implement **DPS with LNCC + edge-magnitude guidance**
- evaluate its stability and gains
- only afterwards design **MCG** as the next-level safeguard against off-manifold behavior under strong guidance

## 21. Detailed TODO List

This section turns the implementation plan into an execution checklist. The intended workflow is:

1. finish Phase 0 before editing sampler logic
2. finish Phase 1 before touching `test/inference.py`
3. finish Phase 2 before running whole-dataset experiments
4. finish Phase 3 before claiming the DPS method is better than baseline
5. treat Phase 4 as a follow-up, not part of the first delivery

### Phase 0: Design Freeze And Prep

- [x] Re-read [`New-idea.md`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/New-idea.md), [`plan_note_AboutLoss.md`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/plan_note_AboutLoss.md), and [`plan.md`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/plan.md) together and confirm the terminology is consistent:
  - DPS for the first version
  - MCG deferred
  - edge-magnitude loss, not signed gradient loss
- [x] Confirm that the first implementation will be inference-only.
- [x] Confirm that old checkpoints must continue to load with strict `load_state_dict`.
- [x] Decide the default guidance mode naming that will appear in config and scripts:
  - `none`
  - `replace`
  - `dps`
  - `hybrid`
- [x] Decide the first default experiment target:
  - recommended: MMWHS CT full -> CT testing set
- [x] Decide the first default hyperparameters:
  - `ddim_steps=50`
  - `eta=0.0`
  - `guidance.mode=dps`
  - `guidance.gamma=0.5`
  - `guidance.lambda_lncc=1.0`
  - `guidance.lambda_edge=0.1`
  - `guidance.apply_to=mask_only`
  - `guidance.grad_clip=1.0`
- [x] Define the minimum success criterion for the first smoke test:
  - no NaNs
  - no crash
  - outputs saved successfully
  - nontrivial mask prediction produced

### Phase 1: Core Sampler Refactor

- [x] Open [`ddpm/diffusion.py`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py) and isolate the current inference-related functions:
  - `p_mean_variance`
  - `p_sample`
  - `_apply_guidance`
  - `p_sample_loop`
- [x] Rename the current `p_sample_loop()` into an explicit baseline function:
  - recommended name: `p_sample_loop_ddpm_replace`
- [x] Add a compatibility wrapper if needed so older internal call sites do not break during refactor.
- [x] Add a `sample()` dispatcher method to `GaussianDiffusion_Nolatent`.
- [x] Implement DDIM timestep generation helper.
- [x] Implement DDIM one-step update helper.
- [x] Verify the DDIM helper does not introduce any persistent model state.
- [x] Verify the DDIM helper uses only existing diffusion buffers already present in checkpoints.
- [x] Add a `sample_loop_ddim()` entry point that can branch on:
  - `guidance_mode=none`
  - `guidance_mode=replace`
  - `guidance_mode=dps`
  - `guidance_mode=hybrid`
- [x] Keep the old DDPM replace path runnable for baseline comparison.

Completion gate for Phase 1:

- [x] Old baseline inference can still run without changing checkpoint files.

### Phase 2: Guidance Loss Module

- [x] Create [`ddpm/guidance.py`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/guidance.py).
- [x] Add imports:
  - `torch`
  - `torch.nn`
  - `torch.nn.functional`
- [x] Implement `LNCCLoss3D`.
- [x] Register the LNCC all-ones convolution kernel as a buffer.
- [x] Ensure LNCC uses the simplified covariance/variance formulas.
- [x] Clamp LNCC variance terms before division.
- [x] Keep LNCC in `float32` for the first implementation.
- [x] Add comments documenting why the clamp is required.
- [x] Implement `gradient_3d()` with shape-restoring padding on all three axes.
- [x] Implement `EdgeLoss3D` using gradient magnitude.
- [x] Add `eps` inside the magnitude square root.
- [x] Add comments documenting why signed gradient matching is unsafe across modalities.
- [x] Implement `build_guidance_scale()`.
- [x] Support at least:
  - `constant`
  - `mid`
  - `late`
- [x] Optionally implement `clamp_joint_x0()`.
- [x] Verify this helper clamps:
  - image channel to `[-1, 1]`
  - SDF channels to `[-0.2, 0.2]`

Completion gate for Phase 2:

- [x] Loss module imports cleanly.
- [x] Forward pass runs on a dummy tensor.
- [x] No NaNs appear on a trivial synthetic test.

### Phase 3: DPS Sampling Logic

- [x] Add a separate DPS sampling loop without `@torch.inference_mode()`.
- [x] Make sure the old no-grad path stays intact.
- [x] Inside the DPS loop, `detach()` and re-enable gradients on `x_t` every iteration.
- [x] Reuse `predict_start_from_noise()` to obtain `x0_hat`.
- [x] Extract `I0_hat` from `x0_hat[:, :1]`.
- [x] Optionally clamp `I0_hat` before computing losses.
- [x] Instantiate LNCC and edge loss modules once per sample loop, not once per timestep.
- [x] Compute the DPS guidance loss:
  - LNCC term
  - edge-magnitude term
- [x] Compute gradient with respect to the full latent state `x_t`.
- [x] Support `apply_to=mask_only`.
- [x] Support `apply_to=full`.
- [x] Normalize the gradient by batch-wise norm.
- [x] Clamp normalized gradient by `guidance.grad_clip`.
- [x] Compute `gamma_t` from the configured schedule.
- [x] Modify the predicted noise using the guidance gradient.
- [x] Feed guided noise into DDIM update.
- [x] For `guidance_mode=hybrid`, also apply noisy-image replacement after the DDIM step.
- [x] Ensure the final returned tensor shape matches the current inference expectation:
  - 1 image channel
  - 5 SDF channels

Completion gate for Phase 3:

- [x] One-case guided sampling runs end to end.
- [x] No NaNs in `loss`, `grad_norm`, or output tensor.
- [x] Saved masks are not trivially empty for every class.

### Phase 4: Inference Driver Integration

- [x] Open [`test/inference.py`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py).
- [x] Replace direct calls to `diffusion.p_sample_loop(...)` with the new `diffusion.sample(...)`.
- [x] Thread through sampler config values:
  - `sampler.name`
  - `sampler.ddim_steps`
  - `sampler.eta`
- [x] Thread through guidance config values:
  - `guidance.mode`
  - `guidance.gamma`
  - `guidance.gamma_schedule`
  - `guidance.lambda_lncc`
  - `guidance.lambda_edge`
  - `guidance.lncc_win`
  - `guidance.grad_clip`
  - `guidance.apply_to`
- [x] Preserve current checkpoint loading logic.
- [x] Preserve current metric computation logic.
- [x] Preserve current NIfTI export logic.
- [x] Add optional DPS debug logging:
  - per-step or summarized guidance loss
  - gradient norm
  - gamma schedule values
  - `I0_hat` and `S0_hat` max magnitude
- [x] Make sure inference still works when guidance is disabled.

Completion gate for Phase 4:

- [x] Baseline replace mode and DPS mode both run from the same inference entry point.

### Phase 5: Config And Script Surface

- [x] Update [`test/confs/infer.yaml`](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/confs/infer.yaml).
- [x] Add a `sampler` section.
- [x] Add a `guidance` section.
- [x] Keep backward compatibility behavior documented.
- [x] Decide whether to keep `use_guide` temporarily or fully replace it.
- [x] Add at least one dedicated script for same-modality DPS smoke testing.
- [x] Add at least one dedicated script for cross-modality DPS testing.
- [x] Make sure old testing scripts are left intact.

Completion gate for Phase 5:

- [x] There is a clean command-line path to run:
  - old baseline
  - DDIM no-guidance baseline
  - DPS
  - hybrid

### Phase 6: Numerical And Stability Checks

- [x] Run a one-case same-modality smoke test.
- [x] Check whether LNCC forward values are finite.
- [x] Check whether gradient norms are finite for all timesteps.
- [x] Check whether `I0_hat` stays in a plausible range.
- [x] Check whether SDF channels stay in a plausible range.
- [x] Check whether generated masks contain all-background collapse.
- [x] Check whether generated masks contain all-foreground collapse.
- [x] If instability occurs, test mitigations in this order:
  - lower `gamma`
  - `lambda_edge=0`
  - `apply_to=mask_only`
  - fewer DDIM steps for debugging only
  - stronger x0 clamping
- [x] Document the first stable hyperparameter combination in the plan or experiment notes.

Completion gate for Phase 6:

- [x] At least one stable DPS configuration exists for a single held-out case.

### Phase 7: Baseline Reproduction

- [x] Re-run the original baseline path on the same validation case(s).
- [x] Run DDIM with `guidance_mode=none`.
- [x] Run DDIM with `guidance_mode=replace`.
- [x] Verify that DDIM replace produces output quality in the expected range relative to DDPM replace.
- [ ] Record runtime differences:
  - DDPM replace
  - DDIM none
  - DDIM replace
  - DDIM DPS

Completion gate for Phase 7:

- [ ] There is a fair comparison table for speed and stability before claiming any gain.

### Phase 8: Same-Modality DPS Evaluation

- [x] Run MMWHS CT full -> CT testing set:
  - replace baseline
  - DPS LNCC only
  - DPS LNCC + edge-magnitude
  - hybrid
- [x] Compare per-class Dice.
- [x] Compare per-class NSD.
- [x] Check whether edge-magnitude improves small-structure behavior or only adds noise.
- [ ] Check whether `mask_only` is more stable than `full`.
- [ ] Check whether `gamma_schedule=mid` helps more than `constant`.

Completion gate for Phase 8:

- [x] Same-modality results show whether DPS is neutral, helpful, or harmful before moving to cross-modality conclusions.

### Phase 9: Cross-Modality DPS Evaluation

- [x] Run CT-all model -> MRI-all test.
- [ ] Run MRI-all model -> CT-all test.
- [ ] Compare:
  - replace baseline
  - DPS LNCC only
  - DPS LNCC + edge-magnitude
  - hybrid
- [ ] Inspect qualitative outputs for:
  - boundary fit
  - missing structures
  - hallucinated structures
  - noisy overlaps between classes
- [ ] Check whether the edge-magnitude term is more useful in cross-modality than in same-modality.
- [ ] Check whether hybrid outperforms pure DPS or pure replace.

Completion gate for Phase 9:

- [ ] Cross-modality results clearly identify the best guidance mode and default loss mix.

### Phase 10: Ablations And Hyperparameter Refinement

- [ ] Ablate `apply_to=full` vs `mask_only`.
- [ ] Ablate `lambda_edge`:
  - `0.0`
  - `0.05`
  - `0.1`
  - `0.2`
- [ ] Ablate `gamma`:
  - `0.3`
  - `0.5`
  - `1.0`
- [ ] Ablate `gamma_schedule`:
  - `constant`
  - `mid`
  - `late`
- [ ] Ablate DDIM steps:
  - `20`
  - `50`
  - `100`
- [ ] Optionally ablate x0 clamping on/off.
- [ ] Optionally compare finite-difference edge magnitude against a later Sobel variant.

Completion gate for Phase 10:

- [ ] A recommended default config is backed by ablation evidence rather than intuition.

### Phase 11: Result Consolidation

- [ ] Summarize best same-modality settings.
- [ ] Summarize best cross-modality settings.
- [ ] Summarize runtime overhead versus baseline.
- [ ] Summarize failure cases.
- [ ] Decide whether DPS is strong enough to become the new default inference mode.
- [ ] If yes, update the README and testing scripts later.
- [ ] If no, keep it as an experimental inference path only.

Completion gate for Phase 11:

- [ ] There is a clear go/no-go decision for merging DPS as a supported inference option.

### Phase 12: Future MCG Extension

This phase is explicitly out of scope for the first implementation, but should remain on the roadmap.

- [ ] Define what “off-manifold” means operationally for this repo.
- [ ] Decide whether MCG correction will act on:
  - latent state
  - predicted x0
  - denoised score
- [ ] Design the manifold constraint term.
- [ ] Decide whether MCG should be combined with DPS or replace it.
- [ ] Define the extra diagnostics needed to prove MCG is helping rather than merely dampening guidance.

Completion gate for Phase 12:

- [ ] MCG design is specified only after DPS is stable and benchmarked.

### Cross-Phase Checklist

- [x] Do not change training unless the inference-only path is proven insufficient.
- [x] Do not break old checkpoint compatibility.
- [x] Do not remove the current replace baseline.
- [x] Do not introduce persistent diffusion buffers that old checkpoints do not contain.
- [x] Do not claim “CG” in the strict academic sense for the first implementation.
- [x] Do not start MCG work before DPS evaluation is complete.
