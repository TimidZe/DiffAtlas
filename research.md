# DiffAtlas Research Notes

## Scope

This report is based on a full local read-through of the repository at `/home/estar/TZNEW/DiffAtlas/DiffAtlas`, plus inspection of the datasets, checkpoints, Hydra outputs, and logs already present in the workspace. It reflects how this codebase actually works in this checkout, not just how the README describes it.

## Executive Summary

DiffAtlas is a 3D medical image segmentation project built around a joint image-mask diffusion model.

The core idea in code is:

1. Represent each training sample as a 6-channel 3D tensor:
   - 1 channel for the image volume
   - 5 channels for class-wise signed distance fields (SDFs) derived from the segmentation mask
2. Train a denoising diffusion model to predict noise on both the image channel and the 5 SDF channels.
3. At inference time, do not generate the image freely. Instead, repeatedly overwrite the image channel at every reverse-diffusion step with a noisy version of the real target image.
4. Let the model generate only the mask/SDF channels under that image guidance, then threshold the generated SDFs at `0` to recover binary class masks.

In practice, this makes the method much closer to image-guided joint denoising than to unconditional synthesis. The implementation is compact and highly specific:

- fixed to `64 x 64 x 64` volumes
- fixed to 5 foreground classes plus implicit background
- fixed to SDF-based mask representation
- fixed in all provided scripts to the custom `Unet3D`
- training supports an alternative MONAI `UNet`, but inference does not

## Repository Structure

The functional core of the repo is small:

- `train/train.py`
  - Hydra entry point for training
  - builds the model, wraps it in `nn.DataParallel`, builds the diffusion object, chooses the dataset, and launches training
- `test/inference.py`
  - Hydra entry point for inference and metric computation
  - loads a checkpoint, rebuilds the diffusion model, performs guided sampling, writes NIfTI outputs, and computes Dice/NSD
- `ddpm/diffusion.py`
  - main 3D diffusion model, noise schedule, sampling loop, and trainer
- `ddpm/unet.py`
  - alternative MONAI-style UNet implementation
- `Dataset/MMWHS_Dataset.py`
  - loader for MMWHS CT/MRI data using precomputed SDFs
- `Dataset/TS_Dataset.py`
  - loader for TotalSegmentator data with online SDF computation
- `sdf.py`
  - SDF construction routine
- `train/config/*`
  - Hydra configuration templates
- `test/confs/infer.yaml`
  - Hydra inference config template
- `traing_scripts/*.sh`
  - canned training commands
- `testing_scripts/*.sh`
  - canned inference commands

There are also local workspace artifacts:

- `Model/`
  - pretrained checkpoints and a newer manual single-GPU run
- `outputs/`
  - Hydra run directories
- `log_train/`, `log_inference/`
  - recorded logs

## What The Project Actually Segments

Although the datasets are named MMWHS and TotalSegmentator, this code is not general-purpose multi-class segmentation for arbitrary label sets.

The implementation is hard-coded for 5 foreground classes:

- label `1`
- label `2`
- label `3`
- label `4`
- label `5`

Background is implicit and reconstructed only at evaluation time.

Every loader converts the integer mask into 5 one-vs-rest binary channels:

- `label_1 = (mask == 1)`
- `label_2 = (mask == 2)`
- `label_3 = (mask == 3)`
- `label_4 = (mask == 4)`
- `label_5 = (mask == 5)`

This means the TotalSegmentator data in this repo has already been reduced to a 5-class problem. The original broader label space of TotalSegmentator is not used here.

## Data Layout And Local Dataset State

### Expected MMWHS layout

The MMWHS loader expects flat directories containing:

- `*-image.nii.gz`
- `*-label.nii.gz`
- `*-sdf.nii.gz`

Examples found locally:

- `data/MMWHS/CT/all/MMWHS-CT-001-image.nii.gz`
- `data/MMWHS/CT/all/MMWHS-CT-001-label.nii.gz`
- `data/MMWHS/CT/all/MMWHS-CT-001-sdf.nii.gz`
- `data/MMWHS/MRI/all/mr_train_1001_image.nii.gz`
- `data/MMWHS/MRI/all/mr_train_1001_label.nii.gz`
- `data/MMWHS/MRI/all/mr_train_1001_sdf.nii.gz`

The loader now explicitly fails if the SDF file is missing.

### Expected TotalSegmentator layout

The TotalSegmentator loader expects per-case subdirectories containing:

- `*image.nii.gz`
- `*label.nii.gz`

Example found locally:

- `data/TotalSegmentator/train/s0331/s0331-image.nii.gz`
- `data/TotalSegmentator/train/s0331/s0331-label.nii.gz`

No SDF file is required for this dataset because the loader computes class-wise SDFs online.

### Local sample counts

Observed in this workspace:

- MMWHS CT `all`: 20 image volumes
- MMWHS CT `training_set_full`: 16 image volumes
- MMWHS CT `testing_set`: 4 image volumes
- MMWHS MRI `all`: 20 image volumes
- MMWHS MRI `training_set_full`: 16 image volumes
- MMWHS MRI `testing_set`: 4 image volumes
- TotalSegmentator `train`: 596 image volumes
- TotalSegmentator `test`: 150 image volumes

The README statement that MMWHS `all = training_set_full + testing_set` is true for the local data.

### Actual volume shapes on disk

The packaged data already appears preprocessed to `64 x 64 x 64`:

- MMWHS CT image: `(64, 64, 64)`
- MMWHS CT label: `(64, 64, 64)`
- MMWHS MRI image: `(64, 64, 64)`
- TotalSegmentator image: `(64, 64, 64)`
- TotalSegmentator label: `(64, 64, 64)`

MMWHS SDF files are stored on disk in a higher-dimensional layout but load through `torchio` as:

- `mask_sdf.shape == (5, 64, 64, 64)`

## Preprocessing And Representation

### Image preprocessing

The project uses different fixed intensity windows per dataset/modality:

MMWHS CT:

- clamp to `[-250, 800]`
- rescale to `[-1, 1]`
- crop/pad to `(64, 64, 64)`

MMWHS MRI:

- clamp to `[0, 1000]`
- rescale to `[-1, 1]`
- crop/pad to `(64, 64, 64)`

TotalSegmentator:

- clamp to `[-250, 450]`
- rescale to `[-1, 1]`
- crop/pad to `(64, 64, 64)`

### Mask preprocessing

Masks are only cropped/padded to `(64, 64, 64)`, then expanded into 5 binary channels.

### SDF preprocessing

The SDF representation is central to the method.

For TotalSegmentator, `compute_sdf()` does:

- inside distance transform
- outside distance transform
- `sdf = outside_dist - inside_dist`
- divide by `32.0`
- clip to `[-0.2, 0.2]`

Interpretation:

- negative values are inside the object
- positive values are outside the object
- zero approximates the object boundary

At inference, the model’s generated SDF channels are converted back to masks by:

- `gen_mask_de_sdf = (gen_mask < 0.0)`

Observed locally, both image tensors and SDF tensors end up numerically normalized:

- images in `[-1, 1]`
- SDFs in `[-0.2, 0.2]`

## Augmentation

Training augmentation is minimal:

- one possible left-right style flip through `torchio.RandomFlip`

The implementation chooses `p` randomly from `{0, 1}` and then builds a deterministic transform with that `flip_probability`. That produces a 50% chance of a guaranteed flip and a 50% chance of no flip, while keeping image, mask, and SDF aligned for the sample.

No elastic deformation, rotation, scaling, cropping jitter, or intensity augmentation is used.

## Dataloader Behavior

This repo has an important asymmetry:

- in training mode, `get_MMWHS_dataloader()` and `get_TS_dataloader()` return the raw `Dataset`
- in test mode, they return a PyTorch `DataLoader`

The `Trainer` in `ddpm/diffusion.py` constructs its own training `DataLoader`.

Another specificity:

- test loaders hard-code `num_workers=20`
- they do not use the Hydra `num_workers` setting

Each sample returned by the loaders has:

- `name`: case identifier
- `img`: shape `(1, 64, 64, 64)`, `float32`
- `mask_sdf`: shape `(5, 64, 64, 64)`, `float32`
- `mask`: shape `(5, 64, 64, 64)`, `float32`
- `affine`: shape `(4, 4)`

## Model Architecture

### Primary model: `Unet3D`

All provided scripts use `ddpm.diffusion.Unet3D`.

With the shipped config values:

- base dim: `64`
- dim multipliers: `(1, 2, 4, 8)`
- input/output channels: `6`
- conditioning dimension: `16`
- parameter count: about `35.85M`

This model is a custom 3D U-Net-like denoiser with:

- residual blocks
- time embedding
- sparse spatial attention
- temporal attention
- skip connections

### Depth is treated specially

A major architectural specificity is that most convolutions and down/up-sampling are anisotropic:

- conv kernels are often `(1, 3, 3)`
- downsample is `(1, 4, 4)` with stride `(1, 2, 2)`
- upsample is `(1, 4, 4)` with stride `(1, 2, 2)`

So the network largely preserves the depth axis and downsamples only in-plane at many stages. Depth mixing comes mainly from the temporal-attention path, which treats the depth dimension like a frame axis.

This is one of the clearest signs that the model was adapted from a video-diffusion style architecture.

### Conditioning path exists but is effectively constant

`Unet3D` is built with `cond_dim=16`, but no meaningful conditioning signal is passed anywhere in training or inference.

If `cond is None`, the forward path creates a fixed 16D vector whose last entry is `1`. So in real use, the conditioning branch is effectively a learned constant rather than data-dependent conditioning.

Classifier-free guidance plumbing exists in the code, but the provided scripts do not use it meaningfully.

### Alternative model: `ddpm/unet.py`

There is a second model implementation based on MONAI building blocks. Training can instantiate it by setting `cfg.model.denoising_fn == 'UNet'`.

However, inference ignores that option and always rebuilds `Unet3D`.

Consequence:

- the repository nominally supports two denoisers for training
- the shipped inference path only supports checkpoints trained with `Unet3D`

## Diffusion Formulation

The diffusion object is `GaussianDiffusion_Nolatent`.

Important characteristics:

- no latent-space compression
- diffusion happens directly in voxel space
- cosine beta schedule
- default `timesteps=300` in all provided scripts
- loss on predicted noise, not directly on segmentation logits

The diffusion state includes standard DDPM buffers:

- `betas`
- `alphas_cumprod`
- `sqrt_alphas_cumprod`
- posterior coefficients
- related cached tensors

Those buffers are saved inside the checkpoint state dict along with the denoiser weights.

### What is diffused

Training separates the sample into:

- `x_start`: image tensor of shape `(B, 1, 64, 64, 64)`
- `mask_start`: SDF tensor of shape `(B, 5, 64, 64, 64)`

Noise is added independently to image and mask-SDF channels:

- `x_noisy = q_sample(x_start, t, noise_x)`
- `m_noisy = q_sample(mask_start, t, noise_m)`

These are concatenated into a 6-channel input to the denoiser:

- `input = cat(x_noisy, m_noisy)`

The denoiser predicts a 6-channel noise tensor, which is split back into:

- predicted image noise
- predicted mask noise

### Loss

The total loss is a simple sum:

- image noise reconstruction loss
- mask noise reconstruction loss

Supported losses:

- `l1`
- `l2`

All provided configs use `l1`.

There is no explicit segmentation-only loss, Dice loss, cross-entropy, topology loss, or mutual-exclusion penalty between classes.

## Training Pipeline

### Entry point

`train/train.py` is the training entry point.

High-level flow:

1. set random seed to `1`
2. build `Unet3D` or MONAI `UNet`
3. wrap the denoiser in `nn.DataParallel`
4. wrap that in `GaussianDiffusion_Nolatent`
5. construct the dataset
6. instantiate `Trainer`
7. optionally load a checkpoint
8. train until `train_num_steps`

One important detail:

- the training entry point hard-codes `set_seed(1)`
- the commented-out line suggests the config seed was intended to be used, but it is not active in the current code

### Default script settings

The provided shell scripts all use:

- `diffusion_img_size=64`
- `diffusion_depth_size=64`
- `diffusion_num_channels=6`
- `timesteps=300`
- `loss_type=l1`

Typical MMWHS training:

- `train_num_steps=10000`
- `save_and_sample_every=100`

TotalSegmentator training:

- `train_num_steps=50000`
- `save_and_sample_every=100`

There is also a newer local helper script:

- `traing_scripts/train_5jobs_singlegpu.sh`

That script launches 5 separate single-GPU jobs with smaller per-step batch sizes and larger gradient accumulation. It writes checkpoints under:

- `Model/manual_20260322_202341_singlegpu/`

### Trainer details

The trainer:

- creates a `DataLoader` internally
- cycles over it indefinitely
- uses AMP optionally, but all provided configs use `amp=False`
- maintains an EMA copy of the entire diffusion model
- saves both raw and EMA states

EMA settings:

- `ema_decay = 0.995`
- `step_start_ema = 2000`
- `update_ema_every = 10`

Checkpoint contents:

- `step`
- `model`
- `ema`
- `scaler`
- `optimizer`

The checkpoint keys `model` and `ema` are full diffusion-model state dicts, not just bare denoiser weights.

### Local evidence from logs

One local training log at `log_train/manual_20260322_202341_singlegpu/train_MMWHSCT_all_gpu1.log` shows:

- `len_dl 10` for the 20-case MMWHS CT all split with batch size 2
- training loss drops from about `1.76` into the `0.3` to `0.8` range over time

The log still contains a leftover source-project message:

- `found 20 videos as gif files`

That message is obviously inherited from upstream diffusion/video code and is not semantically correct for this project.

## Inference Pipeline

### Entry point

`test/inference.py` is the evaluation and sampling entry point.

High-level flow:

1. rebuild `Unet3D`
2. rebuild `GaussianDiffusion_Nolatent`
3. load checkpoint weights from `weight_key` in the checkpoint, default `ema`
4. iterate over test cases
5. run reverse diffusion with image guidance
6. threshold generated SDFs into masks
7. save generated and real NIfTI files
8. compute Dice and NSD

### Checkpoint resolution behavior

The helper `resolve_checkpoint_path(model_path, model_num)` behaves as follows:

- if `model_path` is a file, use it directly
- otherwise look for `model_path / f"model-{model_num}.pt"`

This matches the provided testing scripts, which pass values like:

- `model_path=./Model/DiffAtlas_MMWHS-CT_full`
- `model_num=pretrained_MMWHSCT_full`

So inference looks for:

- `./Model/DiffAtlas_MMWHS-CT_full/model-pretrained_MMWHSCT_full.pt`

### EMA loading

Inference defaults to:

- `weight_key: ema`

This is correct for the saved checkpoint format and matches the README note.

### Guided sampling is the key test-time trick

The most important inference-specific behavior is in `_apply_guidance()`:

- compute a noisy version of the real target image at timestep `t`
- overwrite channel `0` of the current sample with that noisy real image

So during every reverse step:

- the image channel is forcibly aligned to the target case
- only the mask/SDF channels are genuinely being denoised/generated

This is the practical mechanism that makes the method image-guided at inference time.

### Output decoding

After sampling:

- `result[:, 1:]` is taken as generated SDF mask channels
- each channel is thresholded at `0`
- class channels are saved separately

Then a combined label map is formed by:

1. creating a background channel where no foreground class is active
2. concatenating background plus 5 class channels
3. taking `argmax`

One implication:

- if multiple foreground classes overlap after thresholding, `argmax` will break ties by channel order, favoring the lowest-index active class

There is no explicit overlap resolution beyond that.

### Multi-seed inference

Inference supports repeated sampling per case:

- `seed_num`

For each case, the code:

- derives a deterministic case/trial seed
- samples `seed_num` times
- averages Dice and NSD across trials

The default config uses:

- `seed=1`
- `seed_num=1`
- `deterministic=true`

### Saved outputs

Per case, the inference script writes:

- the real image
- generated SDF for each class
- thresholded generated mask for each class
- real SDF for each class
- real binary mask for each class
- combined generated label map

All outputs are written as NIfTI files under:

- `inference_visualization/<dir_name>/<model_num>/`

## Metrics

### Dice

Dice is computed per class from the thresholded binary outputs.

If both prediction and target are empty for a class, Dice is set to `1.0`.

### NSD

The repo includes a custom NSD implementation in `test/inference.py`.

Important specifics:

- `NSDMetric(n_classes=6)` is used
- background is ignored internally
- threshold is fixed at `1.0` for every foreground class
- output is a 5-element vector for the foreground classes

The implementation computes simple boundary distances using binary dilation and Euclidean distance transforms. It is not a library callout; it is a local custom implementation.

A local smoke-test inference log at `log_inference/cuda_smoke_mmwhsct/pretrained_MMWHSCT_full.log` shows one sampled MMWHS CT case with per-class Dice roughly:

- `0.765`
- `0.897`
- `0.877`
- `0.726`
- `0.842`

and NSD roughly:

- `0.739`
- `0.802`
- `0.777`
- `0.525`
- `0.572`

## Configuration System

Hydra is used, but only lightly.

Config templates mostly define placeholders:

- `train/config/base_cfg.yaml`
- `train/config/model/ddpm.yaml`
- `train/config/dataset/MMWHS.yaml`
- `train/config/dataset/TS.yaml`
- `test/confs/infer.yaml`

Most real values are supplied on the command line by the shell scripts.

An observed Hydra output from `outputs/2026-03-22/20-23-43/.hydra/config.yaml` confirms a recent single-GPU run used:

- dataset: MMWHS MRI full training set
- batch size: `1`
- gradient accumulation: `24`
- save interval: `1000`
- train steps: `10000`

## Checkpoints Present In This Workspace

Pretrained checkpoint files found locally:

- `Model/DiffAtlas_MMWHS-CT_all/model-pretrained_MMWHSCT_all.pt`
- `Model/DiffAtlas_MMWHS-CT_full/model-pretrained_MMWHSCT_full.pt`
- `Model/DiffAtlas_MMWHS-MRI_all/model-pretrained_MMWHSMRI_all.pt`
- `Model/DiffAtlas_MMWHS-MRI_full/model-pretrained_MMWHSMRI_full.pt`

Additional local training outputs:

- `Model/manual_20260322_202341_singlegpu/DiffAtlas_MMWHS-CT_all_gpu1/model-1.pt` through `model-4.pt`
- same pattern for MMWHS CT full, MMWHS MRI all, MMWHS MRI full
- TotalSegmentator local run currently has `model-1.pt` and `model-2.pt`

Inspecting one pretrained checkpoint confirmed:

- top-level keys are `ema`, `model`, `optimizer`, `scaler`, `step`
- one local pretrained file is currently at `step = 6300`

## Important Implementation Specificities

These are the details most likely to matter if this repo is extended or debugged.

### 1. The method is joint image-mask diffusion, but inference is image-clamped

The model is trained to denoise both image and mask channels. But at test time, the image channel is forcibly replaced with the real noisy target image at every reverse step. So the deployed behavior is much closer to conditional mask generation than to full joint image-mask synthesis.

### 2. The segmentation output is SDF-first, not label-first

The model never predicts class logits directly. It predicts continuous SDF values per class and later thresholds them at zero.

### 3. Foreground classes are independent during training

The training loss treats the 5 SDF channels independently. Class exclusivity is not enforced during training. Any competition between classes appears only when the final hard label map is assembled with background plus `argmax`.

### 4. The code is tightly bound to 5 classes

This assumption appears everywhere:

- dataset loaders
- channel count in scripts
- inference post-processing
- metric class count

Changing the class count would require coordinated edits in multiple files.

### 5. Inference is hard-coded to `Unet3D`

This is one of the most important practical limitations in the repo. Training exposes a `denoising_fn` switch, but inference does not mirror it.

### 6. `num_frames` exists but is mostly structural

The diffusion object stores `num_frames`, but it is not actively used to enforce shapes in the core logic. The effective depth assumption instead comes from the dataset preprocessing and the network layout.

### 7. DataParallel is used only for training-time denoising

`train/train.py` wraps the denoiser in `nn.DataParallel`, and inference later strips the `module.` prefix from saved weights. This is why the loader includes `strip_module_prefix()`.

### 8. TotalSegmentator uses a custom Nibabel reader

The TS loader explicitly routes image reading through `nibabel_reader()`, returning a tensor plus affine. The MMWHS loader uses standard `torchio` image loading. This difference is intentional in the local code.

### 9. The codebase still carries upstream diffusion/video artifacts

Evidence includes:

- the “videos as gif files” training log message
- the temporal-attention framing
- the depth-as-frame architectural treatment

This is not a problem by itself, but it explains several naming choices that otherwise look odd in a medical segmentation project.

### 10. The text-conditioning path is dormant and incomplete

`ddpm/text.py` is inherited support code for BERT-based conditioning. It is not used by the provided training or inference scripts.

There is also a correctness issue in that file:

- `bert_embed()` computes a masked mean embedding in the non-CLS branch
- but ends with a bare `return`
- so that branch returns `None`

This does not affect the current DiffAtlas pipeline because text conditioning is not used here, but it would matter if someone tried to enable that path later.

## Mismatches, Limitations, And Risks

### Training/inference model mismatch risk

If someone trains with the alternative MONAI `UNet`, the provided inference script will not load it correctly because it always instantiates `Unet3D`.

### Hard-coded evaluation assumptions

NSD assumes:

- 6 total classes including background
- 5 evaluated foreground classes
- 1 mm threshold for all classes

That is tightly coupled to the current datasets and label encoding.

### Limited augmentation

The augmentation policy is extremely light. Generalization is therefore carried mostly by the diffusion formulation, the SDF representation, and the cross-domain training setups rather than rich data augmentation.

### No explicit use of spacing in final reported Dice

Dice is pure voxel overlap. NSD can accept spacing but the inference call uses the default spacing tuple rather than pulling per-case voxel spacing from the affine/header.

### Background handling is deferred

Background is not predicted directly during training. It is reconstructed at inference by checking where no foreground class is active.

## How To Think About This Repository

The most accurate mental model for this codebase is:

- a compact 3D DDPM
- trained on concatenated image plus per-class SDF channels
- operating on already standardized `64^3` medical volumes
- using a video-style 3D U-Net that treats depth partly as a temporal axis
- evaluated by clamping the image trajectory to the real target scan and letting the model denoise only the anatomical shape representation

So despite the “atlas” framing in the paper and README, the implementation here is not classical atlas registration. It is a learned diffusion prior over image-mask pairs with strong test-time image guidance.

## Bottom Line

This repo is technically focused and relatively easy to reason about because almost everything important is fixed:

- one spatial size
- one joint 6-channel representation
- five foreground classes
- one dominant denoiser architecture
- one SDF thresholding rule
- one guided-sampling strategy

Its main strengths are simplicity and a clean end-to-end path from dataset to checkpoint to NIfTI outputs.

Its main constraints are the hard-coded class/channel assumptions, the inference-only support for `Unet3D`, and the fact that “conditioning” in the denoiser is mostly vestigial while the real conditioning happens by directly overwriting the image channel during reverse diffusion.
