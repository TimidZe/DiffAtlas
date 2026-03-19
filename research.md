# DiffAtlas Repository Research Report

## Scope

This report is based on reading the repository source, configs, scripts, and shipped data layout inside this workspace. I also ran a few lightweight local sanity checks:

- instantiated the default `Unet3D` and diffusion wrapper
- loaded representative MMWHS and TotalSegmentator samples
- verified tensor shapes through one forward loss pass
- inspected the checked-in pretrained artifact under `Model/`

The goal of this report is to describe what the project actually does, how it is wired together, what assumptions it makes, and where the code/docs/scripts diverge.

## Executive Summary

DiffAtlas is a 3D medical image segmentation project built around a joint image-mask diffusion model. Instead of training a direct image-to-label network, it trains a denoiser on a 6-channel tensor consisting of:

- 1 image channel
- 5 signed distance field (SDF) mask channels

At training time, the model learns to predict the diffusion noise added to both the image and the SDF mask channels. At inference time, it does not freely generate an image-mask pair. Instead, it repeatedly replaces the image channel with a noisy version of the target scan and only lets the reverse process generate the mask channels. In practice, this means the deployed method is a guided conditional segmentation process, where the "conditioning" is implemented by direct image-channel replacement rather than by an explicit conditioning encoder.

The repository is small in terms of code and large in terms of bundled data. Most of the project logic lives in:

- `train/train.py`
- `ddpm/diffusion.py`
- `Dataset/MMWHS_Dataset.py`
- `Dataset/TS_Dataset.py`
- `test/inference.py`

The implementation is workable, but it also contains several inherited or unfinished paths, plus multiple documentation/script issues that would block a fresh user from reproducing results without manual fixes.

## Repository Structure

High-value directories/files:

- `train/`
  - Hydra-driven training entrypoint and configs.
- `ddpm/`
  - Main model, diffusion process, and training loop.
- `Dataset/`
  - Dataset wrappers for MMWHS and TotalSegmentator.
- `test/`
  - Inference and metric computation.
- `traing_scripts/`
  - Shell launchers for training. The directory name is misspelled as `traing_scripts`, not `training_scripts`.
- `testing_scripts/`
  - Shell launchers for inference/evaluation.
- `data/`
  - Preprocessed NIfTI volumes are already present in this workspace.
- `Model/`
  - Intended checkpoint location.
- `sdf.py`
  - SDF generation utility used by TotalSegmentator.

Code footprint is modest; data footprint is the bulk of the repo.

## What the Project Actually Models

### Core representation

The default configuration uses `diffusion_num_channels=6`. That is not six image modalities. It is:

- channel 0: normalized 3D medical image
- channels 1-5: per-class SDF volumes

This is visible in the training loss path in `ddpm/diffusion.py:745-776`, where:

- image noise is added to `x_start`
- mask noise is added to `mask_start`
- the two are concatenated and passed through the denoiser
- the denoiser predicts noise for both parts

The segmentation output is therefore not trained as categorical logits. It is trained as five continuous SDF channels, one per foreground class.

### Why SDFs matter here

`sdf.py:5-10` defines the SDF as:

- `outside_dist - inside_dist`
- divided by `32.0`
- clipped to `[-0.2, 0.2]`

So inside an object the SDF is negative, and outside it is positive. During inference, each generated SDF channel is converted back to a binary mask by thresholding at zero in `test/inference.py:283-307`.

This is an important design choice. The model is not directly predicting hard masks. It predicts smooth signed geometry proxies and then thresholds them.

## Data Pipeline and Dataset Assumptions

### Datasets present in this workspace

Observed counts in the checked-in data:

- `data/MMWHS/CT/training_set_full`: 16 scans
- `data/MMWHS/CT/testing_set`: 4 scans
- `data/MMWHS/CT/all`: 20 scans
- `data/MMWHS/MRI/training_set_full`: 16 scans
- `data/MMWHS/MRI/testing_set`: 4 scans
- `data/MMWHS/MRI/all`: 20 scans
- `data/TotalSegmentator/train`: 596 scans
- `data/TotalSegmentator/test`: 150 scans

These counts match the README claim that MMWHS `all` is training plus testing.

### MMWHS loader

`Dataset/MMWHS_Dataset.py`:

- loads `*image.nii.gz`, corresponding `*label.nii.gz`, and precomputed `*sdf.nii.gz`
- applies intensity preprocessing based on modality:
  - CT: clamp `[-250, 800]`, rescale to `[-1, 1]`
  - MRI: clamp `[0, 1000]`, rescale to `[-1, 1]`
- crops/pads image and label to `(64, 64, 64)`
- during training, optionally flips along axis 1
- converts integer labels `1..5` into five one-vs-rest foreground channels

Important implementation detail:

- the MMWHS SDF is expected to already exist on disk and is not recomputed in the loader
- the code does not apply an explicit crop/pad transform to the SDF before training, but the shipped files are already effectively aligned and load as `(5, 64, 64, 64)` after TorchIO ingestion

### TotalSegmentator loader

`Dataset/TS_Dataset.py` is similar, but with a key difference:

- it computes SDFs on the fly from the loaded categorical label map via `compute_sdf`

Preprocessing:

- clamp `[-250, 450]`
- rescale to `[-1, 1]`
- crop/pad to `(64, 64, 64)`
- optional flip on train

### Data format observations

Local sample inspection shows:

- images and labels are already stored at `64 x 64 x 64`
- labels are six-valued categorical masks: background `0` plus foreground `1..5`
- MMWHS SDF files are stored on disk with an unusual extra-dimension NIfTI layout, but TorchIO ultimately yields `(5, 64, 64, 64)`

This means the repo is not handling arbitrary raw clinical scans. It assumes a very specific preprocessed data contract.

### Dead code in dataset classes

Both dataset classes contain unused helpers for:

- 2D projection
- minimal enclosing circles
- circle mask creation and expansion to 3D

These methods are not referenced anywhere in the repository. They look like remnants from an earlier prompting or ROI-masking idea.

## Model Architecture

### Default backbone: `Unet3D`

The default denoiser is `Unet3D` in `ddpm/diffusion.py:299-448`.

Main characteristics:

- 3D convolutions with kernels shaped to preserve depth more conservatively than height/width in several places
- four resolution levels from `dim_mults=(1, 2, 4, 8)`
- sinusoidal timestep embedding
- residual blocks with FiLM-like time/condition modulation
- sparse spatial linear attention
- temporal attention across the depth/frame axis
- optional classifier-free guidance machinery via `cond` and `null_cond_prob`

With the default training script settings:

- base dim = 64
- channels = 6
- parameter count is about 35.85M

### Conditioning

The code nominally supports conditioning:

- `cond_dim=16` is passed in training and inference
- `Unet3D.forward` fabricates a constant one-hot-like fallback condition if `cond is None`

In practice, no meaningful external conditioning signal is used anywhere in this repo. The condition is effectively a fixed dummy vector.

There is also dormant BERT/text-conditioning code in `ddpm/text.py`, inherited from a more general diffusion codebase. It is not used by the actual medical segmentation workflow.

### Alternate backbone: `UNet`

`train/train.py:45-50` allows selecting a MONAI-based `UNet` from `ddpm/unet.py`.

Important limitation:

- `test/inference.py` always instantiates `Unet3D`
- there is no corresponding inference branch for the alternate `UNet`

So the alternative backbone exists, but the shipped inference path is effectively hard-wired to `Unet3D`.

## Diffusion Objective and Training Behavior

### Training entrypoint

`train/train.py`:

- uses Hydra config composition
- chooses dataset and denoiser
- wraps the denoiser in `nn.DataParallel`
- builds `GaussianDiffusion_Nolatent`
- creates a `Trainer`
- logs stdout to `log_train/<timestamp>.log`

### Noise-prediction objective

`GaussianDiffusion_Nolatent` in `ddpm/diffusion.py:469-807` uses:

- cosine beta schedule
- standard DDPM forward noising
- standard reverse mean/variance reconstruction from predicted noise

Training loss in `p_losses` is:

- `L1(noise_x, predicted_noise_x) + L1(noise_m, predicted_noise_m)` by default

So the model is trained as a conventional noise predictor over the concatenated image-plus-SDF tensor.

### What "Nolatent" means here

The class name suggests there is no separate latent autoencoder stage. The diffusion process operates directly in the voxel space of the preprocessed image and SDF channels.

### Trainer behavior

`Trainer` in `ddpm/diffusion.py:811-965`:

- wraps the dataset in a PyTorch `DataLoader`
- uses gradient accumulation
- keeps an EMA copy of the model
- saves checkpoints every `save_and_sample_every`

Notable specifics:

- MMWHS scripts train for `10001` optimizer steps
- TotalSegmentator trains for `50001` optimizer steps
- effective batch size is `batch_size * gradient_accumulate_every`
- the trainer saves checkpoints as `model-<milestone>.pt`
- despite the class being called `Trainer`, it does not sample or validate during training; it only saves checkpoints

The logging string `found {len(self.ds)} videos as gif files` is clearly inherited from another project and has nothing to do with this repo's NIfTI datasets.

## Inference and Segmentation Mechanics

### Inference entrypoint

`test/inference.py`:

- loads config from `test/confs/infer.yaml` plus CLI overrides
- instantiates `Unet3D`
- wraps it in `GaussianDiffusion_Nolatent`
- loads weights from `model_path/model-{model_num}.pt`
- iterates over the test dataloader
- runs `diffusion.p_sample_loop`
- thresholds generated SDFs to binary masks
- computes Dice and NSD
- writes outputs under `inference_visualization/`

### The key project-specific inference trick

The main inference behavior is in `ddpm/diffusion.py:643-666`.

For every reverse timestep:

1. sample the target image forward to the current noise level with `q_sample`
2. overwrite channel 0 of the current state with that noisy real image
3. run one reverse denoising step on the concatenated image-plus-mask tensor

This means:

- the image channel is not actually generated at test time
- the mask channels are denoised while the image channel is forcibly anchored to the target scan at every timestep

This is the most important implementation detail in the repo. It is the practical mechanism by which the diffusion model becomes a segmentation model.

### Mask decoding

After sampling:

- each generated SDF channel is binarized with `gen_mask_i < 0`
- a background channel is synthesized
- all channels are concatenated
- `argmax` is used to form a single categorical mask volume

Implication:

- overlapping foreground predictions are resolved by naive argmax over equal-valued binary channels
- if multiple foreground classes claim the same voxel, the lower-index class wins

### What is saved

The inference script saves:

- the real input image
- generated SDF masks per class
- thresholded generated binary masks per class
- real SDF masks
- real binary masks
- the merged generated categorical mask

It does not save the generated image channel, even though `gen_image` is extracted.

## Metrics

### Dice

Dice is computed per class on the thresholded binary outputs.

### NSD

NSD is implemented manually in `test/inference.py` rather than using MONAI's built-in metrics.

Important caveat:

- the metric implementation supports physical spacing
- the call site does not pass real voxel spacing
- default spacing `(1.0, 1.0, 1.0)` is therefore used for all scans

That matters because inspected MMWHS affines are anisotropic, for example roughly:

- CT sample: `(2.43, 1.99, 1.96)`
- MRI sample: `(1.17, 1.12, 1.19)`

So the reported NSD is effectively computed in normalized voxel space, not true physical millimeters.

## Config and Script Behavior

### Hydra setup

Training config composition is minimal:

- `train/config/base_cfg.yaml`
- dataset-specific YAML under `train/config/dataset/`
- model YAML under `train/config/model/ddpm.yaml`

The shell scripts provide almost all meaningful values via command-line overrides.

### Effective training settings in scripts

The provided scripts consistently use:

- image size `64`
- depth size `64`
- channels `6`
- timesteps `300`
- batch size `12`
- `num_workers=20`

This repo is tuned around the preprocessed `64^3` regime. It is not a generic multi-resolution pipeline.

## Empirical Sanity Checks

I ran a forward pass using one MMWHS CT sample and the default model/diffusion stack. The tensor contract is internally consistent:

- image tensor: `(1, 1, 64, 64, 64)`
- SDF tensor: `(1, 5, 64, 64, 64)`
- concatenated diffusion state: 6 channels total

I also ran a tiny 2-step random-weight sampling pass and confirmed the inference path returns `(1, 6, 64, 64, 64)` with image and mask values clamped within `[-1, 1]` by the diffusion reconstruction path.

## Specific Findings and Rough Edges

1. Critical Execution Crashers (Needs Immediate Fixes)
These bugs will immediately crash the pipeline or result in completely invalid out-of-the-box behavior.

CUDA Device Mismatch in Inference: During inference, gen_mask remains on the GPU because it is sliced directly from result. The code then mixes it with CPU tensors like torch.tensor(1.0) and background_mask. This will cause a device mismatch crash on the GPU.


Broken map_location Checkpoint Loading: In Trainer.load, if map_location is provided, the code executes torch.load(milestone, map_location=map_location) instead of joining the milestone integer with the results_folder path. Passing an integer to torch.load will throw an error.

MMWHS MRI "All" Script Directory Mismatch: The script train_MMWHSMRI_all.sh points to ./data/MMWHS/MRI/training_all, but the data is instructed to be placed in ./data/MMWHS/MRI/all. The script will fail immediately due to a missing directory.

2. Methodological & Metric Flaws
These bugs allow the code to run but silently compromise the validity of the model's outputs and evaluation metrics.

Inference Ignores EMA Weights: Diffusion models heavily rely on Exponential Moving Average (EMA) weights for high-quality sampling. While Trainer.save correctly saves both model and ema, the inference script explicitly extracts and loads only ["model"].

Guidance Toggles and Unguided Sampling are Broken: The sampling loop condition if self.use_guide is not None: evaluates to True even if use_guide=False. Furthermore, if use_guide=None is explicitly passed, the lack of an else branch means the reverse diffusion loop does absolutely no denoising, returning the initial random noise.

NSD Evaluation Ignores Real Voxel Spacing: The get_nsd metric is called without passing a spacing argument. This forces compute_surface_dice to fall back to the default (1.0, 1.0, 1.0) spacing. For anisotropic medical volumes, this will result in mathematically incorrect boundary metrics.

3. Silent Logic Bugs & Scheduling Errors
These are underlying logical errors where the code behaves differently than its variables or configurations suggest.

Trainer Off-By-One Errors: In Trainer.train, optimizer steps occur before the save and EMA conditions are checked, and self.step is only incremented at the very end of the loop. This causes checkpoints to lag by one update step, and explains why the training scripts must specify 10001 or 50001 steps to trigger the final save.

Unused Reverse Diffusion Schedule Variables: The main p_sample_loop constructs a recurrent schedule array intended for RePaint-style scheduling, but never actually reads from it during the reverse process. 

Inference RNG is Deliberately Discarded: At the end of every inference case, th.random.seed() and th.cuda.seed() are called without arguments, which reseeds the generators from entropy. This intentionally prevents reproducible evaluations.

4. Repository Housekeeping & Data Asymmetries
These points affect the readability and maintainability of the project.

README Path and Naming Typos: The README refers to training_scripts, but the actual folder is traing_scripts. Additionally, inference expects checkpoints named model-{model_num}.pt, which does not align with the pretrained_MMWHSCT_full output filenames suggested in the README.
