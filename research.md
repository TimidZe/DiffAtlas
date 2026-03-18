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

### 1. README and script paths are inconsistent

Examples:

- README says `training_scripts`, but the repo directory is `traing_scripts`
- README refers to commands like `train_MMWHSCT_full_all.sh` and `train_MMWHSMRI_full_all.sh`, which do not exist
- README dataset section says TotalSegmentator uses `training_set` and `testing_set`, but the repo actually uses `train` and `test`

### 2. The MMWHS MRI "all" training script points to a non-existent directory

`traing_scripts/train_MMWHSMRI_all.sh` uses:

- `dataset.root_dir=./data/MMWHS/MRI/training_all`

But the actual directory in this workspace is:

- `./data/MMWHS/MRI/all`

As written, that training script will fail.

### 3. Pretrained model download instructions are broken

The README uses `wget -o ...`, which writes logs to a file rather than saving the downloaded payload to that filename.

Observed result in this workspace:

- `Model/DiffAtlas_MMWHS-CT_full/pretrained_MMWHSCT_full` is plain ASCII text
- it contains a failed `wget` log, not a PyTorch checkpoint

### 4. Pretrained filename conventions do not match the inference loader

Inference expects:

- `model_path/model-{model_num}.pt`

The README examples download files named like:

- `pretrained_MMWHSCT_full`

Even if the download succeeded, inference would still look for:

- `model-pretrained_MMWHSCT_full.pt`

So the README/scripts and the actual loader naming convention do not line up.

### 5. Inference loads raw model weights, not EMA weights

Training checkpoints save both:

- `model`
- `ema`

But inference explicitly loads `["model"]`, not `["ema"]`.

That may be intentional, but it is atypical for diffusion projects, where EMA weights are often preferred at evaluation time.

### 6. Guidance cannot really be disabled with `False`

The sampling loops check:

- `if self.use_guide is not None`

So `False` still counts as "guided". Guidance is only skipped if `use_guide=None`.

### 7. There is substantial dead or inherited code

Unused or effectively dormant pieces include:

- alternate sampling loops `p_sample_loop_v2/v3/v4`
- `p_losses_image_only`
- circle-mask helper methods in both datasets
- text/BERT conditioning path
- the MONAI `UNet` inference path

This does not break the main workflow, but it makes the codebase look broader than it really is.

### 8. NSD evaluation does not use real voxel spacing

This can materially affect reported boundary metrics on anisotropic medical volumes.

### 9. MMWHS SDFs are external prerequisites, TotalSegmentator SDFs are computed on demand

This asymmetry is easy to miss and is one of the biggest practical data-specific assumptions in the repo.

### 10. The project depends on aggressively standardized inputs

The whole system assumes:

- fixed `64^3` shape
- fixed class set of five foreground labels
- preprocessed and intensity-clamped inputs
- one image channel only

This is not a drop-in segmentation framework for arbitrary 3D medical datasets without substantial preprocessing adaptation.

## Bug Audit: Scheduling and Control Flow

This section focuses specifically on the scheduling flow the code uses at train time and inference time: reverse diffusion timestep flow, checkpoint cadence, EMA cadence, resume flow, and per-case evaluation control flow.

### 11. The main reverse-diffusion "schedule" variables are computed and then ignored

In the main sampling code and two alternate variants, the code constructs schedule scaffolding and never uses it:

- `ddpm/diffusion.py:649-653`
- `ddpm/diffusion.py:611-614`
- `ddpm/diffusion.py:676-679`

Examples:

- `R = 2` or `R = 3` is assigned
- `recurrent = [0] * self.num_timesteps` is built
- selected entries are marked
- the array is never read again

Impact:

- the reverse process is not actually using any recurrent/repaint-style timestep schedule
- changing `R` has no effect
- the presence of this code strongly suggests an intended schedule was started and never finished

So the effective scheduler in the deployed path is just a plain monotonic `for`/`while` loop over timesteps.

### 12. `p_sample_loop()` becomes a complete no-op if guidance is disabled with `None`

The main inference path in `ddpm/diffusion.py:643-666` only performs denoising inside:

- `if self.use_guide is not None:`

There is no `else` branch. If `use_guide=None`, the loop simply decrements `i` until it exits and returns the original random initialization unchanged.

I verified this locally with a minimal script: with a fixed seed, `p_sample_loop(..., use_guide=None)` returns exactly the same tensor as the initial concatenated random `img` and `mask` noise.

Impact:

- the "unguided" path is broken
- there is no valid fallback reverse diffusion path in the main sampler
- any caller trying to disable image guidance gets random noise back, not a sampled result

### 13. `use_guide=False` still enables guidance

The same condition:

- `if self.use_guide is not None`

means that `False` still counts as "guided". Only `None` disables guidance, but as described above, `None` breaks the loop entirely.

I verified this locally:

- `use_guide=False` does not return the initial random tensor
- `use_guide=None` does

Impact:

- the guidance flag has misleading semantics
- there is no clean, working way to toggle between guided and unguided sampling in the main sampler

### 14. Guidance is applied every timestep because the unfinished schedule is ignored

Because the recurrent schedule array is unused, `p_sample_loop()` overwrites the image channel with the target image's noisy version at every reverse timestep:

- `ddpm/diffusion.py:657-662`

Impact:

- whatever more selective scheduling behavior the code seems to have been preparing for never happens
- the practical behavior is "hard image replacement on every step", not any kind of sparse or recurrent schedule

This is less a crash bug than an implementation bug: the code advertises a scheduling idea it does not actually execute.

### 15. Checkpoint saving is off by one optimizer update

The trainer's save condition is checked before incrementing `self.step`:

- optimizer step happens first in `ddpm/diffusion.py:949-951`
- save condition is checked using the old `self.step` in `ddpm/diffusion.py:953-960`
- only after that does `self.step += 1` happen in `ddpm/diffusion.py:963`

I verified this locally with a tiny mock model:

- with `save_and_sample_every=2` and `train_num_steps=3`, the only saved file is `model-1.pt`
- that checkpoint contains weights from the 3rd optimizer update, not the 2nd
- its stored `step` field is `2`, even though the weights are already one update ahead

Impact:

- milestone filenames do not correspond to the actual number of completed optimizer updates
- the `step` metadata written into checkpoints lags the saved weights by one update

### 16. The shipped `10001` / `50001` training lengths are compensating for the save bug

The off-by-one save logic also explains the odd script values:

- MMWHS scripts use `model.train_num_steps=10001`
- TotalSegmentator uses `model.train_num_steps=50001`

I verified the behavior locally:

- with `train_num_steps=2` and `save_and_sample_every=2`, no checkpoint is saved at all
- with `train_num_steps=3` and `save_and_sample_every=2`, `model-1.pt` appears

Impact:

- you only get the expected "final" milestone if you run one extra optimizer step past the nominal target
- the current scripts appear to be working around the broken scheduler rather than matching a clean definition of training length

### 17. EMA starts averaging one update later than `step_start_ema` suggests

EMA uses the same pre-increment `self.step` value:

- `ddpm/diffusion.py:877-881`
- `ddpm/diffusion.py:953-954`

Because `step_ema()` is called after the optimizer update but before `self.step += 1`, true EMA averaging starts one update later than the variable name implies.

I verified this with a tiny mock model and `step_start_ema=2`:

- steps 0 and 1 reset EMA to current weights
- actual averaging first occurs on the 3rd optimizer update

Impact:

- `step_start_ema` is effectively off by one update
- EMA warmup behavior is not what the configuration name suggests

### 18. `Trainer.load(..., map_location=...)` is broken

In `ddpm/diffusion.py:893-905`, the load path splits incorrectly:

- if `map_location` is not provided, it loads from `results_folder/model-{milestone}.pt`
- if `map_location` is provided, it tries `torch.load(milestone, map_location=...)`

That means the integer milestone is passed directly to `torch.load` instead of being turned into a file path.

I verified this locally: calling `trainer.load(7, map_location='cpu')` raises an error because PyTorch receives the integer `7` instead of a checkpoint path.

Impact:

- cross-device resume/loading via `map_location` is broken
- any code path that tries to load a numbered milestone onto CPU from GPU training will fail

### 19. The CUDA inference path likely crashes when merging class masks

In `test/inference.py:309-312`, the code mixes a likely GPU tensor with new CPU tensors:

- `gen_mask` comes from `result`, which stays on `device`
- `torch.tensor(1.0)` and `torch.tensor(0.0)` are created on CPU
- `background_mask = torch.ones(...)` is also created on CPU

Then the code performs:

- `torch.where(gen_mask < 0.0, torch.tensor(1.0), torch.tensor(0.0))`
- boolean indexing into `background_mask` using a mask derived from `gen_mask_de_sdf`

On CPU this is fine. On CUDA this is a device mismatch waiting to happen.

Impact:

- the exact path used to build the merged categorical prediction is not device-safe
- per-class output saving may work, but the "label-together-gen" branch is likely to fail on GPU inference

### 20. Inference intentionally randomizes its RNG state after each case, making evaluation non-reproducible

In `test/inference.py:258-264`, each case gets a random seed drawn from the current RNG state. Then at the end of the loop in `test/inference.py:332-335`, the script does:

- `th.random.seed()`
- `th.cuda.seed()`

with no explicit seed value, which reseeds from entropy.

Impact:

- rerunning the same evaluation command does not produce a deterministic per-case seed schedule
- even if a user sets a global seed before calling the script, the script deliberately discards deterministic RNG state between cases

This is a reproducibility bug in the evaluation flow.

### 21. Inference evaluates the non-EMA weights even though training maintains EMA

Training saves both:

- `model`
- `ema`

But inference explicitly extracts `["model"]` in `test/inference.py:217` and loads that into the diffusion object.

Impact:

- evaluation is not using the smoothed checkpoint that diffusion projects usually rely on
- the code is maintaining an EMA schedule during training but not consuming it in the main test flow

This is not a crash bug, but it is a workflow bug: the scheduled EMA maintenance is disconnected from the evaluation path that should benefit from it.

## Bottom Line

The repository implements a concrete and fairly specialized idea:

- train a 3D diffusion denoiser over joint image and per-class SDF volumes
- segment a target scan by anchoring the image channel to that scan throughout reverse diffusion
- recover masks by thresholding generated SDF channels at zero

Conceptually, the system is cleaner than the repo might first suggest. The actual working path is narrow:

- two dataset loaders
- one real backbone (`Unet3D`)
- one diffusion objective
- one image-guided sampling routine

Most of the complexity outside that path is either borrowed infrastructure or unfinished generalization.

The strongest implementation-specific takeaways are:

- DiffAtlas in code is a guided joint diffusion segmenter, not an atlas retrieval system
- SDF representation is central to both training and decoding
- inference is hard-wired around repeated noisy-image replacement
- the repo currently contains multiple reproducibility blockers in its docs, scripts, and scheduling/control flow

If this project were to be extended or productionized, the first priorities would be:

- fix checkpoint naming/download flow
- fix the broken save/EMA scheduling order in `Trainer.train`
- fix the main sampler so unguided sampling actually denoises
- make the inference mask-merging path device-safe
- fix script path mismatches
- make spacing-aware metrics consistent
- clarify the true data contract and preprocessing pipeline
- remove or quarantine dormant code paths
