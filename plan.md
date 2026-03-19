# DiffAtlas Fix Plan

## Goal

Fix the rough edges identified in `research.md` by making the project:

1. correct by construction
2. reproducible
3. executable out of the box
4. easier to maintain without changing the intended method unnecessarily

This plan is based on the current codebase, especially:

- `ddpm/diffusion.py`
- `test/inference.py`
- `train/train.py`
- `Dataset/MMWHS_Dataset.py`
- `Dataset/TS_Dataset.py`
- `traing_scripts/*.sh`
- `testing_scripts/*.sh`
- `README.md`

Execution tracking for this plan lives in `todo.md`, which breaks each phase into concrete implementation and validation tasks.

## Current Status

- Completed: Sections 1-9 and 11-13 are implemented and validated against the current codebase.
- Deferred by user request: Section 10 (`Pass Real Spacing into NSD`) remains intentionally unimplemented, so NSD still uses the repository's original default spacing behavior.

## First-Principles Framing

Before changing anything, it helps to define the invariants the repository should satisfy.

### Execution invariants

- A user should be able to train and infer using the provided scripts without editing paths by hand.
- The same tensor should never be mixed across CPU and GPU in a single operation.
- A sampling loop must always denoise when invoked. There should be no configuration that silently returns initial random noise.

### Training invariants

- `step` should mean "completed optimizer updates".
- checkpoint filenames, checkpoint metadata, and EMA updates should all use the same step definition.
- resume/loading should work identically on GPU and CPU.

### Evaluation invariants

- inference should load the weights that training intends evaluation to use
- metrics should respect the actual spatial meaning of the data
- repeated evaluation with the same seed should produce the same outputs

### Maintenance invariants

- scripts, configs, and README examples should match the actual repository layout
- dead scheduling code should either be completed or removed
- there should be one obvious path for training and one obvious path for inference

The plan below follows those invariants in priority order.

## Priority Order

### Phase 1: Fix hard execution failures

Status: Completed

These are the changes that can directly crash the pipeline or make a public script unusable.

1. Fix GPU/CPU tensor mixing in inference
2. Fix broken checkpoint loading with `map_location`
3. Fix the bad MRI "all" training path

### Phase 2: Fix correctness of training and sampling control flow

Status: Completed

These issues do not always crash immediately, but they break the semantics of the method.

1. Repair the sampler guidance logic
2. Decide what to do with the unfinished recurrent schedule
3. Fix trainer step/save/EMA ordering
4. Make inference deterministic when requested

### Phase 3: Fix evaluation correctness

Status: Partially completed

These issues produce misleading results even if the code runs.

1. Load EMA weights as default and keep the raw model as another option at inference
2. Pass real voxel spacing into NSD computation

### Phase 4: Fix repository entry points and documentation

Status: Completed

These changes make the project usable by a new user.

1. Align scripts with actual paths
2. Align README commands with actual filenames and checkpoint conventions
3. Clarify which data artifacts must already exist

## Detailed Plan

## 1. Fix CUDA Device Mismatch in Inference

Status: Completed

### Why this matters

`test/inference.py` currently builds the final merged categorical mask by mixing tensors that originate on different devices.

Relevant code:

- `result = sample_fn(...)`
- `gen_mask = result[:,1:(result.size()[1]),:,:,:]`
- `gen_mask_de_sdf = torch.where(gen_mask < 0.0, torch.tensor(1.0), torch.tensor(0.0))`
- `background_mask = torch.ones((1, 1, 64, 64, 64), dtype=torch.float16)`

`gen_mask` stays on `device`, while `torch.tensor(...)` and `torch.ones(...)` default to CPU.

### Fix strategy

Keep all computation on one device until serialization.

### Concrete changes

In `test/inference.py`:

- Replace scalar tensor creation with device-aware creation.
  - Use `torch.ones_like(...)`, `torch.zeros_like(...)`, or `torch.where(..., 1.0, 0.0)` on the same device.
- Construct `background_mask` on `gen_mask.device`.
- Perform `torch.cat`, `argmax`, and background masking on the same device.
- Only move tensors to CPU immediately before saving with TorchIO or converting to NumPy.

### Recommended implementation shape

- For per-class thresholding:
  - `gen_mask_i_de_sdf = (gen_mask_i < 0).to(gen_mask_i.dtype)` or `.float()`
- For full-mask thresholding:
  - `gen_mask_de_sdf = (gen_mask < 0).to(gen_mask.dtype)` or `.float()`
- For background:
  - `background_mask = torch.ones((1, 1, 64, 64, 64), device=gen_mask.device, dtype=gen_mask_de_sdf.dtype)`

### Validation

- Run `test/inference.py` on CPU for one case and confirm outputs are unchanged in structure.
- Run `test/inference.py` on CUDA for one case and confirm the merged mask path no longer crashes.
- Confirm `label-together-gen.nii.gz` is still written.

## 2. Fix `Trainer.load(..., map_location=...)`

Status: Completed

### Why this matters

`ddpm/diffusion.py` currently does this:

- if `map_location` is provided: `torch.load(milestone, map_location=map_location)`

That passes an integer or symbolic milestone directly to `torch.load`.

### Fix strategy

Normalize the checkpoint path first, then call `torch.load` exactly once.

### Concrete changes

In `ddpm/diffusion.py`, refactor `Trainer.load` so it:

1. resolves `milestone == -1` to an integer milestone
2. resolves the actual checkpoint path
3. calls `torch.load(checkpoint_path, map_location=map_location)` once

### Recommended implementation shape

- If `milestone` is an int, build `self.results_folder / f"model-{milestone}.pt"`
- If later needed, optionally allow a direct filepath, but do not mix that into this change unless necessary
- Use `Path` consistently since the file already imports it

### Validation

- Save a tiny checkpoint locally
- load it with `trainer.load(1)`
- load it with `trainer.load(1, map_location='cpu')`
- verify model, EMA, optimizer, and scaler states restore without error

## 3. Fix the MMWHS MRI "all" Training Script

Status: Completed

### Why this matters

`traing_scripts/train_MMWHSMRI_all.sh` points to:

- `./data/MMWHS/MRI/training_all`

But the actual dataset path used elsewhere and present in this repo is:

- `./data/MMWHS/MRI/all`

### Fix strategy

Correct the script path, then make sure README references the same path.

### Concrete changes

- Edit `traing_scripts/train_MMWHSMRI_all.sh`
- Replace `training_all` with `all`

### Validation

- Run the script with `bash -n`
- Run the command prefix up to dataset construction and verify the dataloader can enumerate samples

## 4. Repair the Sampler Guidance Logic

Status: Completed

### Why this matters

The main inference loop in `ddpm/diffusion.py` has two coupled bugs:

1. `if self.use_guide is not None:` treats `False` as guided
2. there is no `else` branch, so `use_guide=None` makes the loop return raw random noise

### Fix strategy

The current public inference path is actually guided, not unguided:

- `test/inference.py` instantiates `GaussianDiffusion_Nolatent(...)` without overriding `use_guide`
- `GaussianDiffusion_Nolatent.__init__` sets `use_guide=True` by default
- `test/inference.py` then calls `diffusion.p_sample_loop(...)`

So the out-of-the-box path always executes the guided branch in `p_sample_loop`.

However, the unguided path still needs to be fixed for first-principles reasons:

- `use_guide` is part of the diffusion object's public behavior
- `p_sample_loop_v3` and `p_sample_loop_v4` suggest experimentation around alternative sampling modes
- a boolean flag should never silently degrade into "return untouched random initialization"

The plan should therefore preserve the current guided behavior while making the unguided path semantically valid.

### Concrete changes

In `ddpm/diffusion.py`:

- Replace `if self.use_guide is not None:` with a strict boolean check
- Add an explicit unguided branch that still calls `self.p_sample(...)`
- Keep the guided branch behavior unchanged for the public inference flow

### Recommended implementation shape

At each timestep in `p_sample_loop`:

- if `self.use_guide` is `True`:
  - compute `real_noisy_image`
  - overwrite channel 0 with the guided image state
  - call `self.p_sample(...)`
- else:
  - call `self.p_sample(...)` without image replacement

Use strict boolean semantics:

- `use_guide=True`: guided segmentation, which matches the current shipped inference path
- `use_guide=False`: valid unguided reverse denoising

Do not use `None` as a special mode unless a third state is genuinely needed and documented.

### Validation

- Keep the current `test/inference.py` behavior unchanged with default settings
- Add a small local sampling check that verifies:
  - `use_guide=True` still performs guided denoising
  - `use_guide=False` no longer returns untouched initialization
  - both branches complete the full reverse loop


## 5. Decide What to Do with the Unused Reverse Schedule

Status: Completed

### Why this matters

`p_sample_loop`, `p_sample_loop_v2`, and `p_sample_loop_v4` all build:

- `R`
- `recurrent = [0] * self.num_timesteps`

and then never use the resulting schedule.

This creates two problems:

- the code advertises scheduling behavior that does not exist
- the project remains harder to reason about than necessary

### Fix strategy

Choose one of two paths and commit to it.

### Option A: remove dead schedule scaffolding

Use this if the project goal is to stabilize the current implementation with minimal method changes.

Changes:

- remove `R` and `recurrent` from unused loops
- remove unused alternate loops if they are not part of a documented experiment

Pros:

- simplest and safest
- reduces ambiguity


### Recommendation

Take Option A. The current repo does not expose recurrent scheduling via config, tests, or docs, so dead code removal is the lowest-risk first-principles fix.


## 6. Fix Trainer Step Semantics

Status: Completed

### Why this matters

`Trainer.train` currently defines `self.step` ambiguously. It is used for:

- logging
- EMA scheduling
- save scheduling
- checkpoint metadata

but incremented only after all of those decisions.

This breaks the meaning of milestone numbers and forces script workarounds like `10001`.

### Fix strategy

Define one invariant:

- `self.step` means completed optimizer updates

Then enforce that everywhere.

### Concrete changes

In `ddpm/diffusion.py`, rewrite the bottom half of `Trainer.train` so the order is:

1. accumulate gradients
2. optimizer step
3. scaler update / zero grad
4. increment `self.step`
5. run EMA schedule using the new step count
6. run save schedule using the new step count
7. log using the new step count if desired

### Recommended implementation shape

- Increment `self.step` immediately after the optimizer update completes
- Use `if self.step % self.update_ema_every == 0:`
- Use `if self.step % self.save_and_sample_every == 0:`
- Save milestone as `self.step // self.save_and_sample_every`
- Store `step=self.step` in the checkpoint

### Downstream changes

After this fix:

- training shell scripts should likely change from `10001` to `10000`
- TotalSegmentator should likely change from `50001` to `50000`

Do not edit the scripts until the trainer behavior is fixed and verified.

### Validation

- Create a tiny mock trainer and verify:
  - at step 2 with save interval 2, `model-1.pt` is saved
  - checkpoint `step` equals the actual completed update count
- Resume from that checkpoint and continue training

## 7. Fix EMA Scheduling Semantics

Status: Completed

### Why this matters

EMA depends on the same `step` semantics as saving. If `step` is wrong, EMA warmup is wrong.

### Fix strategy

Once `self.step` is redefined as completed updates, update `step_ema()` semantics to match that meaning.

### Concrete changes

- Keep `step_start_ema` as a completed-update threshold
- Apply EMA after incrementing `self.step`
- Review whether `self.step < self.step_start_ema` or `<=` matches the intended meaning

### Recommendation

Use:

- before threshold: copy model to EMA
- at and after threshold: average

That is the least surprising interpretation.

### Validation

- Mock-train with `step_start_ema=2`
- confirm averaging begins exactly when documentation/config says it should

## 8. Make Inference Reproducible

Status: Completed

### Why this matters

The current inference loop intentionally destroys RNG determinism between cases:

- it draws a random per-case seed
- then reseeds from entropy at the end of the case

This makes regression testing and result comparison harder than necessary.

### Fix strategy

Make reproducibility a first-class config option.

### Concrete changes

In `test/confs/infer.yaml`, add fields like:

- `seed: 1`
- `deterministic: True`

In `test/inference.py`:

- set the initial seed once at startup if provided
- remove the entropy reseeding at the end of each case
- if multiple seeds are desired, derive them deterministically from the base seed and sample index

### Recommended implementation shape

- one helper function like `set_seed(seed, deterministic=True)`
- call it once at startup
- if `seed_num > 1`, use `base_seed + trial_index` or `base_seed + case_index * k + trial_index`

### Validation

- run inference twice with the same seed and compare logged per-case seeds and Dice outputs
- run inference with a different seed and confirm outputs can change

## 9. Load EMA Weights by Default at Inference

Status: Completed

### Why this matters

Training already pays the cost of maintaining EMA weights. Not using them at inference is inconsistent with the current training design.

### Fix strategy

Default to EMA, but keep raw-model loading available as an explicit option for debugging.

### Concrete changes

In `test/confs/infer.yaml`, add a config field:

- `weight_key: ema`

In `test/inference.py`:

- load `state = load_state_dict(... )`
- select `state[conf.weight_key]`
- strip `module.` prefixes as before

### Validation

- verify both `weight_key=model` and `weight_key=ema` load successfully
- compare a small inference run for both modes

## 10. Pass Real Spacing into NSD (Do Not implement yet)

Status: Deferred by user request

### Why this matters

`NSDMetric` supports spacing, but inference never passes it. The dataset already returns `affine`, which contains the scale needed to approximate voxel spacing.

### Fix strategy

Derive spacing from the affine used for the sample and pass it into `get_nsd`.

### Concrete changes

In `test/inference.py`:

- extract voxel spacing from `affine`
- pass it into `get_nsd(..., spacing=spacing_tuple)`

### Recommended implementation shape

Since current data is axis-aligned after preprocessing, a practical first fix is:

- spacing = absolute values of the first three diagonal affine entries

If future data may include rotations/shears, replace this with norm-of-column extraction.

### Caveat

Because the repository crops/pads to `64^3`, verify that preserving the original affine is still semantically correct for these saved tensors. If preprocessing changes spatial support without resampling, document that limitation explicitly.

### Validation

- run NSD on a sample with anisotropic spacing
- confirm spacing passed to `compute_surface_dice` matches the affine-derived values
- defer this change until the more critical execution and control-flow bugs are fixed, since spacing correctness affects metric interpretation rather than pipeline execution

## 11. Align Scripts and README with Actual Repository Behavior

Status: Completed

### Why this matters

The repo currently has mismatches across code, scripts, and documentation. Users enter through these surfaces first.

### Fix strategy

Treat shell scripts and README as public API.

### Concrete changes

#### Training scripts

- Fix `traing_scripts/train_MMWHSMRI_all.sh` dataset path
- After trainer step semantics are fixed, update `train_num_steps` values if the extra `+1` is no longer needed

#### Testing scripts

Review each `testing_scripts/*.sh` entry for:

- valid `model_path`
- `model_num` matching the `model-{model_num}.pt` convention
- correct dataset roots

#### README

Update:

- `training_scripts` vs `traing_scripts`
- command names that do not exist
- checkpoint download names so they match inference loader expectations
- any references that still imply broken paths or missing released checkpoints

### Validation

- For every script, run `bash -n <script>`
- For representative train/test scripts, manually inspect the expanded command line
- Confirm README commands point to real files in the repo

## 12. Clarify the Data Contract

Status: Completed

### Why this matters

The code has a non-obvious asymmetry:

- MMWHS requires precomputed SDF files on disk
- TotalSegmentator computes SDF online

That is a hidden operational dependency.

### Fix strategy

Make the contract explicit in documentation and, where possible, in code.

### Concrete changes

- In `README.md`, state clearly that:
  - MMWHS requires `*-sdf.nii.gz`
  - TotalSegmentator only requires image and label volumes
- Optionally add an assertion in `Dataset/MMWHS_Dataset.py` with a clear error message if the SDF file is missing

### Validation

- remove or rename one MMWHS SDF file in a local test and confirm the error message is actionable

## 13. Clean Up Dead or Experimental Paths

Status: Completed

### Why this matters

The repository currently includes multiple paths that are either unused or partially inherited:

- `p_sample_loop_v2`
- `p_sample_loop_v3`
- `p_sample_loop_v4`
- `p_losses_image_only`
- BERT/text conditioning code
- unused dataset geometric helpers

This increases mental overhead and makes bug fixing riskier.

### Fix strategy

Reduce the supported surface area after the critical fixes land.

### Concrete changes

Short term:

- add comments marking experimental or unused paths
- make the main code path explicit

Medium term:

- remove dead methods that are not referenced anywhere
- keep only paths that are config-selectable or documented

### Validation

- use `rg` to confirm removed code is not referenced
- rerun the primary training and inference smoke tests afterward

## Validation Matrix

After implementation, run this validation sequence.

### Smoke checks

1. `python train/train.py ...` on a tiny CPU mock or one small real batch
2. `python test/inference.py ...` on CPU for one case
3. `python test/inference.py ...` on CUDA for one case

### Scheduler checks

1. checkpoint saved at the exact intended milestone
2. checkpoint `step` equals completed update count
3. EMA starts exactly at the configured threshold
4. `trainer.load(..., map_location='cpu')` works

### Sampling checks

1. `use_guide=True` still performs guided sampling
2. `use_guide=False` performs real denoising
3. no mode returns untouched initialization unless explicitly intended

### Evaluation checks

1. EMA and raw weights both load if configured
2. NSD uses affine-derived spacing
3. repeated inference with the same seed is identical

### Documentation checks

1. every README command points to a real file
2. every shell script references a real data path
3. checkpoint filenames in docs match the inference loader convention

## Suggested Implementation Sequence

Use this order to reduce rework:

1. Fix `Trainer.load`
2. Fix trainer step/save/EMA ordering
3. Update training scripts if step counts no longer need `+1`
4. Fix sampler guidance logic
5. Decide whether to remove or implement recurrent scheduling scaffolding
6. Fix device handling in `test/inference.py`
7. Add deterministic inference seed handling
8. Add EMA-vs-model weight selection in inference
9. Pass affine-derived spacing into NSD
10. Fix MRI-all training script path
11. Update README and shell scripts
12. Clean up dead code

## Deliverables

A complete fix pass should result in:

- corrected training/inference control flow
- deterministic, device-safe inference
- valid checkpoint save/load semantics
- docs and scripts that match the code
- a smaller and clearer public execution surface

That is the minimum bar for turning the current repo from "research code with sharp edges" into "usable engineering artifact."
