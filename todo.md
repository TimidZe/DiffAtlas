# DiffAtlas Fix TODO

This file is the execution checklist for [plan.md](/home/estar/TZNEW/DiffAtlas/DiffAtlas/plan.md). It keeps the implementation order strict so the repository is stabilized from the bottom up instead of mixing critical fixes with cleanup work.

## Working Rules

- [x] Do not change method behavior unless the task explicitly requires it.
- [x] Do not start lower-priority phases until the current phase has code changes and validation completed.
- [x] Keep every fix tied to one invariant from `plan.md`: execution, training, evaluation, or maintenance.
- [x] After each completed task, update `research.md` and `README.md` only if the behavior or user contract actually changed.
- [x] Preserve the current default guided inference behavior unless a task explicitly changes it.

## Phase 0: Baseline and Safety Checks

### 0.1 Confirm current repository entry points

- [x] Re-read [train/train.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/train/train.py) and identify the exact training entry path.
- [x] Re-read [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py) and identify the exact inference entry path.
- [x] Re-read [ddpm/diffusion.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py) and mark the primary training, checkpointing, and sampling methods that will be edited.
- [x] Re-read the shell entry points under [traing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/traing_scripts) and [testing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/testing_scripts) before touching script or README behavior.

### 0.2 Establish a minimal validation workflow

- [x] Define one minimal training smoke command that exercises checkpoint save/load quickly.
- [x] Define one minimal CPU inference smoke command that exercises mask merging and metric computation.
- [x] Define one minimal CUDA inference smoke command that exercises the same path on GPU.
- [x] Record which validations can be run locally without long training.

### 0.3 Protect current assumptions before edits

- [x] Confirm whether existing checkpoints contain both `model` and `ema` keys.
- [x] Confirm which scripts or README examples assume `model-{milestone}.pt` naming.
- [x] Confirm whether any user-edited documentation files contain wording that should be preserved.

## Phase 1: Fix Hard Execution Failures

### 1.1 Fix GPU/CPU tensor mixing in inference

- [x] Inspect the tensor-device flow around `result`, `gen_mask`, `gen_mask_de_sdf`, `background_mask`, `torch.cat`, and `torch.argmax` in [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py).
- [x] Replace CPU scalar tensor construction with device-aware tensor creation.
- [x] Ensure thresholding operations stay on the same device as `gen_mask`.
- [x] Ensure background-mask construction uses the same device and dtype as the active mask tensor.
- [x] Keep all merge logic on device until the final serialization boundary.
- [x] Audit nearby code for any other implicit CPU tensor creation in the inference post-processing path.

Validation:

- [x] Run a CPU inference smoke check and confirm output files are still produced.
- [x] Run a CUDA inference smoke check and confirm the device mismatch crash is gone.
- [x] Confirm the merged categorical mask output keeps the same shape and file naming contract.

### 1.2 Fix broken checkpoint loading with `map_location`

- [x] Inspect `Trainer.load` in [ddpm/diffusion.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py) and trace every path that constructs a checkpoint location.
- [x] Refactor the function so checkpoint-path resolution happens before `torch.load`.
- [x] Ensure `milestone == -1` still resolves to the latest checkpoint correctly.
- [x] Ensure explicit milestone loading still uses the `results_folder / model-{milestone}.pt` convention.
- [x] Keep CPU and CUDA load behavior identical except for device placement.

Validation:

- [x] Save a tiny checkpoint and load it with `trainer.load(milestone)`.
- [x] Load the same checkpoint with `trainer.load(milestone, map_location='cpu')`.
- [x] Confirm model, EMA, optimizer, and scaler states all restore without error.

### 1.3 Fix the MMWHS MRI "all" training script path

- [x] Inspect [traing_scripts/train_MMWHSMRI_all.sh](/home/estar/TZNEW/DiffAtlas/DiffAtlas/traing_scripts/train_MMWHSMRI_all.sh) and confirm the bad dataset root.
- [x] Verify the actual dataset directory contract against the repository layout and README instructions.
- [x] Update the script path to the real MRI `all` directory.
- [x] Check whether any README examples or related scripts repeat the same wrong path.

Validation:

- [x] Run `bash -n` against the updated script.
- [x] Run the script command far enough to confirm dataset discovery no longer fails on the path.

## Phase 2: Fix Training and Sampling Control Flow

### 2.1 Repair the sampler guidance logic

- [x] Re-read `GaussianDiffusion_Nolatent.__init__`, `p_sample_loop`, and all sampling call sites in [ddpm/diffusion.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py).
- [x] Confirm the current default inference path is guided because `use_guide=True` is the constructor default and [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py) does not override it.
- [x] Replace the current truthiness check with strict boolean branching.
- [x] Add an explicit unguided reverse-denoising branch that still calls `self.p_sample(...)`.
- [x] Keep the guided branch behavior unchanged for the public inference path.
- [x] Remove any silent `None` mode unless a third mode is intentionally documented and validated.

Validation:

- [x] Confirm default inference behavior is still guided after the change.
- [x] Confirm `use_guide=False` now runs a real reverse process instead of returning untouched initialization.
- [x] Confirm both branches complete the full timestep loop without shape or device regressions.

### 2.2 Remove or formalize the unfinished recurrent schedule

- [x] Inspect `p_sample_loop`, `p_sample_loop_v2`, and `p_sample_loop_v4` for `R` and `recurrent` usage.
- [x] Confirm the schedule arrays are constructed and never consumed.
- [x] Decide whether the short-term fix is dead-code removal or full feature completion.
- [x] Follow the current recommendation from [plan.md](/home/estar/TZNEW/DiffAtlas/DiffAtlas/plan.md): remove dead schedule scaffolding unless new evidence requires preserving it.
- [x] If alternate loops remain, clearly mark which are experimental and which path is the supported default.

Validation:

- [x] Use `rg` to confirm removed scheduling variables are no longer referenced.
- [x] Re-run the primary sampling smoke check after cleanup.

### 2.3 Fix trainer step semantics

- [x] Re-read the full `Trainer.train` loop in [ddpm/diffusion.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py).
- [x] Trace the current order of gradient accumulation, optimizer step, scaler update, save scheduling, EMA scheduling, and `self.step` increment.
- [x] Rewrite the loop so `self.step` means completed optimizer updates.
- [x] Ensure save milestones, checkpoint metadata, and log lines all use the same step definition.
- [x] Check whether resume logic assumes the old off-by-one semantics anywhere else in the file.

Validation:

- [x] Run a tiny trainer scenario with a small save interval and confirm the first checkpoint is created at the correct completed-update step.
- [x] Confirm checkpoint metadata stores the actual completed update count.
- [x] Resume from that checkpoint and confirm subsequent saves continue on the correct schedule.

### 2.4 Fix EMA scheduling semantics

- [x] Re-read `step_ema()` in [ddpm/diffusion.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/ddpm/diffusion.py) after redefining `self.step`.
- [x] Decide the exact warmup boundary semantics for `step_start_ema`.
- [x] Ensure EMA copy-vs-average behavior aligns with completed optimizer updates.
- [x] Verify that checkpoint save timing and EMA update timing now agree.

Validation:

- [x] Run a controlled small-step training check with a low `step_start_ema`.
- [x] Confirm EMA starts averaging exactly at the chosen threshold and not one step early or late.

### 2.5 Make inference reproducible when requested

- [x] Inspect every RNG touchpoint in [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py), including startup seeding, per-case seeding, and end-of-case reseeding.
- [x] Add a clear config contract in [test/confs/infer.yaml](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/confs/infer.yaml) for base seed and deterministic mode.
- [x] Remove the entropy reseeding that destroys reproducibility.
- [x] Decide whether multi-sample inference should derive per-trial seeds from a deterministic base-seed formula.
- [x] Keep current stochastic behavior available when deterministic mode is disabled.

Validation:

- [x] Run inference twice with the same seed and compare outputs or reported metrics.
- [x] Run inference with a different seed and confirm stochastic variation is still possible.

## Phase 3: Fix Evaluation Correctness

### 3.1 Load EMA weights by default at inference

- [x] Inspect how checkpoints are loaded in [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py).
- [x] Confirm current code always selects the raw `model` weights.
- [x] Add a config field in [test/confs/infer.yaml](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/confs/infer.yaml) to choose `ema` or `model`.
- [x] Make `ema` the default inference weight key.
- [x] Preserve the existing `module.` prefix stripping logic for both key choices.
- [x] Confirm the change is compatible with existing checkpoints in this repo.

Validation:

- [x] Load one checkpoint with `weight_key=ema`.
- [x] Load the same checkpoint with `weight_key=model`.
- [x] Run a small inference check for both paths and confirm both execute successfully.

### 3.2 Pass real spacing into NSD

- [x] Re-read the affine handling and NSD call site in [test/inference.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/test/inference.py).
- [x] Confirm the dataset provides an affine and that `get_nsd` currently falls back to default spacing.
- [ ] Decide the first implementation rule for spacing extraction from affine.
- [ ] Verify whether current preprocessing preserves a meaningful affine for the cropped and padded `64^3` tensors.
- [x] Defer implementation until Phases 1 and 2 are complete, because this is a metric-correctness issue rather than an execution blocker.

Validation:

- [ ] Confirm the extracted spacing matches the affine-derived voxel scale for a representative CT sample.
- [ ] Confirm the extracted spacing matches the affine-derived voxel scale for a representative MRI sample.
- [ ] Confirm `compute_surface_dice` receives non-default spacing when appropriate.

## Phase 4: Align Scripts, Documentation, and Data Contract

### 4.1 Align training and testing scripts with actual behavior

- [x] Re-read all files under [traing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/traing_scripts) for stale paths, stale milestone assumptions, and naming mismatches.
- [x] Re-read all files under [testing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/testing_scripts) for checkpoint-name, dataset-root, and config mismatches.
- [x] After trainer-step semantics are fixed, update any `train_num_steps` values that only exist to work around the old off-by-one behavior.
- [x] Confirm every script points to real files and directories in the repo.

Validation:

- [x] Run `bash -n` for every script under [traing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/traing_scripts).
- [x] Run `bash -n` for every script under [testing_scripts](/home/estar/TZNEW/DiffAtlas/DiffAtlas/testing_scripts).
- [x] Manually inspect the expanded commands for at least one representative train script and one representative test script.

### 4.2 Align README with the actual repository

- [x] Re-read [README.md](/home/estar/TZNEW/DiffAtlas/DiffAtlas/README.md) after the code-path fixes are done.
- [x] Fix `training_scripts` versus `traing_scripts` naming mismatches.
- [x] Fix any README commands that reference non-existent files or stale checkpoint names.
- [x] Align checkpoint download or usage instructions with the actual `model-{milestone}.pt` loading convention.
- [x] Document any config knobs added during the fixes, especially reproducibility and weight selection.

Validation:

- [x] Verify every command shown in the README points to a real file path in the repository.
- [x] Verify every referenced checkpoint naming example matches what the loader expects.

### 4.3 Clarify the dataset contract

- [x] Re-read [Dataset/MMWHS_Dataset.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/Dataset/MMWHS_Dataset.py) and [Dataset/TS_Dataset.py](/home/estar/TZNEW/DiffAtlas/DiffAtlas/Dataset/TS_Dataset.py) to restate the real preprocessing/data assumptions.
- [x] Document clearly that MMWHS expects precomputed SDF files while TotalSegmentator computes SDF online.
- [x] Decide whether to add an explicit runtime assertion for missing MMWHS SDF artifacts.
- [x] Ensure the README describes the required file layout precisely enough for first-time setup.

Validation:

- [x] Simulate a missing MMWHS SDF artifact and confirm the resulting failure mode is actionable.
- [x] Confirm TotalSegmentator still needs only image and label volumes.

## Phase 5: Reduce Maintenance Burden

### 5.1 Clean up dead or experimental paths

- [x] Audit `p_sample_loop_v2`, `p_sample_loop_v3`, `p_sample_loop_v4`, `p_losses_image_only`, text-conditioning code, and unused geometric helpers for actual references.
- [x] Label still-relevant experimental paths clearly before removing anything.
- [x] Remove dead methods only after the main training and inference paths are stable and validated.
- [x] Keep at most one obvious training path and one obvious inference path as the supported defaults.

Validation:

- [x] Use `rg` to confirm removed or renamed paths are not referenced.
- [x] Re-run the training and inference smoke checks after cleanup.

## Cross-Phase Verification Checklist

### Training behavior

- [x] `self.step` consistently means completed optimizer updates.
- [x] Save cadence, checkpoint metadata, and EMA cadence all agree on the same step definition.
- [x] `trainer.load(..., map_location='cpu')` works on a real saved checkpoint.

### Sampling behavior

- [x] Default inference remains guided.
- [x] `use_guide=False` performs actual denoising.
- [x] No sampling mode silently returns untouched random initialization unless that mode is intentionally documented.

### Evaluation behavior

- [x] Inference can load both EMA and raw model weights.
- [x] Repeated inference with the same deterministic seed produces matching outputs.
- [ ] NSD uses affine-derived spacing once that deferred task is implemented.

### Repository usability

- [x] Every README command points to a real file.
- [x] Every provided shell script references a real path.
- [x] The dataset prerequisites are explicit for both MMWHS and TotalSegmentator.

## Suggested Execution Order

- [x] Complete Phase 0 before any code edits.
- [x] Complete all Phase 1 fixes before touching control-flow refactors.
- [x] Complete sampler and trainer semantics in Phase 2 before changing script step counts.
- [x] Complete Phase 3 only after the pipeline is executable and deterministic enough to compare outputs.
- [x] Complete Phase 4 before calling the project "usable out of the box".
- [x] Complete Phase 5 last, after the supported path is stable.
