# Inventory Mode (No-Copy Source-Linked Datasets)

Inventory mode lets you run nnU-Net without copying data into a classic `imagesTr/labelsTr` dataset tree.
You provide one inventory JSON that references source files directly.

## What It Does

- Resolves inventory paths (absolute or relative)
- Writes a normalized `dataset.json` to `--cache-dir/DatasetXXX_NAME/`
- Uses `--cache-dir` as both `nnUNet_raw` and `nnUNet_preprocessed`
- Uses `--results-dir` for trained model outputs (training and prediction)

## Required Arguments

Plan / fingerprint / preprocess:

- `--inventory <path_to_inventory.json>`
- `--dataset-id <int>`
- `--dataset-name <name>`
- `--cache-dir <path>`

Training and prediction:

- all of the above, plus `--results-dir <path>`

## Hardware Alignment

Inventory mode does not auto-detect cluster hardware or auto-tune all scaling knobs.
It only normalizes dataset paths and runtime roots.

Align hardware-related settings with your cluster profile:

- Planner/VRAM target: `-pl`, `-gpu_memory_target`
- Fingerprint/preprocess workers: `-npfp`, `-np`
- Training workers: env `nnUNet_def_n_proc`, env `nnUNet_n_proc_DA`
- GPU count used by training: `-num_gpus`

## Quick Start with Defined Resources

Use this pattern when you want a reproducible run with fixed compute limits.
Example budget used below:

- 1 GPU with about 24 GB VRAM
- 12 CPU workers for preprocessing/fingerprint
- 12 training workers (`nnUNet_def_n_proc`, `nnUNet_n_proc_DA`)

```bash
# Pin training worker counts (adjust to your node/job allocation)
export nnUNet_def_n_proc=12
export nnUNet_n_proc_DA=12

# 1) Fingerprint with integrity check
nnUNetv2_extract_fingerprint \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  -npfp 12 \
  --verify_dataset_integrity \
  --clean

# 2) Plan + preprocess with explicit VRAM/CPU targets
nnUNetv2_plan_and_preprocess \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  -gpu_memory_target 24 \
  -npfp 12 \
  -np 12 \
  --clean

# 3) Train fold 0 using 1 GPU
nnUNetv2_train 310 3d_fullres 0 \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  --results-dir /data/.nnunet_results \
  -num_gpus 1
```

Keep `--dataset-id`, `--dataset-name`, `--cache-dir`, and `--results-dir` stable across all stages.

## Inventory JSON Schema

Required top-level keys:

- `channel_names`
- `labels`
- `numTraining`
- `file_ending`
- `dataset`

`dataset` must be a map from case ID to:

- `images`: list of image file paths
- `label`: segmentation file path

Relative paths are resolved relative to the inventory JSON location.

## Minimal Example

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "target": 1},
  "numTraining": 2,
  "file_ending": ".nii.gz",
  "dataset": {
    "case_001": {
      "images": ["./images/case_001_0000.nii.gz"],
      "label": "./labels/case_001.nii.gz"
    },
    "case_002": {
      "images": ["./images/case_002_0000.nii.gz"],
      "label": "./labels/case_002.nii.gz"
    }
  }
}
```

## Commands

Plan + preprocess:

```bash
nnUNetv2_plan_and_preprocess \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache
```

Train:

```bash
nnUNetv2_train 310 3d_fullres 0 \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  --results-dir /data/.nnunet_results
```

Predict (inventory mode):

```bash
nnUNetv2_predict \
  -i /data/test_images \
  -o /data/preds \
  -c 3d_fullres \
  -f 0 1 2 3 4 \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  --results-dir /data/.nnunet_results
```

Predict (model folder mode, no env vars required):

```bash
nnUNetv2_predict_from_modelfolder \
  -i /data/test_images \
  -o /data/preds \
  -m /data/.nnunet_results/Dataset310_MyDataset/nnUNetTrainer__nnUNetPlans__3d_fullres \
  -f 0 1 2 3 4
```

## Edge Cases and Rules

1. Channel count/order must be consistent across all cases
- If `channel_names` has 2 channels, every case must provide exactly 2 entries in `images` in the same semantic order.
- Use `--verify_dataset_integrity` at least once to catch mismatches.

2. Multi-contrast is supported
- nnU-Net supports multi-channel input naturally.
- Example for two contrasts:

```json
"channel_names": {"0": "T1", "1": "T2"},
"dataset": {
  "case_001": {
    "images": [".../case_001_t1.nii.gz", ".../case_001_t2.nii.gz"],
    "label": ".../case_001_seg.nii.gz"
  }
}
```

3. Multi-label is supported, but labels must obey nnU-Net constraints
- `background` must exist and be `0`
- label IDs must be consecutive (`0,1,2,...`)
- Example:

```json
"labels": {"background": 0, "organ": 1, "lesion": 2}
```

4. Region-based labels require `regions_class_order`
- If you use tuple/list regions in `labels`, define `regions_class_order`.

5. Ignore label rules
- If used, `ignore` must be an integer and must be the highest label ID.

6. Natural image reader caveat (`.png/.bmp/.tif`)
- `NaturalImage2DIO` is 2D-only.
- RGB files count as 3 channels from one file. Do not mix grayscale and RGB shapes across cases.

7. Path behavior
- Relative paths are resolved relative to the inventory JSON file.
- Environment variables in paths are expanded.

8. `numTraining` must match `len(dataset)`
- Mismatch is rejected during inventory normalization.

9. Dataset naming
- Prefer simple names like `MyDataset`.
- Resulting runtime dataset name is `DatasetXXX_<sanitized_name>`.

10. Keep identity arguments stable across pipeline stages
- Use the same `--dataset-id`, `--dataset-name`, `--cache-dir`, and `--results-dir` for preprocess, training, and prediction.
- Changing one of these moves nnU-Net to a different runtime dataset/output location.

11. Cache reuse and stale artifacts
- If you change inventory contents (cases, labels, channels), re-run with `--clean` for fingerprint/preprocess.
- Avoid reusing the same dataset ID/name for different data definitions unless you intentionally overwrite cache outputs.

## Practical Validation Checklist

1. Run once with integrity checks:

```bash
nnUNetv2_extract_fingerprint \
  --inventory /data/inventory.json \
  --dataset-id 310 \
  --dataset-name MyDataset \
  --cache-dir /data/.nnunet_cache \
  --verify_dataset_integrity \
  --clean
```

2. Run plan + preprocess.
3. Train.
4. Predict either with inventory mode or with `nnUNetv2_predict_from_modelfolder`.
