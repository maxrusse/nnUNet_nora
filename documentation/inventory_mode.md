# Inventory Mode (No-Copy Source-Linked Datasets)

Inventory mode lets you run nnU-Net without copying data into `imagesTr/labelsTr` under a separate raw dataset tree.
Instead, you pass an inventory JSON that directly references your original files.

## Required Arguments

For planning/preprocessing:

- `--inventory <path_to_inventory.json>`
- `--dataset-id <int>`
- `--dataset-name <name>`
- `--cache-dir <path>`

For training:

- all of the above, plus `--results-dir <path>`

`--cache-dir` contains:

- `DatasetXXX_NAME/dataset.json` (resolved inventory)
- fingerprint, plans and preprocessed outputs

## Inventory JSON Requirements

The inventory JSON must contain:

- `channel_names`
- `labels`
- `numTraining`
- `file_ending`
- `dataset`

`dataset` is a map of case ID to:

- `images`: list of image file paths
- `label`: segmentation file path

Relative paths are resolved relative to the inventory file location.

## Example

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

## Example Commands

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
