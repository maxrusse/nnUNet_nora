import copy
import json
import os
import re
import sys
from typing import Dict, Tuple

from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json


_REQUIRED_KEYS = ("channel_names", "labels", "numTraining", "file_ending", "dataset")


def make_dataset_name(dataset_id: int, dataset_name: str) -> str:
    if dataset_name.startswith("Dataset"):
        expected_prefix = f"Dataset{dataset_id:03d}_"
        if not dataset_name.startswith(expected_prefix):
            raise ValueError(
                f"dataset_name starts with 'Dataset' but does not match dataset_id {dataset_id}. "
                f"Expected prefix '{expected_prefix}', got '{dataset_name}'."
            )
        return dataset_name
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", dataset_name).strip("_")
    if not sanitized:
        raise ValueError("dataset_name is empty after sanitization.")
    return f"Dataset{dataset_id:03d}_{sanitized}"


def _resolve_path(path: str, base_dir: str) -> str:
    expanded = os.path.expandvars(path)
    if not os.path.isabs(expanded):
        expanded = os.path.join(base_dir, expanded)
    return os.path.abspath(expanded)


def load_and_normalize_inventory(inventory_file: str) -> Dict:
    inventory_file = os.path.abspath(inventory_file)
    with open(inventory_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"Inventory JSON is missing required keys: {missing}. Required keys are {_REQUIRED_KEYS}."
        )

    if not isinstance(data["dataset"], dict) or len(data["dataset"]) == 0:
        raise ValueError("Inventory JSON key 'dataset' must be a non-empty object.")

    base_dir = os.path.dirname(inventory_file)
    normalized = copy.deepcopy(data)

    for case_id, case in normalized["dataset"].items():
        if not isinstance(case, dict):
            raise ValueError(f"Case '{case_id}' must be a JSON object.")
        if "images" not in case or "label" not in case:
            raise ValueError(f"Case '{case_id}' must define both 'images' and 'label'.")
        if not isinstance(case["images"], list) or len(case["images"]) == 0:
            raise ValueError(f"Case '{case_id}' must define a non-empty 'images' list.")

        case["images"] = [_resolve_path(str(i), base_dir) for i in case["images"]]
        case["label"] = _resolve_path(str(case["label"]), base_dir)

    if int(normalized["numTraining"]) != len(normalized["dataset"]):
        raise ValueError(
            f"numTraining={normalized['numTraining']} does not match number of dataset entries="
            f"{len(normalized['dataset'])}."
        )

    return normalized


def prepare_inventory_dataset(
    inventory_file: str,
    cache_dir: str,
    dataset_id: int,
    dataset_name: str,
) -> Tuple[str, str, str]:
    dataset_name = make_dataset_name(dataset_id, dataset_name)
    cache_dir = os.path.abspath(cache_dir)
    dataset_dir = join(cache_dir, dataset_name)
    maybe_mkdir_p(dataset_dir)

    normalized = load_and_normalize_inventory(inventory_file)
    normalized["name"] = dataset_name
    normalized["numTraining"] = len(normalized["dataset"])

    dataset_json_file = join(dataset_dir, "dataset.json")
    save_json(normalized, dataset_json_file, sort_keys=False)
    return dataset_name, dataset_dir, dataset_json_file


def set_runtime_roots(raw_root: str = None, preprocessed_root: str = None, results_root: str = None) -> None:
    if raw_root is not None or preprocessed_root is not None or results_root is not None:
        # Child processes spawned by multiprocessing may not receive --inventory in argv.
        # This suppresses paths.py startup warnings in inventory mode across workers.
        os.environ['NNUNET_SUPPRESS_PATH_WARNINGS'] = '1'

    def _set_var(var_name: str, value: str):
        if value is None:
            return
        value = os.path.abspath(value)
        os.environ[var_name] = value
        for module_name, module in sys.modules.items():
            if module is None or not module_name.startswith("nnunetv2"):
                continue
            if hasattr(module, var_name):
                setattr(module, var_name, value)

    _set_var("nnUNet_raw", raw_root)
    _set_var("nnUNet_preprocessed", preprocessed_root)
    _set_var("nnUNet_results", results_root)
