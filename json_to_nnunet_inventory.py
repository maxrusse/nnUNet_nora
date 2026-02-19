#!/usr/bin/env python3
"""
Minimal converter:
- input: NORA-style JSON with inputs.val (or top-level val)
- per case: first .nii -> image, first .nii.gz -> label
- output: one nnU-Net inventory JSON
"""

import argparse
import json
from pathlib import Path


def read_val(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if "val" in raw:
        val = raw["val"]
    elif "inputs" in raw:
        inputs = raw["inputs"]
        if isinstance(inputs, str):
            inputs = json.loads(inputs)
        val = inputs.get("val")
    else:
        val = None
    if not isinstance(val, list) or len(val) < 1:
        raise ValueError("Could not find valid 'val' list in input JSON.")
    return val


def collect_paths(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, list):
        out = []
        for y in x:
            out.extend(collect_paths(y))
        return out
    return []


def case_id_from_image(image_path: str, idx: int) -> str:
    parts = Path(image_path).parts
    if len(parts) >= 5:
        return f"{parts[-4]}_{parts[-3]}"
    return f"case_{idx:04d}"


def pick_case_paths(val, idx: int):
    candidates = []
    for group_i, group in enumerate(val):
        if not isinstance(group, list):
            raise ValueError(f"val[{group_i}] must be a list.")
        if idx >= len(group):
            raise ValueError(f"val[{group_i}] has no entry for case index {idx}.")
        candidates.extend(collect_paths(group[idx]))

    image = next((p for p in candidates if p.endswith(".nii") and not p.endswith(".nii.gz")), None)
    label = next((p for p in candidates if p.endswith(".nii.gz")), None)
    if image is None or label is None:
        raise ValueError(
            f"case {idx}: need at least one .nii image and one .nii.gz label, found paths={candidates}"
        )
    return image, label


def build_inventory(val, channel_name: str, file_ending: str):
    num_cases = len(val[0]) if isinstance(val[0], list) else 0
    if num_cases == 0:
        raise ValueError("val[0] is empty or invalid.")

    dataset = {}
    for idx in range(num_cases):
        image, label = pick_case_paths(val, idx)
        cid = case_id_from_image(image, idx)
        if cid in dataset:
            cid = f"{cid}_{idx:04d}"
        dataset[cid] = {"images": [image], "label": label}

    return {
        "channel_names": {"0": channel_name},
        "labels": {"background": 0, "target": 1},
        "numTraining": len(dataset),
        "file_ending": file_ending,
        "dataset": dataset,
    }


def main():
    ap = argparse.ArgumentParser(description="Minimal NORA JSON -> nnU-Net inventory JSON converter.")
    ap.add_argument("--input-json", required=True, help="Input JSON with inputs.val or val.")
    ap.add_argument("--output-json", required=True, help="Output inventory JSON path.")
    ap.add_argument("--channel-name", default="CT", help="channel_names['0'] value (default: CT).")
    ap.add_argument("--file-ending", default=".nii.gz", help="dataset file_ending (default: .nii.gz).")
    ap.add_argument(
        "--skip-existence-check",
        action="store_true",
        help="Skip checking if image/label files exist.",
    )
    args = ap.parse_args()

    input_path = Path(args.input_json).resolve()
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    val = read_val(input_path)
    inv = build_inventory(val, args.channel_name, args.file_ending)

    if not args.skip_existence_check:
        missing = []
        for v in inv["dataset"].values():
            if not Path(v["images"][0]).exists():
                missing.append(v["images"][0])
            if not Path(v["label"]).exists():
                missing.append(v["label"])
        if missing:
            miss_file = output_path.with_suffix(".missing.txt")
            miss_file.write_text("\n".join(missing) + "\n", encoding="utf-8")
            raise FileNotFoundError(f"Missing {len(missing)} files. See {miss_file}")

    output_path.write_text(json.dumps(inv, indent=2), encoding="utf-8")
    print(f"Wrote: {output_path}")
    print(f"Cases: {inv['numTraining']}")


if __name__ == "__main__":
    main()
