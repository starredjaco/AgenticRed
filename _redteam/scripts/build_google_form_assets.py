#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


def parse_generation_key(filename: str) -> Tuple[int, int, str]:
    m = re.match(r"^gen_(\d+)_", filename)
    if not m:
        return (2, 10**9, filename)
    token = m.group(1)
    if token == "initial":
        return (0, -1, filename)
    return (1, int(token), filename)


def load_entries(input_dir: Path) -> List[Dict]:
    files = sorted(input_dir.glob("*.json"), key=lambda p: parse_generation_key(p.name))
    all_entries: List[Dict] = []
    global_order = 1

    for file_order, path in enumerate(files, start=1):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, got {type(data).__name__}")

        for index_in_file, obj in enumerate(data):
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}[{index_in_file}], got {type(obj).__name__}")

            behavior = obj.get("behavior", "")
            generation = obj.get("generation", "")
            jailbreak = obj.get("jailbreak", None)
            strongreject = obj.get("strongreject", None)

            entry_id = f"E{global_order:05d}"
            all_entries.append(
                {
                    "order": global_order,
                    "entry_id": entry_id,
                    "file_name": path.name,
                    "file_order": file_order,
                    "index_in_file": index_in_file,
                    "behavior": behavior,
                    "generation": generation,
                    "jailbreak": jailbreak,
                    "strongreject": strongreject,
                }
            )
            global_order += 1

    return all_entries


def write_dataset(entries: List[Dict], output_dir: Path) -> Path:
    dataset_path = output_dir / "google_form_dataset.csv"
    fieldnames = [
        "order",
        "entry_id",
        "file_name",
        "file_order",
        "index_in_file",
        "behavior",
        "generation",
        "jailbreak",
        "strongreject",
    ]

    with dataset_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entries:
            writer.writerow(row)

    return dataset_path


def write_ground_truth(entries: List[Dict], output_dir: Path) -> Path:
    gt_path = output_dir / "google_form_ground_truth.csv"
    fieldnames = ["order", "entry_id", "jailbreak"]

    with gt_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in entries:
            writer.writerow(
                {
                    "order": row["order"],
                    "entry_id": row["entry_id"],
                    "jailbreak": row["jailbreak"],
                }
            )

    return gt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ordered Google Form assets from redteam JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing redteam JSON result files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write CSV assets",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    entries = load_entries(args.input_dir)

    dataset_path = write_dataset(entries, args.output_dir)
    gt_path = write_ground_truth(entries, args.output_dir)

    print(f"Wrote {len(entries)} entries")
    print(f"Dataset: {dataset_path}")
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()
