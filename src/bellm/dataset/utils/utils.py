import json
import os
from pathlib import Path
from typing import List

from bellm.dataset.utils.dataset_metadata import DatasetMetadata


def save_shard(
    output_path: Path,
    shard_lines: List[str],
):
    # Escape things like new line chars
    items = [x.encode("unicode_escape").decode("utf-8") for x in shard_lines]

    # Open the file and save the lines
    with open(output_path, "w+") as f:
        f.write("\n".join(items))


def load_shard(path: Path):
    with open(path, "r") as f:
        shard_lines = f.readlines()

    # Unescape new line chars
    return [x.encode('utf-8').decode('unicode_escape') for x in shard_lines]


def should_redownload(metadata_path: Path, dataset_id: str):
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            current_metadata = DatasetMetadata(**json.load(f))

        if current_metadata.id == dataset_id:
            return False

    return True


def save_dataset_metadata(
    metadata_path: Path,
    output_metadata: DatasetMetadata,
):
    # Save the dataset metadata
    with open(metadata_path, "w+") as f:
        f.write(output_metadata.model_dump_json())