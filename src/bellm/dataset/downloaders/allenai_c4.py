import os
import shutil
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from bellm.dataset.utils.utils import save_shard, should_redownload, save_dataset_metadata
from bellm.dataset.utils.dataset_metadata import DatasetMetadata, DatasetShardMetadata

PATH = "allenai/c4"
NAME = "en"
MAX_ITEMS_PER_SHARD = 100_000


def download_c4_english_train(
    parent_path: Path,
    split: str,
    max_length: int
):
    dataset_id = f"hf_{PATH.replace('/', '_')}_{NAME}"
    output_path = parent_path / split / dataset_id
    metadata_path = output_path / "metadata.json"

    dataset_split_id = f"{dataset_id}_{split}"
    print(f" - {dataset_split_id}...")

    if not should_redownload(metadata_path, dataset_split_id):
        return

    if os.path.exists(metadata_path):
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    output_metadata = DatasetMetadata(id=dataset_split_id)

    dataset = load_dataset(
        PATH,
        NAME,
        split=split,
        streaming=True,
    )

    items = dataset.take(max_length)['text']

    shard_id = 0
    current_shard = []

    # Iterate through all items, downloading into the shards
    for item in tqdm(items):
        current_shard.append(item)

        if len(current_shard) >= MAX_ITEMS_PER_SHARD:
            shard_name = f"{shard_id}.txt"
            save_shard(output_path / shard_name, current_shard)

            # Add this shard to the metadata
            output_metadata.length += len(current_shard)
            output_metadata.shards.append(DatasetShardMetadata(uri=shard_name, length=len(current_shard)))

            shard_id += 1
            current_shard = []

    save_dataset_metadata(metadata_path, output_metadata)

    # todo append the dataset metadata to the parent set.


def download_c4(path: Path):
    download_c4_english_train(
        path,
        "train",
        5_000_000
    )
    download_c4_english_train(
        path,
        "validation",
        500_000
    )
