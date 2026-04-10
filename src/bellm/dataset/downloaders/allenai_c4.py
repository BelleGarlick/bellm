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

    # Iterate through all items, downloading into the shards
    for shard_id, batch_start in tqdm(enumerate(range(0, max_length, MAX_ITEMS_PER_SHARD))):
        items = dataset.skip(batch_start).take(MAX_ITEMS_PER_SHARD)
        items = [x['text'] for x in items]

        shard_name = f"{shard_id}.txt"
        save_shard(output_path / shard_name, items)

        # Add this shard to the metadata
        output_metadata.length += len(items)
        output_metadata.shards.append(DatasetShardMetadata(uri=shard_name, length=len(items)))

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
