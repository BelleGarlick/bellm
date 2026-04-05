import os
import shutil
from dataclasses import dataclass
from html import escape
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


@dataclass
class DatasetDownloadContext:
    output_path: Path
    max_entries_per_shard: int


@dataclass
class DatasetSourceHuggingFace:
    # todo account for train / test ones

    path: str
    name: str | None = None
    max_entries: int | None = None

    def download(self, context: DatasetDownloadContext):
        # TODO
        # This only really works for the c4 dataset, will need to change it in future
        dataset_id = f"hf_{self.path.replace('/', '_')}_{self.name.replace('/', '_')}"
        dataset_path = context.output_path / dataset_id

        print(f" - {dataset_id}...")

        # Delete the directory if it exists and create a blank one
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Define the hugging face dataset
        dataset = load_dataset(
            self.path,
            self.name,
            split="train",  # todo maybe at somepoint check if this is needed
            streaming=True,
        )

        if self.max_entries is None:
            self.max_entries = dataset.m

        # Iterate through all items, downloading into the shards
        for shard_id, batch_start in tqdm(enumerate(range(0, self.max_entries, context.max_entries_per_shard))):
            items = dataset.skip(batch_start).take(context.max_entries_per_shard)["text"]
            items = [escape(x) for x in items]

            with open(dataset_path / f"{shard_id}.txt", "w") as f:
                f.writelines(items)
