import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from bellm.dataset.utils.dataset_metadata import DatasetMetadata, DatasetShardMetadata
from bellm.dataset.utils.utils import load_shard, save_shard, save_dataset_metadata

ITEMS_PER_SHARD = 20_000


@dataclass
class DatasetShardRequest:
    dataset: str
    shard_id: str
    shard_item_idx: int


def preprocess_staged_sets(dataset_id, input_path, output_path):
    print(f" - {dataset_id}")

    output_path.mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_path / "metadata.json"):
        return

    # Define the metadata object for the combined output
    output_metadata = DatasetMetadata(id=dataset_id)

    subdirs = os.listdir(input_path)
    subdirs = [x for x in subdirs if os.path.exists(input_path / x / "metadata.json")]

    items: List[DatasetShardRequest] = []

    # create the full list of indexes
    for dataset in subdirs:
        with open(input_path / dataset / "metadata.json", "r") as f:
            metadata = DatasetMetadata(**json.load(f))

        # todo account for weighting and under/oversampling

        # Create dataset samples for each of the dataset shards
        dataset_samples = []
        for shard in metadata.shards:
            dataset_samples += [DatasetShardRequest(
                dataset=dataset,
                shard_id=shard.uri,
                shard_item_idx=idx
            ) for idx in range(shard.length)]

        items += dataset_samples

    # Shuffle the items for randomly sharding them
    random.shuffle(items)

    # split the list of items into shard and load up those shards into the aggregated shards
    shard_count = 0
    for shard_idx in range(0, len(items), ITEMS_PER_SHARD):
        print(f"\r{shard_idx/len(items)*100:.2f}% Shard: {shard_count}", end="")
        shard_samples = items[shard_idx:shard_idx + ITEMS_PER_SHARD]

        shard_outputs = []
        while shard_samples:
            # Get the first item in the array from this sample
            current_sample = shard_samples[0]

            # Find all samples from the same sample to reduce io ops
            current_shard_samples = [x for x in shard_samples if x.dataset == current_sample.dataset and x.shard_id == current_sample.shard_id]

            # Load the shard and extract the items from it into the shard output
            shard_load_shard = load_shard(input_path / current_sample.dataset / current_sample.shard_id)
            for shard_sample_idx in current_shard_samples:
                shard_outputs.append(shard_load_shard[shard_sample_idx.shard_item_idx])

            # filter out shard sampled from this to remove the items
            shard_samples = [x for x in shard_samples if not (x.dataset == current_sample.dataset and x.shard_id == current_sample.shard_id)]

            # Log the current progress
            print(f"\r{(shard_idx + len(shard_outputs))/len(items)*100:.2f}% Shard: {shard_count}", end="")

        # Need to reshuffle the shard items since they were basically sorted when loading the shard items
        random.shuffle(shard_outputs)

        # Save the aggregated shard output
        shard_name = f"{shard_count}.txt"
        save_shard(output_path / shard_name, shard_outputs)
        output_metadata.length += len(shard_outputs)
        output_metadata.shards.append(DatasetShardMetadata(uri=shard_name, length=len(shard_outputs)))
        shard_count += 1

    save_dataset_metadata(output_path / "metadata.json", output_metadata)
    print(f"\rDone" + " " * 20)


def process_dataset(input_path: Path, output_path: Path):
    """Process the different splits of the dataset."""
    preprocess_staged_sets(
        "foundation-train",
        input_path / "foundation" / "train",
        output_path / "foundation" / "train",
    )
    preprocess_staged_sets(
        "foundation-validation",
        input_path / "foundation" / "validation",
        output_path / "foundation" / "validation",
    )
    preprocess_staged_sets(
        "instruction-train",
        input_path / "instruction" / "train",
        output_path / "instruction" / "train",
    )
    preprocess_staged_sets(
        "instruction-validation",
        input_path / "instruction" / "validation",
        output_path / "instruction" / "validation",
    )


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("No source/destination path argument provided")
        sys.exit(1)

    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    confirmation = input(f"Confirm source destination: {source_path}. [Y/n] ").lower()
    if confirmation != "y":
        print("Aborting.")
        sys.exit(0)

    confirmation = input(f"Confirm output destination: {output_path}. [Y/n] ").lower()
    if confirmation != "y":
        print("Aborting.")
        sys.exit(0)


    output_path.mkdir(parents=True, exist_ok=True)

    process_dataset(source_path, output_path)
