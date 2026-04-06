from pathlib import Path

from bellm.dataset.dataset_sources import DatasetDownloadContext
from bellm.dataset.foundation_datasets import FOUNDATION_DATASETS
from bellm.dataset.instruction_datasets import INSTRUCTION_DATASETS


ROOT = Path("/Users/belle/Developer/Belllm/belllm/downloaded_datasets")


# todo dont download ones which already exist


def download_foundation_model_datasets():
    foundation_model_context = DatasetDownloadContext(
        output_path=ROOT / "foundation_datasets",
        max_entries_per_shard=500_000,
    )

    print("Downloading foundation datasets")
    for dataset in FOUNDATION_DATASETS:
        dataset.download(foundation_model_context)


def download_instruction_model_datasets():
    download_context = DatasetDownloadContext(
        output_path=ROOT / "instruction_datasets",
        max_entries_per_shard=1_000_000_000,  # kinda a hack cos of the way the tree builds
    )

    print("Downloading instruction datasets")
    for dataset in INSTRUCTION_DATASETS:
        dataset.download(download_context)


if __name__ == "__main__":
    # download_foundation_model_datasets()
    download_instruction_model_datasets()
