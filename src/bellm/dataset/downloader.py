from pathlib import Path

from bellm.dataset.dataset_sources import DatasetDownloadContext
from bellm.dataset.foundation_datasets import FOUNDATION_DATASETS


def download_foundation_model_datasets():
    foundation_model_context = DatasetDownloadContext(
        output_path=Path("/Users/belle/Developer/Belllm/belllm/downloaded_datasets") / "foundation_datasets",
        max_entries_per_shard=500_000,
    )

    print("Downloading foundation datasets")
    for dataset in FOUNDATION_DATASETS:
        dataset.download(foundation_model_context)


if __name__ == "__main__":
    download_foundation_model_datasets()
