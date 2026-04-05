from bellm.dataset.dataset_sources import DatasetSourceHuggingFace


ALLENAI_C4_ENGLISH = DatasetSourceHuggingFace(
    path="allenai/c4",
    name="en",
    max_entries=5_000_000,
)


# List of all datasets that will be used for foundation model training
FOUNDATION_DATASETS = [
    ALLENAI_C4_ENGLISH,
]
