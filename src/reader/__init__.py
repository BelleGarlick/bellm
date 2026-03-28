from datasets import load_dataset

def get_dataset():
    return load_dataset(
        "allenai/c4",
        "en",
        # data_files="dataset/*.json.gz",
        split="train",
        streaming=True,
    )

#
# if __name__ == "__main__":
#     dataset.save_to_disk("dataset/c4")
