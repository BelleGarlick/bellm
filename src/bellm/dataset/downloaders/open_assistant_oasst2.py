import json
import os

import shutil
from pathlib import Path

from datasets import load_dataset

from bellm.dataset.utils.utils import save_shard, should_redownload, save_dataset_metadata
from bellm.dataset.utils.dataset_metadata import DatasetMetadata, DatasetShardMetadata

PATH = "OpenAssistant/oasst2"


def oasst_adapter(data):
    items = [x for x in data]
    items_map = {x["message_id"]: {"children": [], "text": x['text'], "lang": x["lang"], "role": x["role"]} for x in items}

    heads = []

    for item in items:
        if item["parent_id"] is None:
            heads.append(items_map[item["message_id"]])
        else:
            item_chain = items_map[item["message_id"]]
            items_map[item["parent_id"]]["children"].append(item_chain)

    # Aim for english convos only
    heads = [x for x in heads if x["lang"] == "en"]

    # Traverse the tree forming the conversations
    conversations = []
    def traverse_head(items, conversation_chain):
        if len(items) == 0:
            conversations.append(conversation_chain)

        for item in items:
            traverse_head(
                item['children'],
                [*conversation_chain, {
                    "message": item["text"],
                    "role": {
                        "prompter": "user",
                        "assistant": "assistant"
                    }[item["role"]]
                }]
            )

    # Trigger breath first search
    traverse_head(heads, [])

    conversations = [json.dumps(x) for x in conversations]

    return conversations


def download_oasst_split(parent_path: Path, split: str):
    dataset_id = f"hf_{PATH.replace('/', '_')}"
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
        split=split,
        streaming=True,
    )

    # Load all dataset items, needed to create the conversation tree
    items = list(dataset)
    items = oasst_adapter(items)

    # Create the shard, here we'll just create one large shard but this could be changed in future
    shard_name = f"0.txt"
    save_shard(output_path / shard_name, items)

    # Add this shard to the metadata
    output_metadata.length += len(items)
    output_metadata.shards.append(DatasetShardMetadata(uri=shard_name, length=len(items)))

    # Save the dataset metadata
    save_dataset_metadata(metadata_path, output_metadata)


def download_oasst(parent_path: Path):
    download_oasst_split(parent_path, "train")
    download_oasst_split(parent_path, "validation")
