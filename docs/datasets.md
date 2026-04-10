# Datasets

> Please note, this doc is written with v1 in mind; subsequent versions may change.

This doc contains an overview for the datasets used in the bellm training. 
There are numerous stages for training the bellm model (tokenising, foundation model training, instruction training, rl training), and each stage requires different datasets and different formats. 
This document should be able to explain all of that.

## Format
There are a couple of dataset formats required for E2E training. 
Each dataset format is also then split up into a series of shards of data with a metadata file which contains information about the shards such as their path, number of items, etc. 
This repo has a series of datasets that can be downloaded and preprocess, but as long as the format of the dataset is the same, then you can easily add your own datasets to train on.

**Sharding**  
Each of the datasets is split up into a series of shards of the data, each shard containing a subset of the dataset.
```
- <foundation|instruction>/
    - <train|val>/
        - metadata.json
        - shard1.txt
        - shard2.txt
        - shard3.txt
```

**Dataset Metadata**
Each dataset contains a series of metadata files. Which takes the format of the following:
```javasript
{
  "id": string,  # id of the dataset which will be referrenced in logs
  "length": int,  # total length of the dataset
  "shards": [
    {
      "uri": string,  # uri to the shard file relative to the metadata path.
      "length": int,  # number of items in this shard
    }
  ]
}
```

When training or running and process in bellm, you pass the relevant metadata file for the dataset being trained on.


### Foundational & Tokenization Formats
Both tokeniser and foundational model use the same dataset.
This format comprises a series of shards where each shard is a text file. 
Each line of the text file represents a separate document whereby each document's newline char is escaped and thus is contained on it's own line.

> Note: For v2 onwards this may change as we incorperate multi-modalities

### Instruction & Reasoning Formats
For fine-tuning the model on instruction and reasoning, we use a similar format to before.
However, each line in the text file contains a json entry (not plain text).

Conversation format:
```json
[
  {"role": "system|assistant|user|reasoning", "message": "text..."},
  {"role": "system|assistant|user|reasoning", "message": "text..."},
  ...
]
```

Example Shard format:
```text
[{"role": "system", "message": "..."}, {"role": "assistant", "message": "..."}, ...]
[{"role": "system", "message": "..."}, {"role": "assistant", "message": "..."}]
[{"role": "system", "message": "..."}, {"role": "assistant", "message": "..."}, ..., ...]
...
```

> Note: For v2 onwards this may change as we incorperate multi-modalities

### RL  
TBD

## Downloading the Datasets

----

... whilst you can create your own dataset, you can also download preselected ones. This are what are used for the current version of bellm.

There are numerous stages for training the bellm model, and each stage requires different datasets. Each one requires two stages. Downloading the dataset, then processing the dataset. These two stages work to first pull in and download all the data from the various sources they're pulled from. Then they get preprocessed to shard them up into subsets which can later be used in the training process.

**Step 1: Download the individual datasets**  
Step 1 for this step is to download the various foundation datasets the bellm is trained on. In doing so, metadata about the dataset is downloaded which describes the download which will be utilised in the preprocessing step.

Currently, all the data will be downloaded. In future, it may be useful to add a way to download subsets or percentages, but currently it will download everything.

To trigger the download, run: 
```shell
scripts/download-datasets.sh <dataset path>
```
This will download the various datasets to the given download path. This path will only contain the downloaded datasets.


**Step 2: Preprocess datasets**  
Once the dataset is downloaded, it then needs to be processed. 
This step takes all the individually downloaded datasets and shuffles+combines them together into shards that can be more easily handled during training.

```shell
scripts/preprocess-datasets.sh <downloaded datasets input path> <processed datasets output path>
```

> Note: This step may take a very long time as it has to combine and shuffle all the datasets.

```text

## Datasets
### allenai/c4
This dataset contains a load of assorted text documents. 
This makes it beneficial for pretraining the foundational model.
Currently, the length of the dataset is limited, but future versions this can be modified.

### chat/instruction/reasoning dataset


## Adding more datasets
# todo write-up, this refers to adding ones officially in bellm that get included in the download
How to code it up, must abide by the code
Choosing the weighting
```

## Future
Future work for the dataset preperation cycle:
- Multi-modal data: This will be down to allow for adding multiple types of data into the model which can then output text somehow. This will require a bit of a rework of the dataset format.
- More datasets: add in more datasets to improve different benchmarks / learning capabilities.
- Have a way to train on different tool usage.
- Weight the datasets when processing them
- Process the datasets to remove bias/racism etc
