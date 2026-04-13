import json
import random
from pathlib import Path
from threading import Thread

from bellm.dataset.utils.dataset_metadata import DatasetMetadata
from bellm.dataset.utils.utils import load_shard


class ShardLoader(Thread):

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

        self.items = []
        self.idxs = []

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, slice):
        idxs = self.idxs[slice]

        return [self.items[i] for i in idxs]

    def run(self):
        self.items = load_shard(self.path)

        for idx, item in enumerate(self.items):
            self.idxs.append(idx)

        random.shuffle(self.idxs)


class TokeniserDataLoader:

    def __init__(
            self,
            path: Path,
            batch_size: int
    ):
        self.path = path
        self.batch_size = batch_size

        with open(path / "metadata.json") as f:
            self.metadata = DatasetMetadata(**json.load(f))

        self.current_shard_idx = -1
        self.current_idx_in_shard = 0

        self.current_shard = None
        self.next_shard = None

    def __iter__(self):
        self.current_shard_idx = -1
        self.current_idx_in_shard = 0

        self.current_shard = None
        self.next_shard = None
        self.start_loading_next_shard()

        return self

    def __len__(self):
        if self.current_shard is not None:
            return len(self.current_shard) * len(self.metadata.shards)
        if self.next_shard is not None:
            return len(self.next_shard) * len(self.metadata.shards)
        return 0

    def start_loading_next_shard(self):
        self.next_shard = None
        if self.current_shard_idx < len(self.metadata.shards):
            self.next_shard = ShardLoader(
                self.path / self.metadata.shards[self.current_shard_idx + 1].uri
            )
            self.next_shard.start()

    def __next__(self):
        # Attempt to load the next shard if there or kill stop iteration
        if self.current_shard is None or self.current_idx_in_shard >= len(self.current_shard):
            if self.next_shard is not None:
                self.next_shard.join()
            self.current_shard = self.next_shard
            self.current_shard_idx += 1
            self.current_idx_in_shard = 0
            if self.current_shard_idx >= len(self.metadata.shards):
                raise StopIteration
            self.start_loading_next_shard()

        start = self.current_idx_in_shard
        end = self.current_idx_in_shard + self.batch_size
        batch = self.current_shard[start:end]
        self.current_idx_in_shard = end

        return batch
