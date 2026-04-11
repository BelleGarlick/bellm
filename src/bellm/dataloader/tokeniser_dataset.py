import os
import random
from pathlib import Path
from threading import Thread


class ShardLoader(Thread):

    def __init__(self, path: Path):
        super().__init__()
        self.path = path

        self.lines = []
        self.idxs = []

    def __getitem__(self, slice):
        idxs = self.idxs[slice]
        return self.lines[idxs]

    def run(self):
        with open(self.path) as f:
            self.lines = f.readlines()

        # Create an idx list that gets shuffled
        self.idxs = list(range(len(self.lines)))
        random.shuffle(self.idxs)


class TokeniserDataLoader:

    def __init__(
            self,
            path: Path,
            batch_size: int
    ):
        self.path = path
        self.batch_size = batch_size

        self.shards_paths = os.listdir(self.path)

        # todo async load a thread
        self.next_item_loader = None

        self.current_shard_idx = 0
        self.current_batch_idx = 0

    def __iter__(self):
        self.current_shard_idx = 0
        self.current_batch_idx = 0

        self.load_next_item()

    def load_next_item(self):
        self.next_item_loader = ShardLoader(self.path / self.shards_paths[self.current_shard_idx])
        self.current_shard_idx += 1

    def __next__(self):
        if self.next_item_loader is not None:
            self.next_item_loader.join()

        start = self.current_batch_idx
        end = self.current_batch_idx + self.batch_size
        batch = self.next_item_loader[start:end]  # todo move the batch accessing to the shard so that the tokenisation can be done async ahead of time
        self.current_batch_idx = end

        if end > len(self.next_item_loader):
            self.current_batch_idx = 0
            if not self.load_next_item():
                raise StopIteration

        return batch


