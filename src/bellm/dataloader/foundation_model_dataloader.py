import json
import random
from pathlib import Path
from threading import Thread

import numpy as np

from bellm.dataset.utils.dataset_metadata import DatasetMetadata
from bellm.dataset.utils.utils import load_shard
from bellm.tokeniser import Tokeniser


class ShardLoader(Thread):

    def __init__(self, path: Path, tokeniser: Tokeniser, input_context_length, output_context_length):
        super().__init__()
        self.path = path
        self.tokeniser = tokeniser

        self.input_context_length = input_context_length
        self.output_context_length = output_context_length

        self.items = []
        self.idxs = []

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, slice):
        idxs = self.idxs[slice]

        xs = []
        ys = []

        for item_idx, split_idx in idxs:
            line = self.items[item_idx]
            x, y = line[:split_idx], line[split_idx:]

            y_is_truncated = len(y) >= self.input_context_length

            x = x[-self.input_context_length:]
            y = y[:self.output_context_length]

            if y_is_truncated:
                y[-1] = Tokeniser.NEXT_PAGE

            x += [self.tokeniser.PAD] * (self.input_context_length - len(x))
            y += [self.tokeniser.PAD] * (self.output_context_length - len(y))

            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def run(self):
        lines = load_shard(self.path)[:10000]
        tokenised_items = self.tokeniser.tokenize_batch(lines)
        self.items = [x.token_ids for x in tokenised_items]

        for idx, item in enumerate(self.items):
            self.idxs.append((idx, random.randint(0, len(item) - 1)))

        random.shuffle(self.idxs)


class FoundationDataLoader:

    def __init__(
            self,
            path: Path,
            batch_size: int,
            tokeniser: Tokeniser,
            input_context_length,
            output_context_length
    ):
        self.path = path
        self.batch_size = batch_size
        self.tokeniser = tokeniser

        self.input_context_length = input_context_length
        self.output_context_length = output_context_length

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
                self.path / self.metadata.shards[self.current_shard_idx + 1].uri,
                self.tokeniser,
                self.input_context_length,
                self.output_context_length
            )
            self.next_shard.start()

    def __next__(self):
        # Attempt to load the next shard if there or kill stop iteration
        if self.current_shard is None or self.current_idx_in_shard >= len(self.current_shard):
            self.next_shard.join()
            self.current_shard = self.next_shard
            self.current_shard_idx += 1
            self.current_idx_in_shard = 0
            if self.current_shard_idx > len(self.metadata.shards):
                raise StopIteration
            self.start_loading_next_shard()

        start = self.current_idx_in_shard
        end = self.current_idx_in_shard + self.batch_size
        batch = self.current_shard[start:end]
        self.current_idx_in_shard = end

        return batch


