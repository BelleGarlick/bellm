import json
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

# TODO Add testing and documentation

PREDEFINED_TOKENS = 6

@dataclass
class Tokenised:

    # The input text
    input_text: str

    # The raw token strings
    tokens: List[str]

    # The raw token ids
    token_ids: List[int]

    # Store the number of times each token was traversed
    token_frequencies: Dict[str, int]


class Tokeniser:

    UNK = 0
    PAD = 1
    NEXT_PAGE = 2
    USER = 3
    ASSISTANT = 4
    REASONING = 5

    def __init__(self):
        self.token_map = {}

    def tokenize(self, text, max_steps=None, max_tokens=None):
        tokenised = [(x, self[x]) for x in text]

        # This stores a mask of the tokens we should try to pair
        indexes_to_check = list(range(len(tokenised)))

        token_frequencies = defaultdict(int)
        for token, _ in tokenised:
            token_frequencies[token] += 1

        steps = 0
        while indexes_to_check:
            checked_map = [0 for _ in range(len(tokenised))]

            for i in sorted(indexes_to_check, reverse=True):
                # can skip indexes where i = 0.
                # not this isn't hte same as indexes_to_check[1:] as this list may not start at 0
                if i == 0: continue

                pair_token_text = tokenised[i-1][0] + tokenised[i][0]
                if pair_token_text in self.token_map and (max_tokens is None or self[pair_token_text] < max_tokens):
                    tokenised[i-1] = pair_token_text, self[pair_token_text]
                    token_frequencies[pair_token_text] += 1

                    # Remove the item from tokens and the checked tokens
                    del tokenised[i]
                    del checked_map[i]

                    # Set surrounding items to now get checked in next round
                    if i > 1: checked_map[i-2] = 1 # todo check if this is needed. i think not
                    checked_map[i-1] = 1
                    if i < len(checked_map) - 1: checked_map[i] = 1

            indexes_to_check = [i for i, v in enumerate(checked_map) if v == 1]

            steps += 1

            if max_steps is not None and steps >= max_steps:
                break

        return Tokenised(
            input_text=text,
            tokens=[x[0] for x in tokenised],
            token_ids=[x[1] for x in tokenised],
            token_frequencies=token_frequencies
        )

    def tokenize_batch(self, items: List[str]) -> List[Tokenised]:
        return [
            self.tokenize(text)
            for text in items
        ]

    def detokenise(self, token_ids):
        inverse = {i: v for v, i in self.token_map.items()}
        inverse[0] = "[UNK]"
        inverse[1] = ""
        inverse[2] = "[NEXT_PAGE]"
        inverse[3] = "[USER]"
        inverse[4] = "[ASSISTANT]"
        inverse[5] = "[REASONING]"

        return [inverse[x] for x in token_ids]

    def add_token(self, token):
        self.token_map[token] = len(self)

    def save(self, path):
        with open(path, "w+") as f:
            f.write(json.dumps(self.token_map, indent=2))

    def load(self, path):
        with open(path, "r") as f:
            self.token_map = json.loads(f.read())
        return self

    def __len__(self):
        return len(self.token_map) + PREDEFINED_TOKENS

    def __contains__(self, x):
        return x in self.token_map

    def __getitem__(self, x):
        return self.token_map.get(x, Tokeniser.UNK)

    def __setitem__(self, x, value):
        self.token_map[x] = value
