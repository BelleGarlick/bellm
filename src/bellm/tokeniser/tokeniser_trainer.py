from collections import defaultdict
from pathlib import Path

from bellm.dataloader.tokeniser_dataloader import TokeniserDataLoader
from bellm.tokeniser import Tokeniser

from multiprocessing import Pool

"""Train the tokeniser"""

# TODO Add testing and documentation
# todo this doesn't work if the batch size is larger than each partition file


# The number of records to scan per batch when counting token pairs
BATCH_SIZE = 100_000
TOP_TOKENS_PER_BATCH = 500
MAX_TOKENS = 20_000


def count_frequencies(input):
    text_item, tokeniser = input

    bpe_rankings = defaultdict(int)

    # Tokenise the text with current best tokeniser bpe's
    tokenised = tokeniser.tokenize(text_item)

    # Loop through count how many instances of each pairing exist
    for i in range(0, len(tokenised.tokens)):
        ctoken = tokenised.tokens[i]
        exists = ctoken in tokeniser.token_map
        if not exists:
            bpe_rankings[ctoken] += 1

        if i > 1 and exists:
            pair_token_text = tokenised.tokens[i - 1] + tokenised.tokens[i]
            bpe_rankings[pair_token_text] += 1

    return bpe_rankings


if __name__ == "__main__":
    tokeniser = Tokeniser()

    dataset = TokeniserDataLoader(Path("/Users/belle/Developer/Belllm/belllm/data/preprocessed/foundation/train"), batch_size=BATCH_SIZE)

    while True:
        for batch_idx, batch in enumerate(dataset):
            print(f"\r{round(100*(batch_idx * len(batch))/len(dataset), 2)}%", end="")
            bpe_batch_rankings = defaultdict(int)

            with Pool(12) as p:

                mini_batch_rankings = p.map(count_frequencies, [(item, tokeniser) for item in batch])

                for ranking in mini_batch_rankings:
                    for key, value in ranking.items():
                        bpe_batch_rankings[key] += value

            print("\rFinished batches")

            # For the most part the top rankings dont change too much as tokens get added.
            # Which effectively means that although they won't be in the optimal order,
            #  are likely to all loosely get added based on frequency. So we take the top
            #  tokens and then rerun
            top_tokens = sorted(
                bpe_batch_rankings.items(),
                key=lambda x: x[1],
                reverse=True
            )[:TOP_TOKENS_PER_BATCH]
            for (token, freq) in top_tokens:
                # Register the most frequent pairing in the tokeniser
                print("Adding token:", f"'{token}' ({len(tokeniser)})", freq)
                tokeniser.add_token(token)

            # if len(tokeniser) % 100 == 0:
            tokeniser.save(f"tokeniser.json")
            print(f"Saved token set: {len(tokeniser)}")

            print(f"Token frequency range: {top_tokens[0][1] - top_tokens[-1][1]}")
