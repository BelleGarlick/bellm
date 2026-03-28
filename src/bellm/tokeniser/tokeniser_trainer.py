from collections import defaultdict

from reader import get_dataset
from bellm.tokeniser import Tokeniser

"""Train the tokeniser"""

# TODO Add testing and documentation


# The number of records to scan per batch when counting token pairs
SAMPLE_SIZE_BATCH_SIZE = 500_000
TOP_TOKENS_PER_BATCH = 500
MINI_BATCH_SIZE = 1000  # used ot report the percentage too
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
    dataset = get_dataset()
    tokeniser = Tokeniser().load(f"tokeniser.json")

    from multiprocessing import Pool

    # Cycle through windows of 500_000 records and tokenise on them
    start_offset = 4500000
    # todo change this to represent the batches
    for i in range(1000):
        print(f"Checking: {i * SAMPLE_SIZE_BATCH_SIZE + start_offset}:{(i + 1) * SAMPLE_SIZE_BATCH_SIZE + start_offset}")
        items = list(dataset\
            .skip(i * SAMPLE_SIZE_BATCH_SIZE + start_offset)\
            .take(SAMPLE_SIZE_BATCH_SIZE)["text"])

        bpe_batch_rankings = defaultdict(int)

        with Pool(12) as p:
            # tood use better subsampleing here as we combine with the larger dataset
            for batch_idx in range(0, SAMPLE_SIZE_BATCH_SIZE, MINI_BATCH_SIZE):
                print(f"\r{round(100*batch_idx/SAMPLE_SIZE_BATCH_SIZE, 2)}%", end="")
                batch_items = items[batch_idx:batch_idx + MINI_BATCH_SIZE]

                mini_batch_rankings = p.map(count_frequencies, [(item, tokeniser) for item in batch_items])

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
