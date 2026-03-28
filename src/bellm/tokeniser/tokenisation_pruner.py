from collections import defaultdict

from reader import get_dataset
from bellm.tokeniser import Tokeniser


"""
This module will tokenise the corpus and rank all the tokens based on their frequencies
then discard the least frequent tokens that may get included based on how the tokeniser 
takes large batches of tokens per sweep. This allows tokeniser to remain compact and 
representative of the corpus whilst also training relatively fast.

This pruner takes in a tokeniser.json file and then prune infrequent tokens.
"""

# TODO Add testing and documentation

def count_frequencies(input):
    text_item, tokeniser = input

    # Return the frequencies to which items were traversed
    return tokeniser.tokenize(text_item).token_frequencies


TOTAL_SAMPLES = 5_000_000

BATCH_SIZE = 1000

# Limit to 15k pruned token
MAX_TOKENS = 15_000


if __name__ == "__main__":
    dataset = get_dataset()
    tokeniser = Tokeniser().load(f"tokeniser.json")

    from multiprocessing import Pool


    bpe_rankings = defaultdict(int)

    with Pool(10) as p:
        for batch in range(0, TOTAL_SAMPLES, BATCH_SIZE):
            print(f"\rTokenising {round(batch / TOTAL_SAMPLES * 100, 2)}% complete.", end="")
            items = dataset.skip(batch).take(BATCH_SIZE)["text"]

            # Call to multi-parallelise the map
            rankings = p.map(count_frequencies, [(item, tokeniser) for item in items])

            # Combing rankings from this batch with the full batch
            for ranking in rankings:
                for key, value in ranking.items():
                    bpe_rankings[key] += value

    print("\rCompleted Tokenization")

    new_tokeniser = Tokeniser()
    top_tokens = sorted(bpe_rankings.items(), key=lambda x: x[1], reverse=True)[:MAX_TOKENS]
    for (token, rank) in top_tokens:
        new_tokeniser.add_token(token)
        print(f"{token}: {rank}")

    new_tokeniser.save(f"tokeniser-pruned.json")
