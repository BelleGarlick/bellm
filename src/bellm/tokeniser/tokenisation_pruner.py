from collections import defaultdict
from pathlib import Path

from bellm.dataloader.tokeniser_dataloader import TokeniserDataLoader
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


BATCH_SIZE = 10_000

# Limit to 5k pruned token
MAX_TOKENS = 5_000


if __name__ == "__main__":
    dataset = TokeniserDataLoader(Path("/Users/belle/Developer/Belllm/belllm/data/preprocessed/foundation/train"), batch_size=BATCH_SIZE)

    tokeniser = Tokeniser().load(f"tokeniser.json")

    from multiprocessing import Pool

    bpe_rankings = defaultdict(int)

    with Pool(10) as p:
        count = 0
        for batch_idx, batch in enumerate(dataset):
            if count > 500_000:
                break
            print(f"\rTokenising {round(count / len(dataset) * 100, 2)}% complete {count}.", end="")
            count += len(batch)

            # Call to multi-parallelise the map
            rankings = p.map(count_frequencies, [(item, tokeniser) for item in batch])

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
