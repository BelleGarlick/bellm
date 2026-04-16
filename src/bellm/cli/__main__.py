import sys

from bellm.cli.tokenise import TokeniseCommand
from bellm.cli.training import FoundationModelTrainingConfig
from clpi import parse


if __name__ == '__main__':
    parse(sys.argv[1:], {
        "tokenise": TokeniseCommand,
        "train": {
            "foundation": FoundationModelTrainingConfig
        }
    })
