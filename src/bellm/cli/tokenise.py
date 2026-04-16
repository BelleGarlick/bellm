from pydantic import Field

from bellm.tokeniser import Tokeniser
from clpi import BaseClpIModel


COLORS = [
    "\033[38;2;0;0;0;48;2;112;230;255m",
    "\033[38;2;0;0;0;48;2;255;230;112m",
    "\033[38;2;0;0;0;48;2;255;112;166m",
    "\033[38;2;0;0;0;48;2;112;255;151m",
    "\033[38;2;0;0;0;48;2;255;151;112m",
    "\033[38;2;0;0;0;48;2;233;255;112m",
]


class TokeniseCommand(BaseClpIModel):

    tokeniser: str = Field(description="The file path to the tokeniser file")

    ids: bool = Field(default=False, description="If true, the token ids will be shown rather than the tokens")

    def run(self, *args, **kwargs):
        tokeniser = Tokeniser().load(self.tokeniser)

        tokens = tokeniser.tokenize(" ".join(args[1:]))
        items = tokens.token_ids if self.ids else tokens.tokens

        for i, token in enumerate(items):
            text = COLORS[i % len(COLORS)] + str(token) + "\033[0m"
            if self.ids: text += ","
            print(text, end="")
        print()

        print(f"Tokens: {len(tokens.token_ids)}")
