from fastapi import FastAPI
from pydantic import BaseModel

from bellm.tokeniser import Tokeniser

app = FastAPI()

tokeniser = Tokeniser().load("./bellm/tokeniser/tokeniser.json")


@app.get("/")
async def root():
    return {"message": "Hello World"}


class TokeniserRequest(BaseModel):

    text: str


@app.post("/tokeniser")
async def tokenise(request: TokeniserRequest, max_tokens: int | None = None):
    tokenised = tokeniser.tokenize(request.text, max_tokens=max_tokens)

    return {
        "tokens": tokenised.tokens,
        "tokenIds": tokenised.token_ids,
    }
