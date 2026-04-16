from pydantic import Field, BaseModel


class ModelConfig(BaseModel):

    tokeniser: str

    layers: int = Field(default=6, description='Number of attention blocks')

    input_context_size: int = Field(default=1400, description='The size of the input context window')

    output_context_size: int = Field(default=100, description='The size of the output context window')

    embedding_dim: int = Field(default=512, description='Embedding dimension size')

    window: int = 500

    heads: int = Field(default=32, description='Number of attention heads')
