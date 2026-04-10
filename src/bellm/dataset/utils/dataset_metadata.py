from typing import List

from pydantic import BaseModel, Field


class DatasetShardMetadata(BaseModel):
    uri: str
    length: int


class DatasetMetadata(BaseModel):
    id: str
    length: int = 0
    shards: List[DatasetShardMetadata] = Field(default_factory=list)
