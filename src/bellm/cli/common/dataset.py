from pydantic import BaseModel


class DatasetConfig(BaseModel):

    path: str

    portion: float = 1
