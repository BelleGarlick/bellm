from pydantic import BaseModel


class BaseClpIModel(BaseModel):

    def run(self):
        raise NotImplementedError()
