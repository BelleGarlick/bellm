from pydantic import Field

from bellm.cli.common.dataset import DatasetConfig
from bellm.cli.common.model import ModelConfig
from clpi import BaseClpIModel


class FoundationModelTrainingConfig(BaseClpIModel):

    model: ModelConfig = Field(description="The model config to train")

    dataset: DatasetConfig = Field(description="The dataset config to train on")

    epochs: int = Field(default=1000, description="The number of epochs to train for")

    batch_size: int = Field(default=10, description="The training batch size")

    def run(self, *args, **kwargs):
        from bellm.training.foundational_model import train_foundational_model

        train_foundational_model(self)
