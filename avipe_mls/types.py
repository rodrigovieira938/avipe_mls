from pydantic import BaseModel, ValidationError, ConfigDict, Field, field_validator, ValidationInfo
from typing import Optional, Union, List, Literal

class BaseConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

class DatasetSourceConfig(BaseConfigModel):
    type: str  # discriminator for pydantic

class KaggleSource(DatasetSourceConfig):
    type: Literal["kagglehub"]
    competition: str = Field(min_length=1)

class HuggingFaceSource(DatasetSourceConfig):
    type: Literal["huggingface"]
    repo: str
    repo_type: Literal["dataset"] = "dataset"

DatasetSourceUnion = Union[KaggleSource, HuggingFaceSource]
class DatasetConfig(BaseConfigModel):
    source: DatasetSourceUnion
    num_classes: int
    labels: Optional[List[str]] = None
    post_download_scripts: List[str] = []
    images_path: str
    masks_path: str

    @field_validator('labels', mode='after')
    @classmethod
    def check_labels(cls, values, info: ValidationInfo):
        labels = values
        num_classes = info.data.get('num_classes')

        if labels is not None and len(labels) != num_classes:
            raise ValueError(
                f"If 'labels' are provided, there must be exactly {num_classes} labels "
                f"(got {len(labels)})"
            )
        return labels

class ModelConfig(BaseConfigModel):
    name: str = Field(min_length=1)
    backbone: str = Field(min_length=1)
    in_channels: int = 3
    input_size: List[int] = Field(max_length=2, min_length=2)

class TrainingConfig(BaseConfigModel):
    lr: float
    epochs: int

class FullConfig(BaseConfigModel):
    dataset: DatasetConfig
    model: List[ModelConfig]
    training: TrainingConfig