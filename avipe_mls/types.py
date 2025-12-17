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

# Transforms and augmentations

class Crop(BaseModel):
    type: Literal["random_crop"]
    width: int
    height: int

class HorizontalFlip(BaseModel):
    type: Literal["horizontal_flip"]
    p: float = 0.5

class VerticalFlip(BaseModel):
    type: Literal["vertical_flip"]
    p: float = 0.5

class RandomRotate90(BaseModel):
    type: Literal["random_rotate_90"]
    p: float = 0.5

class ShiftScaleRotate(BaseModel):
    type: Literal["shift_scale_rotate"]
    shift_limit: float = 0.1
    scale_limit: float = 0.1
    rotate_limit: int = 45
    p: float = 0.5

class ColorJitter(BaseModel):
    type: Literal["color_jitter"]
    p: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.2

Transform = Union[Crop, HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, ColorJitter]

TRANSFORM_TYPE_MAP = {
    "RandomCrop": "random_crop",
    "HorizontalFlip": "horizontal_flip",
    "VerticalFlip": "vertical_flip",
    "RandomRotate90": "random_rotate_90",
    "ShiftScaleRotate": "shift_scale_rotate",
    "ColorJitter": "color_jitter",
}

class DatasetConfig(BaseConfigModel):
    source: DatasetSourceUnion
    num_classes: int
    labels: Optional[List[str]] = None
    post_download_scripts: List[str] = []
    images_path: str
    masks_path: str
    transforms: Optional[List[Transform]] = None

    @field_validator("transforms", mode="before")
    @classmethod
    def normalize_transforms(cls, v):
        """Convert - Crop:, - HorizontalFlip:, etc. into {'type': ...} format"""
        if not v:
            return v

        normalized = []
        for item in v:
            if isinstance(item, dict) and len(item) == 1:
                key, value = next(iter(item.items()))
                type_name = TRANSFORM_TYPE_MAP.get(key)
                if type_name is None:
                    raise ValueError(f"Unknown transform key: {key}")

                # If value is None, create dict with just type
                if value is None:
                    normalized.append({"type": type_name})
                else:
                    value["type"] = value.get("type", type_name)
                    normalized.append(value)
            else:
                normalized.append(item)
        return normalized

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

class TrainingConfig(BaseConfigModel):
    lr: float
    epochs: int
    validation_split: float = 0.2

class ModelTrainingConfig(BaseConfigModel):
    lr: Optional[float] = None
    epochs: Optional[int] = None
    validation_split: Optional[float] = None

class ModelConfig(BaseConfigModel):
    name: str = Field(min_length=1)
    backbone: str = Field(min_length=1)
    in_channels: int = 3
    input_size: List[int] = Field(max_length=2, min_length=2)
    training: Optional[ModelTrainingConfig] = None

class FullConfig(BaseConfigModel):
    name: str = Field(min_length=1)
    dataset: DatasetConfig
    model: List[ModelConfig]
    training: TrainingConfig

    def get_training_for_model(self, model_index: int) -> TrainingConfig:
        """
        Returns a TrainingConfig for the given model, 
        merging the global config with model-specific overrides.
        """
        model = self.model[model_index]

        if model.training is None:
            return self.training  # no overrides, return global

        # Merge field by field
        merged_data = self.training.model_dump()
        override_data = model.training.model_dump(exclude_unset=True)
        merged_data.update(override_data)
        return TrainingConfig(**merged_data)